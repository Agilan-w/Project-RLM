"""
playback.py - High-Fidelity Showcase Script

Loads the best trained agent from mario_checkpoint.npz and plays World 1-1.
Records a smooth MP4 video with a HUD overlay.
"""
import os
import cv2
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import the Neural Network and Vision Extractor from main
from main import NeuralNet, GeneticAlgorithm, extract_vision_grid

def make_smooth_env():
    # We do NOT use FrameSkipEnv here because we want to capture every single frame 
    # for a smooth 60 FPS video recording.
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def record_champion():
    checkpoint_file = "mario_checkpoint.npz"
    if not os.path.exists(checkpoint_file):
        print(f"Error: {checkpoint_file} not found!")
        return

    print("Loading Champion AI...")
    ga = GeneticAlgorithm(population_size=100)
    # This automatically upgrades older checkpoints to the current architecture (64 hidden neurons)
    ga.load_checkpoint(checkpoint_file)
    
    # The best agent is always at index 0 after evolve()
    champion_net = ga.population[0]
    
    env = make_smooth_env()
    env.reset()
    done = False
    
    last_x, last_y = 40, 79
    dx, dy = 0, 0
    
    import cupy as cp
    
    # Video Writer setup (60 FPS, high-res canvas)
    # 256*3 for Mario, 512 for Vision Grid, 256 for Neural Net Vis
    canvas_w = 256 * 3 + 512 + 256 
    canvas_h = 240 * 3        # Mario Screen (3x scale) height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('champion_run.mp4', fourcc, 60.0, (canvas_w, canvas_h))
    
    print("Recording started. Press ESC to stop early.")
    
    # Initial prediction (matches training reset state)
    vision_flat = extract_vision_grid(env)
    vision_flat_258 = np.append(vision_flat, [0, 0])
    
    # Manual forward pass to get activations for visualization
    inputs_gpu = cp.array(vision_flat_258)
    z1 = cp.dot(inputs_gpu, champion_net.weights1) + champion_net.bias1
    a1_gpu = cp.maximum(0, z1)
    z2_gpu = cp.dot(a1_gpu, champion_net.weights2) + champion_net.bias2
    
    a1 = a1_gpu.get()
    z2 = z2_gpu.get()
    action = int(np.argmax(z2))
    
    vision_2d = vision_flat.reshape((16, 16))
    
    frame_counter = 1
    
    while not done:
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        curr_x = info.get('x_pos', 40)
        curr_y = info.get('y_pos', 79)
        
        # After completing 4 frames, evaluate the AI for the NEXT action
        if frame_counter % 4 == 0:
            dx = curr_x - last_x
            dy = curr_y - last_y
            last_x, last_y = curr_x, curr_y
            
            vision_flat = extract_vision_grid(env)
            vision_flat_258 = np.append(vision_flat, [dx, dy])
            
            inputs_gpu = cp.array(vision_flat_258)
            z1 = cp.dot(inputs_gpu, champion_net.weights1) + champion_net.bias1
            a1_gpu = cp.maximum(0, z1)
            z2_gpu = cp.dot(a1_gpu, champion_net.weights2) + champion_net.bias2
            
            a1 = a1_gpu.get()
            z2 = z2_gpu.get()
            action = int(np.argmax(z2))
            
            vision_2d = vision_flat.reshape((16, 16))
            
        frame_counter += 1
        
        # --- RENDER SEPARATE WINDOWS ---
        
        # 1. Mario Window
        mario_bgr = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        mario_h, mario_w, _ = mario_bgr.shape
        mario_scaled = cv2.resize(mario_bgr, (mario_w * 2, mario_h * 2), interpolation=cv2.INTER_NEAREST)
        
        # 2. Vision Grid Window (HUD)
        grid_size = 320
        vision_img = np.zeros((grid_size + 100, grid_size, 3), dtype=np.uint8)
        cell_size = grid_size // 16
        
        for r in range(16):
            for c in range(16):
                val = vision_2d[r, c]
                text = str(val)
                if val == 0: color = (30, 30, 30)       # Air
                elif val == 1: color = (255, 150, 50)   # Block
                elif val == -1: color = (0, 0, 255)     # Enemy
                elif val == 2: color = (255, 255, 255)  # Mario
                else: color = (100, 100, 100)
                    
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.3, 1)[0]
                text_x = c * cell_size + (cell_size - text_size[0]) // 2
                text_y = r * cell_size + (cell_size + text_size[1]) // 2
                
                cv2.rectangle(vision_img, (c * cell_size, r * cell_size), ((c+1) * cell_size, (r+1) * cell_size), (50, 50, 50), 1)
                cv2.putText(vision_img, text, (text_x, text_y), font, 0.3, color, 1)
                
        # Stats below vision grid
        stats_y = grid_size + 30
        cv2.putText(vision_img, f"Velocity X: {dx}", (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)
        cv2.putText(vision_img, f"Velocity Y: {dy}", (20, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)
        cv2.putText(vision_img, f"Score: {info.get('score', 0)}", (180, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vision_img, f"Dist: {curr_x}", (180, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 3. Neural Network Dense Graph Window
        nn_w, nn_h = 550, 420
        nn_img = np.zeros((nn_h, nn_w, 3), dtype=np.uint8)
        
        # Define layer X coordinates
        layer_x = [70, 250, 450] # Input (abstracted), Hidden, Output
        
        a1_max = np.max(a1) if np.max(a1) > 0 else 1
        a1_norm = a1 / a1_max
        
        z2_exp = np.exp(z2 - np.max(z2))
        z2_norm = z2_exp / np.sum(z2_exp)
        
        # Draw connections from Hidden -> Output first (so they are under the nodes)
        weights2 = champion_net.weights2.get() if hasattr(champion_net.weights2, 'get') else champion_net.weights2
        # Max weight for coloring lines
        w2_max = np.max(np.abs(weights2)) if np.max(np.abs(weights2)) > 0 else 1
        
        for h in range(64):
            # Calculate hidden node Y (2 columns of 32 for better fit)
            col = h % 2
            row = h // 2
            hx = layer_x[1] - 25 + (col * 50)
            hy = 30 + row * 11
            
            # Only draw lines if the hidden neuron is active to reduce visual clutter
            if a1_norm[h] > 0.1:
                for o in range(7):
                    oy = 90 + o * 40
                    ox = layer_x[2]
                    
                    weight = weights2[h, o]
                    # Color based on weight sign, brightness based on activation * weight
                    strength = abs(weight) / w2_max * a1_norm[h]
                    if strength > 0.2:  # Only draw significant connections
                        color = (50, 255, 50) if weight > 0 else (50, 50, 255) # BGR
                        thickness = max(1, int(strength * 3))
                        cv2.line(nn_img, (hx, hy), (ox, oy), color, thickness)
        
        # Draw Hidden Nodes
        cv2.putText(nn_img, "Hidden (64)", (layer_x[1]-45, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for h in range(64):
            col = h % 2
            row = h // 2
            hx = layer_x[1] - 25 + (col * 50)
            hy = 30 + row * 11
            intensity = int(a1_norm[h] * 255)
            cv2.circle(nn_img, (hx, hy), 4, (0, intensity, int(intensity*0.8)), -1)
            cv2.circle(nn_img, (hx, hy), 4, (100, 100, 100), 1)
            
        # Draw Output Nodes
        cv2.putText(nn_img, "Output (7)", (layer_x[2]-40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for o in range(7):
            oy = 90 + o * 40
            ox = layer_x[2]
            intensity = int(z2_norm[o] * 255)
            color = (50, 150, 255) if o == action else (intensity, 0, 0)
            
            cv2.circle(nn_img, (ox, oy), 10, color, -1)
            cv2.circle(nn_img, (ox, oy), 10, (200, 200, 200), 2)
            
            actions = ["NOOP", "RIGHT", "R+A", "R+B", "R+A+B", "A", "LEFT"]
            act_text = actions[o]
            cv2.putText(nn_img, act_text, (ox + 15, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Abstract Input Layer (since 258 nodes is too many, draw a block)
        cv2.putText(nn_img, "Input (258)", (layer_x[0]-45, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.rectangle(nn_img, (layer_x[0]-30, 150), (layer_x[0]+30, 250), (40, 40, 40), -1)
        cv2.putText(nn_img, "Grid", (layer_x[0]-15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(nn_img, "+ Vel", (layer_x[0]-18, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Connect Input block to Hidden layer (abstract lines)
        for h in range(0, 64, 4):
            col = h % 2
            row = h // 2
            hx = layer_x[1] - 25 + (col * 50)
            hy = 30 + row * 11
            cv2.line(nn_img, (layer_x[0]+30, 200), (hx, hy), (80, 80, 80), 1)
            
        # Show all 3 windows
        cv2.imshow("Super Mario Bros", mario_scaled)
        cv2.imshow("AI Vision Grid", vision_img)
        cv2.imshow("Neural Network Brain", nn_img)
        
        # Optional: Combine and write to video (scaled to fit)
        # We will stop recording to disk to keep it simple, or you can record the Mario window only.
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

    cv2.destroyAllWindows()
    env.close()
    print("Playback finished.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    record_champion()
