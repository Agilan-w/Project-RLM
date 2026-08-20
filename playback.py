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
    
    # Video Writer setup (60 FPS, high-res canvas)
    canvas_w = 256 * 3 + 512  # Mario Screen (3x scale) + HUD width
    canvas_h = 240 * 3        # Mario Screen (3x scale) height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('champion_run.mp4', fourcc, 60.0, (canvas_w, canvas_h))
    
    print("Recording started. Press ESC to stop early.")
    
    action = 0
    frame_counter = 0
    
    while not done:
        # The AI was trained with a frame skip of 4.
        # To replicate its exact behavior but get smooth video, we only ask 
        # the neural network for a new decision every 4th frame, and hold the 
        # button down for the intermediate frames.
        if frame_counter % 4 == 0:
            vision_flat = extract_vision_grid(env)
            vision_flat_258 = np.append(vision_flat, [dx, dy])
            action = champion_net.predict(vision_flat_258)
            vision_2d = vision_flat.reshape((16, 16))
            
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        curr_x = info.get('x_pos', 40)
        curr_y = info.get('y_pos', 79)
        # Update velocities only when the AI evaluates (every 4 frames)
        if frame_counter % 4 == 0:
            dx = curr_x - last_x
            dy = curr_y - last_y
            last_x, last_y = curr_x, curr_y
            
        frame_counter += 1
        
        # --- RENDER HUD ---
        # 1. Get raw Mario frame and scale it 3x
        mario_bgr = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        mario_h, mario_w, _ = mario_bgr.shape
        mario_scaled = cv2.resize(mario_bgr, (mario_w * 3, mario_h * 3), interpolation=cv2.INTER_NEAREST)
        
        # 2. Draw the Neural Network Vision Grid (HUD)
        hud = np.zeros((canvas_h, 512, 3), dtype=np.uint8)
        cell_size = 512 // 16
        
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
                text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                text_x = c * cell_size + (cell_size - text_size[0]) // 2
                text_y = r * cell_size + (cell_size + text_size[1]) // 2
                
                # Draw cell border
                cv2.rectangle(hud, (c * cell_size, r * cell_size), ((c+1) * cell_size, (r+1) * cell_size), (50, 50, 50), 1)
                cv2.putText(hud, text, (text_x, text_y), font, 0.4, color, 1)
                
        # 3. Draw Stats Panel below the grid
        stats_y = 512 + 30
        cv2.putText(hud, f"Velocity X (dx): {dx}", (20, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        cv2.putText(hud, f"Velocity Y (dy): {dy}", (20, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        cv2.putText(hud, f"Score: {info.get('score', 0)}", (20, stats_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hud, f"Distance: {curr_x} px", (20, stats_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        actions = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]
        action_name = actions[action] if 0 <= action < len(actions) else str(action)
        cv2.putText(hud, f"Action: {action_name}", (20, stats_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 150, 255), 2)
        
        # 4. Combine into final canvas
        canvas = np.hstack((mario_scaled, hud))
        
        # Write to video and show on screen
        out_video.write(canvas)
        cv2.imshow("Champion Replay", canvas)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

    out_video.release()
    cv2.destroyAllWindows()
    env.close()
    print("Recording saved as champion_run.mp4")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    record_champion()
