"""
=============================================================================
 Super Mario Bros — Neuroevolution AI (Phase 1: Environment Loop)
=============================================================================
 This script scaffolds the game environment and runs a random-action agent
 so we can visually inspect the emulator and examine the raw data coming
 out of env.step().

 Phase 2 will replace the random agent with a custom NumPy neural network
 that is evolved via a genetic algorithm (no PyTorch / TensorFlow).
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────

# The gym-super-mario-bros package registers the Mario environments with Gym
# when imported.  We only need the make() wrapper from the top-level module.
import gym_super_mario_bros

# JoypadSpace wraps the raw 256-button NES controller into a small, discrete
# action space.  SIMPLE_MOVEMENT gives us 7 intuitive combos:
#   0: NOOP
#   1: right
#   2: right + A (run right)
#   3: right + B (jump right)
#   4: right + A + B (run-jump right)
#   5: A (jump)
#   6: left
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import numpy as np
import cupy as cp
import random


# ── Environment Setup ────────────────────────────────────────────────────────

def make_env():
    """
    Create and return the wrapped Super Mario Bros environment.

    Returns
    -------
    env : gym.Env
        The Mario environment with a 7-action discrete action space.
    """
    # Create the raw NES environment for World 1-1.
    # The observation is an (240, 256, 3) RGB numpy array — a raw frame
    # from the NES PPU at 256×240 resolution.
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array')

    # Wrap the controller so the action space shrinks from 256 → 7.
    # This makes learning dramatically easier for our neural network later.
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    return env


# ── Data Inspection Helper ───────────────────────────────────────────────────

def inspect_step_output(state, reward, terminated, truncated, info):
    """
    Print a human-readable summary of a single env.step() return value.
    Called exactly once so we can understand the data shapes before we
    start feeding frames into a neural network.

    Parameters
    ----------
    state      : np.ndarray  – The RGB pixel array for the current frame.
    reward     : float        – The reward signal from the environment.
    terminated : bool         – True if Mario died or cleared the level.
    truncated  : bool         – True if the episode hit a time limit.
    info       : dict         – RAM-derived metadata (coins, score, position…).
    """
    done = terminated or truncated
    print("\n" + "=" * 65)
    print("  ENV.STEP() -- RAW OUTPUT INSPECTION")
    print("=" * 65)

    print(f"\n  state      | dtype: {state.dtype}  shape: {state.shape}")
    print(f"               {state.shape[0]}h x {state.shape[1]}w x {state.shape[2]}ch  RGB frame")

    print(f"\n  reward     | {reward}")
    print(f"               Positive = rightward progress, negative = death/time")

    print(f"\n  terminated | {terminated}")
    print(f"  truncated  | {truncated}")
    print(f"  done       | {done}  (terminated or truncated)")
    print(f"               True when Mario dies or clears the level")

    print(f"\n  info       | keys: {list(info.keys())}")
    for key, value in info.items():
        print(f"               {key:20s} = {value}")

    print("\n" + "=" * 65 + "\n")

# ── Vision Grid Extractor ────────────────────────────────────────────────────

def extract_vision_grid(env):
    """
    Reads the raw NES RAM to build a 16x16 vision grid centered on Mario.
    Returns a flattened 1D numpy array of 256 integers.
    """
    ram = env.unwrapped.ram
    
    # 1. Parse Mario's pixel position from RAM
    mario_x = ram[0x6D] * 256 + ram[0x86]
    mario_y = ram[0x03B8]
    
    # 2. Convert Mario's pixel position to tilemap block coordinates.
    mario_col = mario_x // 16
    mario_row = (mario_y + 16 - 32) // 16
    
    # Initialize a 16x16 grid (default to 0: empty space)
    grid = np.zeros((16, 16), dtype=np.int8)
    
    # 3. Populate the static level architecture (blocks, pipes, ground)
    for dy in range(-8, 8):
        for dx in range(-8, 8):
            row = mario_row + dy
            col = mario_col + dx
            val = 0
            
            if 0 <= row < 13 and col >= 0:
                page = (col // 16) % 2
                subx = col % 16
                addr = 0x500 + page * 208 + row * 16 + subx
                if ram[addr] != 0:
                    val = 1
            
            grid[dy + 8, dx + 8] = val

    # 4. Populate enemies (Goombas, Koopas)
    for i in range(5):
        if ram[0x0F + i] != 0: 
            ex = ram[0x6E + i] * 256 + ram[0x87 + i]
            ey = ram[0xCF + i]
            
            ecol = ex // 16
            erow = (ey + 16 - 32) // 16
            
            gy = erow - mario_row + 8
            gx = ecol - mario_col + 8
            
            if 0 <= gy < 16 and 0 <= gx < 16:
                grid[gy, gx] = -1

    # 5. Place Mario exactly in the center of the grid
    grid[8, 8] = 2
            
    return grid.flatten()


# ── Neural Network (Phase 3) ─────────────────────────────────────────────────

class NeuralNet:
    """
    A simple pure-CuPy Feedforward Neural Network.
    No PyTorch, TensorFlow, or Keras.
    """
    def __init__(self, input_size=256, hidden_size=18, output_size=7):
        # Initialize weights and biases with random values between -1 and 1
        self.weights1 = cp.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias1 = cp.random.uniform(-1, 1, hidden_size)
        
        self.weights2 = cp.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias2 = cp.random.uniform(-1, 1, output_size)
        
    def relu(self, x):
        # ReLU activation function: max(0, x)
        return cp.maximum(0, x)
        
    def predict(self, inputs):
        # inputs is a numpy array from extract_vision_grid. Transfer to GPU.
        inputs_gpu = cp.array(inputs)
        
        # Forward pass through the hidden layer
        z1 = cp.dot(inputs_gpu, self.weights1) + self.bias1
        a1 = self.relu(z1)
        
        # Forward pass through the output layer
        z2 = cp.dot(a1, self.weights2) + self.bias2
        
        # Return the index (0-6) of the highest output score as a standard int
        return int(cp.argmax(z2).get())


# ── Genetic Algorithm (Phase 4) ──────────────────────────────────────────────

class GeneticAlgorithm:
    def __init__(self, population_size=100):
        self.population_size = population_size
        # Generate initial population with random weights
        self.population = [NeuralNet() for _ in range(population_size)]
        
    def evaluate_fitness(self, network, env):
        env.reset()
        done = False
        
        max_x_pos = 0
        frames_since_progress = 0
        
        vision_flat = extract_vision_grid(env)
        
        while not done:
            action = network.predict(vision_flat)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            vision_flat = extract_vision_grid(env)
            
            current_x = info.get('x_pos', 0)
            if current_x > max_x_pos:
                max_x_pos = current_x
                frames_since_progress = 0
            else:
                frames_since_progress += 1
                
            # Timeout feature: Break if stuck for 50 frames
            if frames_since_progress >= 50:
                break
                
        return max_x_pos

    def breed(self, parent1, parent2):
        child = NeuralNet()
        
        # 50% chance to inherit from parent 1 or 2
        mask_w1 = cp.random.rand(*child.weights1.shape) > 0.5
        child.weights1 = cp.where(mask_w1, parent1.weights1, parent2.weights1)
        
        mask_b1 = cp.random.rand(*child.bias1.shape) > 0.5
        child.bias1 = cp.where(mask_b1, parent1.bias1, parent2.bias1)
        
        mask_w2 = cp.random.rand(*child.weights2.shape) > 0.5
        child.weights2 = cp.where(mask_w2, parent1.weights2, parent2.weights2)
        
        mask_b2 = cp.random.rand(*child.bias2.shape) > 0.5
        child.bias2 = cp.where(mask_b2, parent1.bias2, parent2.bias2)
        
        return child
        
    def mutate(self, network, mutation_rate=0.05):
        # 5% chance to multiply weight/bias by random factor between 0.5 and 1.5
        def apply_mutation(matrix):
            mask = cp.random.rand(*matrix.shape) < mutation_rate
            scale = cp.random.uniform(0.5, 1.5, matrix.shape)
            return cp.where(mask, matrix * scale, matrix)
            
        network.weights1 = apply_mutation(network.weights1)
        network.bias1 = apply_mutation(network.bias1)
        network.weights2 = apply_mutation(network.weights2)
        network.bias2 = apply_mutation(network.bias2)

    def evolve(self, env):
        # Evaluate fitness for the entire population
        fitness_scores = []
        for network in self.population:
            score = self.evaluate_fitness(network, env)
            fitness_scores.append((score, network))
            
        # Sort descending by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        best_fitness = fitness_scores[0][0]
        
        # Elitism: keep top 10% exactly as they are
        elite_count = int(self.population_size * 0.1)
        elites = [item[1] for item in fitness_scores[:elite_count]]
        
        new_population = list(elites)
        
        # Fill remaining 90% by breeding random elite parents and mutating offspring
        while len(new_population) < self.population_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            
            child = self.breed(p1, p2)
            self.mutate(child)
            new_population.append(child)
            
        self.population = new_population
        return best_fitness


# ── Main Loop ────────────────────────────────────────────────────────────────

def watch_agent(network, env):
    """
    Helper function to visually watch a specific agent play using OpenCV.
    Extracted from Phase 3 loop.
    """
    env.reset()
    done = False
    vision_flat = extract_vision_grid(env)
    
    while not done:
        action = network.predict(vision_flat)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        mario_frame = env.render()
        mario_bgr = cv2.cvtColor(mario_frame, cv2.COLOR_RGB2BGR)
        
        vision_flat = extract_vision_grid(env)
        vision_2d = vision_flat.reshape((16, 16))
        
        grid_size = 512
        grid_canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        cell_size = grid_size // 16
        
        for r in range(16):
            for c in range(16):
                val = vision_2d[r, c]
                text = str(val)
                if val == 0: color = (100, 100, 100)
                elif val == 1: color = (255, 200, 100)
                elif val == -1: color = (0, 0, 255)
                elif val == 2: color = (255, 255, 255)
                else: color = (255, 255, 255)
                    
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
                text_x = c * cell_size + (cell_size - text_size[0]) // 2
                text_y = r * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(grid_canvas, text, (text_x, text_y), font, 0.5, color, 1)
        
        h, w, _ = mario_bgr.shape
        scale = grid_size / h
        mario_bgr_resized = cv2.resize(mario_bgr, (int(w * scale), grid_size), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Mario Gameplay", mario_bgr_resized)
        cv2.imshow("AI Vision Grid", grid_canvas)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cv2.destroyAllWindows()


def run():
    print("[Phase 4]  Neuroevolution Training Started")
    
    env = make_env()
    ga = GeneticAlgorithm(population_size=100)
    
    for generation in range(1, 501):
        best_fitness = ga.evolve(env)
        print(f"Generation {generation:3d} | Best Fitness (Max X-Pos): {best_fitness}")
        
    env.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
