import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state = env.reset()

for _ in range(100):
    state, r, term, trunc, info = env.step(1)  # move right

def extract_vision_grid(info, env):
    # We must read from the raw RAM because info doesn't contain tile/enemy data
    ram = env.unwrapped.ram
    
    # Mario's pixel position
    mario_x = ram[0x6D] * 256 + ram[0x86]
    mario_y = ram[0x03B8]
    
    # Convert Mario's pixel position to tilemap block coordinates.
    # Tilemap starts 32 pixels down (menu bar). Mario's Y is 176 on the ground.
    # Ground is at suby=11 and 12. Mario occupies suby=10.
    # So: (176 + 16 - 32) // 16 = 160 // 16 = 10.
    mario_col = mario_x // 16
    mario_row = (mario_y + 16 - 32) // 16
    
    grid = np.zeros((16, 16), dtype=np.int8)
    
    for dy in range(-8, 8):
        for dx in range(-8, 8):
            row = mario_row + dy
            col = mario_col + dx
            val = 0
            
            # Check tilemap
            if 0 <= row < 13 and col >= 0:
                page = (col // 16) % 2
                subx = col % 16
                addr = 0x500 + page * 208 + row * 16 + subx
                if ram[addr] != 0:
                    val = 1
            
            grid[dy + 8, dx + 8] = val

    # Check enemies (they are placed based on pixel coords)
    for i in range(5):
        if ram[0x0F + i] != 0:
            ex = ram[0x6E + i] * 256 + ram[0x87 + i]
            ey = ram[0xCF + i]
            ecol = ex // 16
            erow = (ey + 16 - 32) // 16
            
            # Calculate grid index for the enemy
            gy = erow - mario_row + 8
            gx = ecol - mario_col + 8
            
            if 0 <= gy < 16 and 0 <= gx < 16:
                grid[gy, gx] = -1

    # Place Mario exactly in the center
    grid[8, 8] = 2
            
    return grid.flatten()

grid = extract_vision_grid(None, env).reshape((16, 16))
for r in range(16):
    print(" ".join(f"{grid[r, c]:2}" for c in range(16)))

env.close()
