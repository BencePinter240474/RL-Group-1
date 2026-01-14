"""Simple test script for OT2Env wrapper."""
from ot2_env_wrapper import OT2Env

env = OT2Env(render_mode=None)
obs, info = env.reset()

# Running the environment for 1000 steps
for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Test completed successfully.")