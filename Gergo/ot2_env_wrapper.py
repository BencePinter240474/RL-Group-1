import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        # If render=False, this starts in "HEADLESS" mode (Fast, no window)
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.steps = 0
        self.prev_distance = None

    def reset(self, seed=None):
        # 1. Initialize the internal RNG (Random Number Generator)
        super().reset(seed=seed)

        # 2. Use self.np_random instead of np.random
        # This ensures that if you set a seed, the goal is always the same
        self.goal_position = self.np_random.uniform(
            low=[-0.15, -0.15, 0.19],
            high=[0.15, 0.15, 0.24],
            size=3
        ).astype(np.float32)

        # 3. Reset Simulation
        observation = self.sim.reset(num_agents=1)

        # 4. Process Observation
        robot_id = next(iter(observation))
        pipette_pos = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate((pipette_pos, self.goal_position)).astype(np.float32)

        # Initialize previous distance for progress reward
        self.prev_distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        self.steps = 0
        info = {}
        return observation, info

    def step(self, action):
        # Append 0 for the drop action
        action = np.append(action, 0)

        # Run Simulation
        observation = self.sim.run([action])
        robot_id = next(iter(observation))
        pipette_pos = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)

        # Create Observation
        observation = np.concatenate((pipette_pos, self.goal_position)).astype(np.float32)

        # Calculate current distance
        current_distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        # MULTI-COMPONENT REWARD FUNCTION
        # 1. Progress reward (most important) - positive when getting closer
        progress_reward = (self.prev_distance - current_distance) * 10.0  # Scale up for importance
        
        # 2. Distance penalty (scaled down) - small continuous penalty
        distance_penalty = -current_distance * 0.1  # Scaled down to not dominate
        
        # 3. Success bonus - significant reward when reaching goal
        success_bonus = 0.0
        if current_distance < 0.001:
            success_bonus = 50.0  # Large bonus for success
        
        # Combine all reward components
        reward = float(progress_reward + distance_penalty + success_bonus)
        
        # Update previous distance for next step
        self.prev_distance = current_distance

        # Check Termination (Success)
        terminated = False
        if current_distance < 0.001:
            terminated = True
            info = {'success': True}
        else:
            info = {'success': False}

        # Check Truncation (Time limit)
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        if hasattr(self.sim, 'close'):
            self.sim.close()