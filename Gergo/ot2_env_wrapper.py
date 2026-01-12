import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sim_class import Simulation


class OT2EnvSparse2D(gym.Env):
    """
    Modified OT-2 Environment with SPARSE REWARDS and DIFFERENT OBSERVATION SPACE.
    
    Key Differences:
    1. SPARSE REWARD: Only gives reward when reaching target (no dense distance reward)
    2. RELATIVE OBSERVATION: Uses relative position to goal instead of absolute
    3. INCLUDES ACCELERATION: Adds acceleration to observation space
    4. DIFFERENT SUCCESS ZONES: Multiple success thresholds with different rewards
    
    Observation (7D): [rel_x, rel_y, vel_x, vel_y, acc_x, acc_y, distance_to_goal]
    Action (2D): [velocity_x, velocity_y] continuous [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}
    
    # Workspace bounds
    PIPETTE_X_MIN, PIPETTE_X_MAX = -0.187, 0.253
    PIPETTE_Y_MIN, PIPETTE_Y_MAX = -0.1705, 0.2195
    MAX_VELOCITY = 1.0
    MAX_ACCELERATION = 2.0
    
    def __init__(self, render_mode=None, max_steps=500, normalize=True, fixed_z=0.125, 
                 reward_type="sparse", success_zones=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.normalize = normalize
        self.fixed_z = fixed_z
        self.reward_type = reward_type  # "sparse" or "shaped"
        self.steps = 0
        
        # Multiple success zones with different rewards
        if success_zones is None:
            self.success_zones = [
                {"threshold": 0.005, "reward": 10.0},   # 5mm - good
                {"threshold": 0.002, "reward": 50.0},   # 2mm - better
                {"threshold": 0.001, "reward": 100.0},  # 1mm - best
                {"threshold": 0.0005, "reward": 200.0}, # 0.5mm - perfect
            ]
        else:
            self.success_zones = success_zones
        
        # Initialize simulation
        self.sim = Simulation(
            num_agents=1,
            render=(render_mode == "human"),
            rgb_array=(render_mode == "rgb_array")
        )
        
        # Action space: 2D velocity
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Modified observation space (7D): relative position + velocity + acceleration + distance
        self._obs_low = np.array([
            -0.5, -0.5,  # relative position (max workspace diagonal ~0.5m)
            -self.MAX_VELOCITY, -self.MAX_VELOCITY,  # velocity
            -self.MAX_ACCELERATION, -self.MAX_ACCELERATION,  # acceleration
            0.0  # distance to goal
        ], dtype=np.float32)
        
        self._obs_high = np.array([
            0.5, 0.5,  # relative position
            self.MAX_VELOCITY, self.MAX_VELOCITY,  # velocity
            self.MAX_ACCELERATION, self.MAX_ACCELERATION,  # acceleration
            0.7  # max distance (workspace diagonal)
        ], dtype=np.float32)
        
        if self.normalize:
            self.observation_space = spaces.Box(
                low=-np.ones(7, dtype=np.float32),
                high=np.ones(7, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=self._obs_low,
                high=self._obs_high,
                dtype=np.float32
            )
        
        # Goal and tracking variables
        self.goal_position = np.zeros(2, dtype=np.float32)
        self.prev_velocity = np.zeros(2, dtype=np.float32)
        self.prev_distance = 0.0
        self.velocity_scale = 0.5
        
        # Statistics tracking
        self.episode_stats = {
            "total_distance_traveled": 0.0,
            "time_in_zones": {zone["threshold"]: 0 for zone in self.success_zones}
        }
    
    def _normalize_obs(self, obs):
        """Normalize observation to [-1, 1] range."""
        return 2.0 * (obs - self._obs_low) / (self._obs_high - self._obs_low) - 1.0
    
    def _move_to_fixed_z(self):
        """Move pipette to fixed Z height."""
        robot_id = next(iter(self.sim.get_states()))
        
        for _ in range(500):
            states = self.sim.get_states()
            current_z = states[robot_id]['pipette_position'][2]
            
            z_error = self.fixed_z - current_z
            if abs(z_error) < 0.001:
                break
            
            z_velocity = np.clip(z_error * 10.0, -1.0, 1.0) * self.velocity_scale
            sim_action = [[0.0, 0.0, float(z_velocity), 0]]
            self.sim.run(sim_action, num_steps=1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set goal position
        if options and "goal" in options:
            goal = np.array(options["goal"], dtype=np.float32)
            self.goal_position = goal[:2]
        else:
            # Curriculum learning: start with closer goals, expand over time
            range_scale = 1.0  # Can be modified for curriculum
            self.goal_position = self.np_random.uniform(
                low=[0.10, 0.05],
                high=[0.25 * range_scale, 0.21 * range_scale],
            ).astype(np.float32)
        
        # Reset simulation
        self.sim.reset(num_agents=1)
        self._move_to_fixed_z()
        
        # Get initial state
        sim_observation = self.sim.get_states()
        robot_id = next(iter(sim_observation))
        pipette_pos = np.array(sim_observation[robot_id]['pipette_position'], dtype=np.float32)
        
        # Get velocity
        joint_states = sim_observation[robot_id]['joint_states']
        velocity = np.array([
            -joint_states['joint_0']['velocity'],
            -joint_states['joint_1']['velocity'],
        ], dtype=np.float32)
        
        # Calculate relative position and distance
        relative_pos = self.goal_position - pipette_pos[:2]
        distance = np.linalg.norm(relative_pos)
        
        # Initial acceleration is zero
        acceleration = np.zeros(2, dtype=np.float32)
        
        # Build observation
        observation = np.concatenate([
            relative_pos,     # Relative position to goal
            velocity,         # Current velocity
            acceleration,     # Current acceleration (0 at start)
            [distance]        # Distance to goal
        ]).astype(np.float32)
        
        # Clip and normalize
        observation = np.clip(observation, self._obs_low, self._obs_high)
        if self.normalize:
            observation = self._normalize_obs(observation)
        
        # Initialize tracking variables
        self.prev_velocity = velocity.copy()
        self.prev_distance = distance
        self.steps = 0
        
        # Reset episode stats
        self.episode_stats = {
            "total_distance_traveled": 0.0,
            "time_in_zones": {zone["threshold"]: 0 for zone in self.success_zones}
        }
        
        info = {
            "distance_to_target": distance,
            "pipette_position": pipette_pos.copy(),
            "goal_position": self.goal_position.copy(),
            "fixed_z": self.fixed_z,
        }
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Scale action to velocity
        scaled_action = np.array(action, dtype=np.float32) * self.velocity_scale
        
        # Run simulation
        sim_action = [[float(scaled_action[0]), float(scaled_action[1]), 0.0, 0]]
        sim_observation = self.sim.run(sim_action)
        
        # Get state
        robot_id = next(iter(sim_observation))
        pipette_pos = np.array(sim_observation[robot_id]['pipette_position'], dtype=np.float32)
        
        # Get velocity
        joint_states = sim_observation[robot_id]['joint_states']
        velocity = np.array([
            -joint_states['joint_0']['velocity'],
            -joint_states['joint_1']['velocity'],
        ], dtype=np.float32)
        
        # Calculate acceleration
        acceleration = (velocity - self.prev_velocity) * 240.0  # 240 Hz simulation
        acceleration = np.clip(acceleration, -self.MAX_ACCELERATION, self.MAX_ACCELERATION)
        
        # Calculate relative position and distance
        relative_pos = self.goal_position - pipette_pos[:2]
        distance = np.linalg.norm(relative_pos)
        
        # Build observation
        observation = np.concatenate([
            relative_pos,     # Relative position
            velocity,         # Velocity
            acceleration,     # Acceleration
            [distance]        # Distance
        ]).astype(np.float32)
        
        # Clip and normalize
        observation = np.clip(observation, self._obs_low, self._obs_high)
        if self.normalize:
            observation = self._normalize_obs(observation)
        
        # ==================== SPARSE REWARD FUNCTION ====================
        reward = 0.0
        terminated = False
        
        if self.reward_type == "sparse":
            # Only reward at goal - no intermediate rewards
            for zone in sorted(self.success_zones, key=lambda x: x["threshold"]):
                if distance < zone["threshold"]:
                    reward = zone["reward"]
                    if zone["threshold"] <= 0.001:  # Terminate at 1mm
                        terminated = True
                    break
            
            # Small penalty for time
            reward -= 0.1
            
        else:  # "shaped" reward as alternative
            # Shaped reward with multiple components
            progress_reward = (self.prev_distance - distance) * 50.0
            
            # Zone bonus
            zone_bonus = 0.0
            for zone in self.success_zones:
                if distance < zone["threshold"]:
                    zone_bonus = zone["reward"] / 10.0  # Smaller continuous bonus
                    if zone["threshold"] <= 0.001:
                        terminated = True
                    break
            
            # Smoothness penalty (high acceleration = bad)
            smoothness_penalty = -np.linalg.norm(acceleration) * 0.1
            
            # Time penalty
            time_penalty = -0.1
            
            reward = progress_reward + zone_bonus + smoothness_penalty + time_penalty
        # ================================================================
        
        # Update tracking
        self.prev_velocity = velocity.copy()
        self.prev_distance = distance
        
        # Track statistics
        self.episode_stats["total_distance_traveled"] += np.linalg.norm(velocity) / 240.0
        for zone in self.success_zones:
            if distance < zone["threshold"]:
                self.episode_stats["time_in_zones"][zone["threshold"]] += 1
        
        # Check truncation
        truncated = self.steps >= self.max_steps
        
        info = {
            "success": terminated,
            "distance_to_target": distance,
            "pipette_position": pipette_pos.copy(),
            "goal_position": self.goal_position.copy(),
            "fixed_z": self.fixed_z,
            "episode_stats": self.episode_stats.copy() if (terminated or truncated) else None,
            "reward_components": {
                "base_reward": reward,
                "current_zone": min([z["threshold"] for z in self.success_zones if distance < z["threshold"]], default=None)
            }
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self.sim.current_frame
        return None
    
    def close(self):
        if hasattr(self.sim, 'close'):
            self.sim.close()


# Test the environment
if __name__ == "__main__":
    print("="*60)
    print("Testing OT2EnvSparse2D (Modified Environment)")
    print("="*60)
    
    # Test sparse reward version
    env = OT2EnvSparse2D(render_mode=None, normalize=True, reward_type="sparse")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space} (7D)")
    print(f"Reward type: SPARSE")
    print(f"Success zones: {env.success_zones}")
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Goal: {info['goal_position']}")
    
    # Run a few steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: distance={info['distance_to_target']*1000:.2f}mm, reward={reward:.2f}")
        if terminated:
            break
    
    env.close()
    print("\nDone!")