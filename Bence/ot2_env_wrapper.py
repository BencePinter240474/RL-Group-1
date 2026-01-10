import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# Change working directory to the location of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sim_class import Simulation


class OT2Env2D(gym.Env):
    """
    Gymnasium wrapper for the OT-2 PyBullet simulation.
    2D VERSION: Z is fixed, only X and Y are controlled.
    
    Observation (6D): [pipette_x, pipette_y, vel_x, vel_y, goal_x, goal_y]
    Action (2D): [velocity_x, velocity_y] continuous [-1, 1]
    
    Workspace bounds (from URDF):
        X: [-0.187, 0.253]
        Y: [-0.1705, 0.2195]
        Z: FIXED at 0.125 (or configurable)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}
    
    # Workspace bounds from URDF joint limits + pipette offset
    PIPETTE_X_MIN, PIPETTE_X_MAX = -0.187, 0.253
    PIPETTE_Y_MIN, PIPETTE_Y_MAX = -0.1705, 0.2195
    PIPETTE_Z_MIN, PIPETTE_Z_MAX = 0.1250, 0.2895
    
    # Velocity bounds
    MAX_VELOCITY = 1.0
    
    def __init__(self, render_mode=None, max_steps=1000, normalize=True, fixed_z=0.125):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.normalize = normalize
        self.fixed_z = fixed_z  # Fixed Z height
        self.steps = 0
        
        # Initialize simulation
        self.sim = Simulation(
            num_agents=1,
            render=(render_mode == "human"),
            rgb_array=(render_mode == "rgb_array")
        )
        
        # Action space: 2D velocity commands [-1, 1] for X and Y only
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space bounds (6D: pos_x, pos_y, vel_x, vel_y, goal_x, goal_y)
        self._obs_low = np.array([
            self.PIPETTE_X_MIN, self.PIPETTE_Y_MIN,
            -self.MAX_VELOCITY, -self.MAX_VELOCITY,
            self.PIPETTE_X_MIN, self.PIPETTE_Y_MIN
        ], dtype=np.float32)
        
        self._obs_high = np.array([
            self.PIPETTE_X_MAX, self.PIPETTE_Y_MAX,
            self.MAX_VELOCITY, self.MAX_VELOCITY,
            self.PIPETTE_X_MAX, self.PIPETTE_Y_MAX
        ], dtype=np.float32)
        
        # Observation space (normalized to [-1, 1] if normalize=True)
        if self.normalize:
            self.observation_space = spaces.Box(
                low=-np.ones(6, dtype=np.float32),
                high=np.ones(6, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=self._obs_low,
                high=self._obs_high,
                dtype=np.float32
            )
        
        # Goal position (only X, Y - Z is fixed)
        self.goal_position = np.zeros(2, dtype=np.float32)
        
        # Previous distance for progress reward
        self.prev_distance = 0.0
        
        # Velocity scaling
        self.velocity_scale = 0.5
        
        # Flag to track if Z has been set
        self._z_initialized = False
    
    def _normalize_obs(self, obs):
        """Normalize observation to [-1, 1] range."""
        return 2.0 * (obs - self._obs_low) / (self._obs_high - self._obs_low) - 1.0
    
    def _denormalize_obs(self, obs_normalized):
        """Convert normalized observation back to original range."""
        return (obs_normalized + 1.0) / 2.0 * (self._obs_high - self._obs_low) + self._obs_low
    
    def _move_to_fixed_z(self):
        """Move the pipette to the fixed Z height at the start of each episode."""
        robot_id = next(iter(self.sim.get_states()))
        
        for _ in range(500):  # Max iterations to reach Z
            states = self.sim.get_states()
            current_z = states[robot_id]['pipette_position'][2]
            
            z_error = self.fixed_z - current_z
            
            if abs(z_error) < 0.001:  # Within 1mm
                break
            
            # Move only in Z direction
            z_velocity = np.clip(z_error * 10.0, -1.0, 1.0) * self.velocity_scale
            sim_action = [[0.0, 0.0, float(z_velocity), 0]]
            self.sim.run(sim_action, num_steps=1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate random goal position (X, Y only)
        if options and "goal" in options:
            goal = np.array(options["goal"], dtype=np.float32)
            self.goal_position = goal[:2]  # Only take X, Y
        else:
            self.goal_position = self.np_random.uniform(
                low=[-0.15, -0.15],
                high=[0.15, 0.15],
            ).astype(np.float32)
        
        # Reset simulation
        self.sim.reset(num_agents=1)
        
        # Move to fixed Z height
        self._move_to_fixed_z()
        
        # Get state after Z adjustment
        sim_observation = self.sim.get_states()
        robot_id = next(iter(sim_observation))
        pipette_pos = np.array(sim_observation[robot_id]['pipette_position'], dtype=np.float32)
        
        # Get velocity from joint states
        joint_states = sim_observation[robot_id]['joint_states']
        velocity = np.array([
            -joint_states['joint_0']['velocity'],
            -joint_states['joint_1']['velocity'],
        ], dtype=np.float32)
        
        # Build 6D observation (X, Y only)
        observation = np.concatenate([
            pipette_pos[:2],  # X, Y position
            velocity,          # X, Y velocity
            self.goal_position # X, Y goal
        ]).astype(np.float32)
        
        # Clip to bounds
        observation = np.clip(observation, self._obs_low, self._obs_high)
        
        # Normalize if enabled
        if self.normalize:
            observation = self._normalize_obs(observation)
        
        # Initialize previous distance for progress reward (2D distance)
        self.prev_distance = np.linalg.norm(pipette_pos[:2] - self.goal_position)
        
        self.steps = 0
        info = {
            "distance_to_target": self.prev_distance,
            "pipette_position": pipette_pos.copy(),
            "goal_position": self.goal_position.copy(),
            "fixed_z": self.fixed_z,
        }
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Scale action to velocity (only X, Y)
        scaled_action = np.array(action, dtype=np.float32) * self.velocity_scale
        
        # Format action for simulation: [vx, vy, vz=0, drop=0]
        # Z velocity is 0 to maintain fixed height
        sim_action = [[float(scaled_action[0]), float(scaled_action[1]), 0.0, 0]]
        
        # Run simulation
        sim_observation = self.sim.run(sim_action)
        
        # Get pipette position
        robot_id = next(iter(sim_observation))
        pipette_pos = np.array(sim_observation[robot_id]['pipette_position'], dtype=np.float32)
        
        # Get velocity from joint states (X, Y only)
        joint_states = sim_observation[robot_id]['joint_states']
        velocity = np.array([
            -joint_states['joint_0']['velocity'],
            -joint_states['joint_1']['velocity'],
        ], dtype=np.float32)
        
        # Build 6D observation
        observation = np.concatenate([
            pipette_pos[:2],   # X, Y position
            velocity,           # X, Y velocity
            self.goal_position  # X, Y goal
        ]).astype(np.float32)
        
        # Clip to bounds
        observation = np.clip(observation, self._obs_low, self._obs_high)
        
        # Normalize if enabled
        if self.normalize:
            observation = self._normalize_obs(observation)
        
        # Calculate current distance (2D only)
        current_distance = np.linalg.norm(pipette_pos[:2] - self.goal_position)
        
        # ==================== REWARD FUNCTION ====================
        # 1. Progress reward
        progress_reward = (self.prev_distance - current_distance) * 10.0
        
        # 2. Exponential distance penalty (adjusted for 0.5mm target)
        if current_distance < 0.01:  # Within 10mm
            distance_penalty = -np.exp(current_distance * 200) + 1
        else:
            distance_penalty = -current_distance * 0.1
        
        # 3. Success bonus (for 0.5mm precision)
        success_bonus = 0.0
        if current_distance < 0.0005:  # 0.5mm threshold
            success_bonus = 100.0
        
        # 4. Precision bonus tiers
        precision_bonus = 0.0
        if current_distance < 0.002:  # Under 2mm
            precision_bonus = 5.0
        if current_distance < 0.001:  # Under 1mm
            precision_bonus = 15.0
        if current_distance < 0.0005:  # Under 0.5mm
            precision_bonus = 25.0
        
        # Combine all reward components
        reward = float(progress_reward + distance_penalty + success_bonus + precision_bonus)
        # =========================================================
        
        # Update previous distance
        self.prev_distance = current_distance
        
        # Check termination (success at 0.5mm)
        terminated = current_distance < 0.0005
        
        # Check truncation (time limit)
        truncated = self.steps >= self.max_steps
        
        info = {
            "success": terminated,
            "distance_to_target": current_distance,
            "pipette_position": pipette_pos.copy(),
            "goal_position": self.goal_position.copy(),
            "fixed_z": self.fixed_z,
            "reward_components": {
                "progress": progress_reward,
                "distance_penalty": distance_penalty,
                "success_bonus": success_bonus,
                "precision_bonus": precision_bonus,
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