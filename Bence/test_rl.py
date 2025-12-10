import numpy as np
import os
from stable_baselines3 import PPO
from ot2_env_wrapper import OT2Env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate_agent(model_path, vec_norm_path, num_episodes=50):
    """
    Runs the agent with normalization enabled.
    """
    # 1. Create the base environment
    env = DummyVecEnv([lambda: OT2Env(render=False)])

    # 2. Load the normalization statistics
    # This is crucial! The model expects inputs normalized exactly like during training.
    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"Could not find normalization stats at {vec_norm_path}. Did you save them?")

    env = VecNormalize.load(vec_norm_path, env)

    # 3. Disable training and reward normalization for evaluation
    # We want the stats (mean/var) to be frozen, and we want to see raw rewards.
    env.training = False
    env.norm_reward = False

    # 4. Load Model
    model = PPO.load(model_path)

    # Metrics storage
    success_count = 0
    total_rewards = []
    steps_taken = []
    final_distances = []

    print(f"=== Starting Evaluation for {num_episodes} Episodes ===")

    for i in range(num_episodes):
        # Reset returns (n_envs, obs_dim), so shape is (1, 6)
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Model predicts action based on normalized observation
            action, _ = model.predict(obs, deterministic=True)

            # Step returns arrays for all environments: obs, rewards, dones, infos
            obs, reward, terminated, info = env.step(action)

            # Accumulate reward (scalar value from the first env)
            episode_reward += reward[0]
            steps += 1

            # Check if the episode ended (terminated is an array of booleans)
            if terminated[0]:
                # === CRITICAL FIX FOR DISTANCE CALCULATION ===
                # When using VecEnv, 'obs' is already the reset observation for the next episode.
                # The terminal observation (the state where the agent stopped) is in the info dict.
                terminal_obs = info[0]['terminal_observation']

                # We must UN-NORMALIZE to get real world coordinates (meters)
                real_obs = env.unnormalize_obs(terminal_obs)

                # Now extract positions from the real observation
                pipette_pos = real_obs[:3]
                goal_pos = real_obs[3:]

                dist = np.linalg.norm(pipette_pos - goal_pos)
                final_distances.append(dist)

                # Define success threshold (e.g., 1mm)
                if dist < 0.001:
                    success_count += 1

                total_rewards.append(episode_reward)
                steps_taken.append(steps)
                done = True

    env.close()

    # --- CALCULATE STATISTICS ---
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(steps_taken)
    avg_distance = np.mean(final_distances)
    min_distance = np.min(final_distances) if final_distances else 0.0

    # --- PRINT REPORT ---
    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT ({num_episodes} Episodes)")
    print("=" * 40)
    print(f"Success Rate:      {success_rate:.2f}%")
    print(f"Average Reward:    {avg_reward:.2f} (std: {std_reward:.2f})")
    print(f"Average Steps:     {avg_steps:.2f}")
    print(f"Avg Final Error:   {avg_distance:.5f} m ({avg_distance * 1000:.2f} mm)")
    print(f"Best Accuracy:     {min_distance:.5f} m ({min_distance * 1000:.2f} mm)")
    print("=" * 40)


if __name__ == "__main__":
    # Example usage - Update paths to match your run ID
    run_id = "eyn9liz5"  # Replace with your actual run ID

    evaluate_agent(
        model_path=f"models/{run_id}/{run_id}.zip",  # Path to model
        vec_norm_path=f"models/{run_id}/{run_id}_vecnormalize.pkl",  # Path to stats
        num_episodes=10
    )