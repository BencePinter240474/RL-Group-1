"""
Test script for OT2Env wrapper.
Run this before training to validate the environment.
"""

import numpy as np
from ot2_env_wrapper import OT2Env


def test_env_checker():
    """Test 1: Gymnasium's built-in environment checker."""
    print("=" * 60)
    print("TEST 1: Gymnasium Environment Checker")
    print("=" * 60)
    
    from gymnasium.utils.env_checker import check_env
    
    env = OT2Env(render_mode=None, normalize=True)
    try:
        check_env(env, warn=True)
        print("✓ Gymnasium check_env PASSED\n")
    except Exception as e:
        print(f"✗ Gymnasium check_env FAILED: {e}\n")
    env.close()


def test_sb3_checker():
    """Test 2: Stable Baselines 3 environment checker."""
    print("=" * 60)
    print("TEST 2: Stable Baselines 3 Environment Checker")
    print("=" * 60)
    
    try:
        from stable_baselines3.common.env_checker import check_env
        
        env = OT2Env(render_mode=None, normalize=True)
        try:
            check_env(env, warn=True)
            print("✓ SB3 check_env PASSED\n")
        except Exception as e:
            print(f"✗ SB3 check_env FAILED: {e}\n")
        env.close()
    except ImportError:
        print("⚠ Stable Baselines 3 not installed, skipping...\n")


def test_spaces():
    """Test 3: Validate observation and action spaces."""
    print("=" * 60)
    print("TEST 3: Space Validation")
    print("=" * 60)
    
    env = OT2Env(render_mode=None, normalize=True)
    obs, info = env.reset(seed=42)
    
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action space bounds: [{env.action_space.low}, {env.action_space.high}]")
    print()
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation bounds: [{env.observation_space.low}, {env.observation_space.high}]")
    print()
    
    # Check observation is within bounds
    obs_in_bounds = env.observation_space.contains(obs)
    print(f"Initial observation in bounds: {'✓' if obs_in_bounds else '✗'}")
    print(f"Observation values: {obs}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Check random actions are valid
    for i in range(5):
        action = env.action_space.sample()
        action_valid = env.action_space.contains(action)
        print(f"Random action {i+1} valid: {'✓' if action_valid else '✗'} {action}")
    
    print()
    env.close()


def test_random_actions(num_episodes=3, max_steps_per_ep=100, render=False):
    """Test 4: Run episodes with random actions."""
    print("=" * 60)
    print("TEST 4: Random Actions Test")
    print("=" * 60)
    
    env = OT2Env(render_mode="human" if render else None, normalize=True)
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"  Goal: {info['goal_position']}")
        print(f"  Start distance: {info['distance_to_target']*1000:.2f}mm")
        
        for step in range(max_steps_per_ep):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"  ✓ Success at step {step+1}! Distance: {info['distance_to_target']*1000:.3f}mm")
                break
            if truncated:
                print(f"  Truncated at step {step+1}")
                break
        
        print(f"  Final distance: {info['distance_to_target']*1000:.2f}mm")
        print(f"  Total reward: {total_reward:.2f}")
    
    print()
    env.close()


def test_p_controller(num_episodes=2, render=False):
    """Test 5: Run episodes with a simple P-controller."""
    print("=" * 60)
    print("TEST 5: P-Controller Test (should reach goal)")
    print("=" * 60)
    
    env = OT2Env(render_mode="human" if render else None, normalize=True)
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode + 100)
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"  Goal: {info['goal_position']}")
        print(f"  Start distance: {info['distance_to_target']*1000:.2f}mm")
        
        for step in range(1000):
            # Denormalize observation to get real values
            raw_obs = env._denormalize_obs(obs)
            pipette_pos = raw_obs[:3]
            goal_pos = raw_obs[6:9]
            
            # Simple proportional controller
            error = goal_pos - pipette_pos
            action = np.clip(error * 10.0, -1.0, 1.0).astype(np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                print(f"  Step {step}: distance={info['distance_to_target']*1000:.2f}mm, reward={reward:.2f}")
            
            if terminated:
                print(f"  ✓ Success at step {step+1}! Distance: {info['distance_to_target']*1000:.4f}mm")
                break
            if truncated:
                print(f"  Truncated at step {step+1}")
                break
        
        print(f"  Total reward: {total_reward:.2f}")
    
    print()
    env.close()


def test_reward_function():
    """Test 6: Verify reward function behavior."""
    print("=" * 60)
    print("TEST 6: Reward Function Verification")
    print("=" * 60)
    
    env = OT2Env(render_mode=None, normalize=True)
    obs, info = env.reset(seed=42)
    
    print("Taking steps toward goal with P-controller...\n")
    print(f"{'Step':<6} {'Distance (mm)':<14} {'Progress':<10} {'Penalty':<10} {'Precision':<10} {'Total':<10}")
    print("-" * 60)
    
    for step in range(10):
        raw_obs = env._denormalize_obs(obs)
        pipette_pos = raw_obs[:3]
        goal_pos = raw_obs[6:9]
        
        error = goal_pos - pipette_pos
        action = np.clip(error * 10.0, -1.0, 1.0).astype(np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        rc = info['reward_components']
        print(f"{step+1:<6} {info['distance_to_target']*1000:<14.3f} {rc['progress']:<10.3f} "
              f"{rc['distance_penalty']:<10.3f} {rc['precision_bonus']:<10.1f} {reward:<10.3f}")
        
        if terminated:
            print(f"\n✓ Reached goal! Success bonus: {rc['success_bonus']}")
            break
    
    print()
    env.close()


def test_normalization():
    """Test 7: Verify normalization works correctly."""
    print("=" * 60)
    print("TEST 7: Normalization Test")
    print("=" * 60)
    
    # Test with normalization
    env_norm = OT2Env(render_mode=None, normalize=True)
    obs_norm, _ = env_norm.reset(seed=42)
    
    # Test without normalization
    env_raw = OT2Env(render_mode=None, normalize=False)
    obs_raw, _ = env_raw.reset(seed=42)
    
    print("Normalized observation:")
    print(f"  Range: [{obs_norm.min():.3f}, {obs_norm.max():.3f}]")
    print(f"  Values: {obs_norm}")
    print()
    print("Raw observation:")
    print(f"  Range: [{obs_raw.min():.3f}, {obs_raw.max():.3f}]")
    print(f"  Values: {obs_raw}")
    print()
    
    # Verify denormalization recovers original values
    obs_recovered = env_norm._denormalize_obs(obs_norm)
    match = np.allclose(obs_recovered, obs_raw, atol=1e-5)
    print(f"Denormalization recovers raw values: {'✓' if match else '✗'}")
    
    print()
    env_norm.close()
    env_raw.close()


def run_all_tests(render=False):
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OT2Env WRAPPER TEST SUITE")
    print("=" * 60 + "\n")
    
    test_env_checker()
    test_sb3_checker()
    test_spaces()
    test_normalization()
    test_reward_function()
    test_random_actions(num_episodes=2, max_steps_per_ep=50, render=render)
    test_p_controller(num_episodes=1, render=render)
    
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OT2Env wrapper")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--test", type=int, choices=[1,2,3,4,5,6,7], 
                        help="Run specific test (1-7)")
    args = parser.parse_args()
    
    if args.test:
        tests = {
            1: test_env_checker,
            2: test_sb3_checker,
            3: test_spaces,
            4: lambda: test_random_actions(render=args.render),
            5: lambda: test_p_controller(render=args.render),
            6: test_reward_function,
            7: test_normalization,
        }
        tests[args.test]()
    else:
        run_all_tests(render=args.render)