import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ot2_env_wrapper import OT2Env2D
import time

def test_model(model_path, stats_path, n_tests=10, max_steps=500, fixed_z=0.125, success_threshold=0.001):
    """
    Test the trained SAC model on random coordinates within training range.
    
    Args:
        model_path: Path to the trained model (.zip)
        stats_path: Path to VecNormalize stats (.pkl)
        n_tests: Number of test coordinates
        max_steps: Maximum steps per episode
        fixed_z: Fixed Z height for 2D environment
        success_threshold: Distance threshold for success in meters (default 0.001 = 1mm)
    """
    
    # Use the SAME goal range as training!
    x_min, x_max = 0.10, 0.25
    y_min, y_max = 0.05, 0.21
    
    # Load environment with same config as training
    env = DummyVecEnv([lambda: OT2Env2D(render_mode=None, normalize=True, fixed_z=fixed_z)])
    env = VecNormalize.load(stats_path, env)
    env.training = False  # Disable normalization updates during testing
    env.norm_reward = False
    
    # Load trained model
    model = SAC.load(model_path)
    
    # Generate random test coordinates within workspace
    np.random.seed(42)  # For reproducibility
    test_coords = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(n_tests, 2)
    )
    
    print("="*60)
    print(f"TESTING SAC MODEL - {n_tests} Random Coordinates")
    print(f"Goal Range (training range): X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"Fixed Z: {fixed_z}")
    print(f"Max steps per test: {max_steps}")
    print(f"Success threshold: {success_threshold*1000:.1f}mm")
    print("="*60)
    
    results = []
    
    for i, goal in enumerate(test_coords):
        print(f"\nTest {i+1}/{n_tests}")
        print(f"Target: X={goal[0]:.4f}, Y={goal[1]:.4f}")
        
        # Reset with specific goal
        obs = env.reset()
        # Set goal directly in the underlying environment
        env.envs[0].goal_position = goal.astype(np.float32)
        obs = env.reset()
        
        start_time = time.time()
        steps = 0
        final_distance = None
        success = False
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            # Get actual info from underlying env
            actual_info = info[0]
            final_distance = actual_info['distance_to_target']
            
            # Check success with custom threshold
            if final_distance < success_threshold:
                success = True
                break
        
        elapsed_time = time.time() - start_time
        
        # Convert distance to mm
        final_distance_mm = final_distance * 1000
        
        # Store results
        result = {
            'target': goal,
            'success': success,
            'final_distance_mm': final_distance_mm,
            'steps': steps,
            'time_seconds': elapsed_time
        }
        results.append(result)
        
        # Print result
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} - Distance: {final_distance_mm:.2f}mm, Steps: {steps}, Time: {elapsed_time:.2f}s")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    successes = [r for r in results if r['success']]
    success_rate = len(successes) / n_tests * 100
    
    all_distances = [r['final_distance_mm'] for r in results]
    all_steps = [r['steps'] for r in results]
    all_times = [r['time_seconds'] for r in results]
    
    print(f"\nSuccess Rate: {success_rate:.1f}% ({len(successes)}/{n_tests})")
    
    print(f"\nAccuracy (Final Distance):")
    print(f"  Mean: {np.mean(all_distances):.2f}mm")
    print(f"  Std:  {np.std(all_distances):.2f}mm")
    print(f"  Min:  {np.min(all_distances):.2f}mm")
    print(f"  Max:  {np.max(all_distances):.2f}mm")
    
    print(f"\nSpeed (Steps to Target):")
    print(f"  Mean: {np.mean(all_steps):.1f} steps")
    print(f"  Std:  {np.std(all_steps):.1f} steps")
    print(f"  Min:  {np.min(all_steps)} steps")
    print(f"  Max:  {np.max(all_steps)} steps")
    
    print(f"\nExecution Time:")
    print(f"  Mean: {np.mean(all_times):.2f}s per episode")
    print(f"  Total: {np.sum(all_times):.2f}s for all tests")
    
    if successes:
        success_steps = [r['steps'] for r in successes]
        success_times = [r['time_seconds'] for r in successes]
        print(f"\nFor Successful Reaches Only:")
        print(f"  Mean Steps: {np.mean(success_steps):.1f}")
        print(f"  Mean Time: {np.mean(success_times):.2f}s")
    
    env.close()
    return results


if __name__ == "__main__":
    # OPTION 1: Hardcoded paths (modify these to match your files)
    USE_HARDCODED = True  # Set to False to use command line arguments
    
    if USE_HARDCODED:
        # Modify these paths to match your actual model files
        model_path = "models/final_model_2d.zip"  # Add .zip extension if missing
        stats_path = "models/vec_normalize_2d.pkl"
        n_tests = 10
        max_steps = 500
        fixed_z = 0.125
        success_threshold = 0.001  # 1mm success threshold
        
        print(f"Using hardcoded paths:")
        print(f"  Model: {model_path}")
        print(f"  Stats: {stats_path}")
        
        # Run tests
        results = test_model(
            model_path=model_path,
            stats_path=stats_path,
            n_tests=n_tests,
            max_steps=max_steps,
            fixed_z=fixed_z,
            success_threshold=success_threshold
        )
    else:
        # OPTION 2: Use command line arguments
        import argparse
        
        parser = argparse.ArgumentParser(description="Test trained SAC model")
        parser.add_argument("--model_path", type=str, required=True, help="Path to model .zip file")
        parser.add_argument("--stats_path", type=str, required=True, help="Path to vec_normalize .pkl file")
        parser.add_argument("--n_tests", type=int, default=10, help="Number of test coordinates")
        parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
        parser.add_argument("--fixed_z", type=float, default=0.125, help="Fixed Z height")
        
        args = parser.parse_args()
        
        # Run tests
        results = test_model(
            model_path=args.model_path,
            stats_path=args.stats_path,
            n_tests=args.n_tests,
            max_steps=args.max_steps,
            fixed_z=args.fixed_z
        )
    
    print("\nTest complete!")