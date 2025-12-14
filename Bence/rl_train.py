from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import torch.nn as nn
import os
import argparse
from ot2_env_wrapper import OT2Env
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task

# It is safe to leave imports and global variables (like os.environ) here
os.environ["WANDB_API_KEY"] = "00dfdda8605c784f772ee2a8f94cc00d861e8bf7"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Key hyperparameter changes explained below
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--n_envs", type=int, default=16)

    args = parser.parse_args()

    task = Task.init(
        project_name='Mentor Group - Karna/Group 1',
        task_name='Experiment_PPO_Optimized'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.execute_remotely(queue_name="default")

    run = wandb.init(
        project="RL controller",
        entity="240474-breda-university-of-applied-sciences",
        sync_tensorboard=True,  # Enable for better logging
        config=vars(args)
    )

    os.makedirs(f"models/{run.id}", exist_ok=True)

    print(f"Creating {args.n_envs} parallel environments...")
    env = make_vec_env(
        OT2Env,
        n_envs=args.n_envs,
        env_kwargs={"render": False},
        vec_env_cls=SubprocVecEnv,
        seed=42
    )

    # Improved normalization settings
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,  # Clip rewards to prevent instability
        gamma=args.gamma   # Use same gamma for reward normalization
    )

    # Create eval environment for monitoring
    eval_env = make_vec_env(
        OT2Env,
        n_envs=4,
        env_kwargs={"render": False},
        vec_env_cls=SubprocVecEnv,
        seed=123
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Custom policy architecture (wider networks often help robotics)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256]   # Value network
        ),
        activation_fn=nn.Tanh  # Tanh often works better for continuous control
    )

    # Linear learning rate schedule
    def lr_schedule(progress_remaining):
        return args.learning_rate * progress_remaining

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=lr_schedule,
        gamma=args.gamma,
        clip_range=args.clip_range,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run.id}",
        seed=42
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run.id}/best",
        log_path=f"logs/{run.id}",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True
    )

    wandb_callback = WandbCallback(
        model_save_freq=100000,
        model_save_path=f"models/{run.id}",
        verbose=2
    )

    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([wandb_callback, eval_callback]),
        progress_bar=True,
        tb_log_name=f"runs/{run.id}"
    )

    # Save final model and normalization stats
    model.save(f"models/{run.id}/final_model.zip")
    env.save(f"models/{run.id}/vec_normalize.pkl")

    # Also sync eval env stats
    eval_env.save(f"models/{run.id}/eval_vec_normalize.pkl")

    task.upload_artifact(name="final_model", artifact_object=f"models/{run.id}/final_model.zip")
    task.upload_artifact(name="best_model", artifact_object=f"models/{run.id}/best/best_model.zip")
    task.upload_artifact(name="vec_normalize", artifact_object=f"models/{run.id}/vec_normalize.pkl")

    print("Training Complete!")
    run.finish()