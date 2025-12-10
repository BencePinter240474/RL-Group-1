from stable_baselines3 import PPO
import gymnasium as gym
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

env = gym.make('Pendulum-v1', g=9.81)

run = wandb.init(
    entity="240474-breda-university-of-applied-sciences",
    project="RL controller",
    sync_tensorboard=False,
    config=vars(args),  # Log your hyperparameters
    settings=wandb.Settings(start_method="thread")  # Can help with socket issues on Windows
)

model = PPO('MlpPolicy', 
            env, 
            verbose=1,
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",
            )

wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

time_steps = 10000

for i in range(10):
    model.learn(
        total_timesteps=time_steps, 
        callback=wandb_callback, 
        progress_bar=True,
        tb_log_name="PPO",  # Just use a simple name here, not the full path
        reset_num_timesteps=False  # Important: continue counting timesteps
    )
    model.save(f"models/{run.id}/{time_steps*(i+1)}")

wandb.finish()  # Properly close the wandb run