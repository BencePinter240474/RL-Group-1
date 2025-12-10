from clearml import Task
import os
import argparse

os.environ["WANDB_API_KEY"] = "a9d5663872aaae616cbdd9ef0c57193447b5e12d"

# 1. Setup Arguments First
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--learning_starts", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--train_freq", type=int, default=1)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--ent_coef", type=str, default="auto")
parser.add_argument("--total_timesteps", type=int, default=1000000, help="Timesteps")

args = parser.parse_args()

# 2. ClearML Init with COMPATIBLE package versions
task = Task.init(
    project_name='Mentor Group - Karna/Group 1',
    task_name='Experiment_SAC_1M_MultiReward',  # Disable auto-detection
)

# FIXED: Use compatible versions
requirements = [
    "gymnasium==0.29.1",  # Compatible with stable-baselines3 2.1.0
    "numpy==1.24.3",      # Compatible numpy version
    "stable-baselines3==2.1.0",  # Stable version that works with gymnasium 0.29
    "pybullet==3.2.5",
    "tensorboard==2.15.0",
    "wandb==0.16.0",
    "torch==2.1.0",
    "clearml"
]

task.set_packages(requirements)
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# NOW import the packages after ClearML setup
from stable_baselines3 import SAC
from ot2_env_wrapper import OT2Env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

# 3. WandB Init
run = wandb.init(
    project="RL controller",
    entity="240474-breda-university-of-applied-sciences", 
    sync_tensorboard=False, 
    config=vars(args),
    name="SAC_MultiReward"
)

# 4. Create directory for models
os.makedirs(f"models/{run.id}", exist_ok=True)

# 5. Environment
env = DummyVecEnv([lambda: OT2Env(render=False)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 6. Model - Using SAC for better continuous control
model = SAC(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    batch_size=args.batch_size,
    tau=args.tau,
    gamma=args.gamma,
    train_freq=args.train_freq,
    gradient_steps=args.gradient_steps,
    ent_coef=args.ent_coef,
    policy_kwargs=dict(
        net_arch=[256, 256]  # Good size for robotic control
    ),
    tensorboard_log=f"runs/{run.id}"
)

wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{run.id}",
    verbose=2
)

# 7. Learning
model.learn(
    total_timesteps=args.total_timesteps,
    callback=wandb_callback,
    progress_bar=True,
    tb_log_name=f"runs/{run.id}"
)

# 8. Saving & Uploading
model_path = f"models/{run.id}/final_model.zip"
stats_path = f"models/{run.id}/vec_normalize.pkl"

model.save(model_path)
env.save(stats_path)

# Upload to ClearML
task.upload_artifact(name="final_model", artifact_object=model_path)
task.upload_artifact(name="vec_normalize", artifact_object=stats_path)

print("Training Complete! Artifacts uploaded.")
wandb.finish()