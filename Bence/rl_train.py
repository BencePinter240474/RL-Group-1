import os
# Disable GUI for headless servers
os.environ["PYBULLET_EGL_DEVICE_ID"] = "-1"

from stable_baselines3 import SAC
import argparse
from ot2_env_wrapper import OT2Env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task

# Set WandB API key BEFORE wandb.init()
os.environ["WANDB_API_KEY"] = "00dfdda8605c784f772ee2a8f94cc00d861e8bf7"

# 1. Setup Arguments First
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--learning_starts", type=int, default=100)
parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
parser.add_argument("--total_timesteps", type=int, default=1000000, help="Timesteps")

args = parser.parse_args()

# 2. ClearML Init
task = Task.init(
    project_name='Mentor Group - Karna/Group 1',
    task_name='Experiment_SAC_1M'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# 3. WandB Init
run = wandb.init(project="RL controller", entity="240474-breda-university-of-applied-sciences", sync_tensorboard=True, config=vars(args))

# 4. SAFETY FIX: Create the directory before using it
os.makedirs(f"models/{run.id}", exist_ok=True)

# 5. Environment
env = DummyVecEnv([lambda: OT2Env(render=False)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 6. Model (Fixed for SAC - removed PPO-specific parameters)
model = SAC('MlpPolicy', env, verbose=1,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            tau=args.tau,
            tensorboard_log=f"runs/{run.id}")

wandb_callback = WandbCallback(model_save_freq=100000,
                               model_save_path=f"models/{run.id}",
                               verbose=2)

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

# This attaches the files to the 'Mentor Group - Karna/Group 1' experiment
task.upload_artifact(name="final_model", artifact_object=model_path)
task.upload_artifact(name="vec_normalize", artifact_object=stats_path)

print("Training Complete! Artifacts uploaded.")