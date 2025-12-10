from clearml.task import TaskInstance
from stable_baselines3 import PPO
import os
import argparse
from ot2_env_wrapper import OT2Env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task

os.environ["WANDB_API_KEY"] = "a9d5663872aaae616cbdd9ef0c57193447b5e12d"

# 1. Setup Arguments First
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--clip_range", type=float, default=0.2, help="PPO Clipping parameter")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=1000000, help="Timesteps")

args = parser.parse_args()

# 2. ClearML Init
task: TaskInstance = Task.init(
    project_name='Mentor Group - Karna/Group 1',
    task_name='Experiment_PPO_1M'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# 3. WandB Init
# Added config=vars(args) so your hyperparameters are logged to WandB too!
run = wandb.init(project="RL controller",entity="240474-breda-university-of-applied-sciences", sync_tensorboard=True, config=vars(args))

# 4. SAFETY FIX: Create the directory before using it
os.makedirs(f"models/{run.id}", exist_ok=True)

# 5. Environment
env = DummyVecEnv([lambda: OT2Env(render=False)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 6. Model
model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            clip_range=args.clip_range,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(model_save_freq=100000, # Increased freq to save space
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