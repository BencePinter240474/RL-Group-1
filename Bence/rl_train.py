from clearml import Task
import os
import argparse

os.environ["WANDB_API_KEY"] = "00dfdda8605c784f772ee2a8f94cc00d861e8bf7"

# 1. Setup Arguments First
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--buffer_size", type=int, default=1500000)
parser.add_argument("--learning_starts", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--train_freq", type=int, default=1)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--ent_coef", type=str, default="auto")
parser.add_argument("--total_timesteps", type=int, default=3000000, help="Timesteps")
parser.add_argument("--fixed_z", type=float, default=0.125, help="Fixed Z height")

args = parser.parse_args()

# 2. ClearML Init
task = Task.init(
    project_name='Mentor Group - Karna/Group 1',
    task_name='SAC_2D_NewReward',
)

requirements = [
    "gymnasium==0.29.1",
    "numpy==1.24.3",
    "stable-baselines3==2.1.0",
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
from ot2_env_wrapper import OT2Env2D
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

# 3. WandB Init
run = wandb.init(
    project="RL controller",
    entity="240474-breda-university-of-applied-sciences", 
    sync_tensorboard=False, 
    config={
        **vars(args),
        "goal_range_x": "[0.10, 0.25]",
        "goal_range_y": "[0.05, 0.21]",
        "reward": "continuous (distance + progress + success_bonus)",
    },
    name=f"SAC_2D_Z{args.fixed_z}_NewReward"
)

# 4. Create directory for models
os.makedirs(f"models/{run.id}", exist_ok=True)

# 5. Environment - 2D version with fixed Z
# Goal range: X=[0.10, 0.25], Y=[0.05, 0.21]
# Reward: continuous (no cliff at 10mm)
env = DummyVecEnv([lambda: OT2Env2D(render_mode=None, normalize=True, fixed_z=args.fixed_z)])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# 6. Model - SAC
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
        net_arch=[256, 256]
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
model_path = f"models/{run.id}/final_model_2d.zip"
stats_path = f"models/{run.id}/vec_normalize_2d.pkl"

model.save(model_path)
env.save(stats_path)

# Upload to ClearML
task.upload_artifact(name="final_model_2d", artifact_object=model_path)
task.upload_artifact(name="vec_normalize_2d", artifact_object=stats_path)

print("Training Complete! Artifacts uploaded.")
print(f"Goal range: X=[0.10, 0.25], Y=[0.05, 0.21]")
print(f"Fixed Z: {args.fixed_z}")
wandb.finish()