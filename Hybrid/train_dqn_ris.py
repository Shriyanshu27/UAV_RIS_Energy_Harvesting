import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_foo.envs.foo_env_dqn import FooEnvDQN

# === Initialize Environment ===
env = DummyVecEnv([lambda: FooEnvDQN()])

# === Create DQN Model ===
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    train_freq=1,
    target_update_interval=250,
    verbose=1,
    tensorboard_log="./models/dqn_tensorboard"
)

# === Train Model ===
total_timesteps = 10000  # Lower for fast test runs
print(f"🚀 Starting DQN training for {total_timesteps} timesteps...\n")

model.learn(total_timesteps=total_timesteps)

# === Save Model ===
model.save("models/DQN_RIS_Agent")
print("\n✅ DQN RIS model saved as models/DQN_RIS_Agent.zip")
