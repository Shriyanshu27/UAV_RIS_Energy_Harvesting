import os
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_foo.envs.foo_env_dqn import FooEnvDQN

# === Ensure models directory exists ===
os.makedirs("models", exist_ok=True)

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
total_timesteps = 10000  # Use smaller value for quick test runs
print(f"🚀 Starting DQN training for {total_timesteps} timesteps...\n")

model.learn(total_timesteps=total_timesteps)

# === Save Model ===
model.save("models/DQN_RIS_Agent")
print("\n✅ DQN RIS model saved at: models/DQN_RIS_Agent.zip")

# === (Optional) Evaluate briefly ===
obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
