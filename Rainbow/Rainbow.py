import gym
import gym_foo
import numpy as np
import math
import os
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.envs import VecNormalize
from stable_baselines3.dqn import Rainbow

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')

# Vectorize the environment and apply normalization
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.)

n_actions = env.action_space.shape[-1]

model = Rainbow("MlpPolicy", env, buffer_size=pow(2, 20), batch_size=pow(2, 10), learning_starts=1000,
                train_freq=4, target_update_interval=8000, max_grad_norm=10, verbose=1,
                tensorboard_log="./Rainbow_MultiUT_Time_tensorboard/")

model.learn(total_timesteps=410000)

model.save("rainbow_MultiUT_Time")

obs = env.reset()
R = 0

while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    R += rewards

    if dones:
        break

print("Total rewards:", R)
