import gym
import gym_foo
import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')

n_actions = env.action_space.shape[-1]

model = PPO("MlpPolicy", env, learning_rate=1e-3, n_steps=pow(2, 10), batch_size=pow(2, 10), verbose=1, tensorboard_log="./PPO_MultiUT_Time_tensorboard/")
model.learn(total_timesteps=1640000, log_interval=10)
model.save("ppo_MultiUT_Time")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_MultiUT_Time")

obs = env.reset()
R = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    R += rewards
    if dones==True:
        break
        
    env.render()
print(R)
