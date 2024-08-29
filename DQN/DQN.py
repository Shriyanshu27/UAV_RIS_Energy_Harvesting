import gym
import gym_foo
import numpy as np
import math
from stable_baselines3 import DQN
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')

# DQN works directly with the discrete action space, so no need for action space noise
model = DQN("MlpPolicy", env, learning_rate=0.001, buffer_size=pow(2, 14), learning_starts=1000, batch_size=64, target_update_interval=250, verbose=1, tensorboard_log="./DQN_MultiUT_Two_tensorboard/")
model.learn(total_timesteps=3240000, log_interval=10)
model.save("dqn_MultiUT_Two")
env = model.get_env()

del model  # remove to demonstrate saving and loading

model = DQN.load("DQN_MultiUT_Two")

obs = env.reset()
R = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    R += rewards
    if dones:
        break

    env.render()
print(R)
