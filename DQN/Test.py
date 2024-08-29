import gym
import gym_foo
import numpy as np
import math
from stable_baselines3 import DQN
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
env = gym.make('foo-v0')

# Load the trained DQN model
model = DQN.load("DQN_MultiUT_Two")

obs = env.reset()
env.Train = False
Rewards = []
Harvest = []
Received = []
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    info = list(info)[0]

    harvestEnergy = np.float(info.split(",")[0])
    receivedEnergy = np.float(info.split(",")[1])
    Rewards.append(rewards)
    Harvest.append(harvestEnergy)
    Received.append(receivedEnergy)
    if dones:
        break
        
    env.render()

print(np.sum(Harvest)/np.sum(Received))
np.savetxt("Rewards.csv", Rewards, delimiter=',')
np.savetxt("Total_Reward.txt", [np.sum(Harvest)/np.sum(Received)])
