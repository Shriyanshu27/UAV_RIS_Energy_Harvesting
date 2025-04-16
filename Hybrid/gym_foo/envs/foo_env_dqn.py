import gym
import numpy as np
import math as mt
import globe

class FooEnvDQN(gym.Env):
    def __init__(self):
        super(FooEnvDQN, self).__init__()

        # RIS setup
        self.RIS_L = 16
        globe.set_value('RIS_L', self.RIS_L)
        self.action_space = gym.spaces.Discrete(2 * self.RIS_L)  # Reflect: 0–15, Harvest: 16–31

        # Observation: flattened [radio_state + RIS state]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=20.0, shape=(4 + self.RIS_L,), dtype=np.float32
        )

        # Internal RIS control state
        self.omega_r = np.zeros(self.RIS_L)  # 0: reflect, 1: harvest
        self.theta_r = np.ones(self.RIS_L) * np.pi / 2  # constant angles

        self.tau = 0.5
        self.power = [mt.pow(10, 3)] * 3  # 1W per user

        self.step_count = 0
        self.max_steps = 41

        # Dummy UAV & UT positions for now
        globe.set_value('L_U', [0, 0, 20])
        globe.set_value('L_AP', [0, 0, 10])
        self._load_dummy_data()

    def _load_dummy_data(self):
        # Normally load trajectories — for now just static points
        self.L_U = [0, 0, 20]
        self.L_AP = [0, 0, 10]
        self.UT_0 = [10, 0, 0]
        self.UT_1 = [-10, 0, 0]
        self.UT_2 = [0, 10, 0]

    def step(self, action):
        if action < self.RIS_L:
            self.omega_r[action] = 0  # Reflect
        else:
            self.omega_r[action - self.RIS_L] = 1  # Harvest

        reward = self._compute_reward()
        self.step_count += 1
        done = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, done, {}

    def reset(self):
        self.omega_r = np.zeros(self.RIS_L)
        self.theta_r = np.ones(self.RIS_L) * np.pi / 2
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        L_U = np.array(self.L_U)
        L_AP = np.array(self.L_AP)
        UTs = [np.array(self.UT_0), np.array(self.UT_1), np.array(self.UT_2)]

        distances = [np.linalg.norm(L_U - L_AP)] + [np.linalg.norm(L_U - ut) for ut in UTs]
        radio_state = np.array(distances, dtype=np.float32) / np.sum(distances)
        return np.concatenate([radio_state, self.omega_r.astype(np.float32)])

    def _compute_reward(self):
        # Dummy reward for now: penalize more harvesting (simulate energy use)
        throughput_bonus = (self.RIS_L - np.sum(self.omega_r)) / self.RIS_L
        return throughput_bonus

    def render(self, mode="human"):
        print(f"Step {self.step_count} | Omega_R: {self.omega_r}")
