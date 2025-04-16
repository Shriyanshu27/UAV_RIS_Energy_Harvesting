import gym
from gym import spaces
import numpy as np
import math as mt
import globe

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, LoadData=True, Train=False, multiUT=True, Trajectory_mode='Kmeans', MaxStep=41):
        globe._init()

        globe.set_value('L_U', [0, 0, 20])
        globe.set_value('L_AP', [0, 0, 10])
        globe.set_value('BS_Z', 8)
        globe.set_value('RIS_L', 16)
        globe.set_value('BW', 2e7)
        globe.set_value('N_0', mt.pow(10, (-174 / 3) / 10))
        globe.set_value('Xi', mt.pow(10, 3 / 10))
        globe.set_value('a', 9.61)
        globe.set_value('b', 0.16)
        globe.set_value('eta_los', 1)
        globe.set_value('eta_nlos', 20)
        globe.set_value('AWGN', mt.pow(10, -102 / 10))
        globe.set_value('N_ris', 100)
        globe.set_value('eta', 0.7)
        globe.set_value('alpha', 3)
        globe.set_value('varphi', mt.pow(10, 20 / 10))
        globe.set_value('P_max', 5e5)
        globe.set_value('N_u', 3)
        globe.set_value('fc', 750e6)
        globe.set_value('c', 3e8)
        globe.set_value('gamma_min', mt.pow(10, 12 / 10))
        globe.set_value('power_i', 500)
        globe.set_value('t', MaxStep)
        globe.set_value('step', 0)
        globe.set_value('kappa', mt.pow(10, -30 / 10))
        globe.set_value('hat_alpha', 2)

        if LoadData:
            suffix = "Train" if Train else "Test"
            if multiUT:
                for i in range(3):
                    UT = np.loadtxt(f"../CreateData/{suffix}_Trajectory_UT_{i}.csv", delimiter=",")
                    globe.set_value(f'UT_{i}', UT)

                UAV_Trajectory = np.loadtxt(f"../CreateData/{Trajectory_mode}_{suffix}_Trajectory_3.csv", delimiter=",")
            else:
                UT = np.loadtxt(f"../CreateData/{suffix}_Trajectory_UT_0.csv", delimiter=",")
                globe.set_value('UT_0', UT)
                UAV_Trajectory = np.loadtxt(f"../CreateData/{Trajectory_mode}_{suffix}_Trajectory_1.csv", delimiter=",")

            globe.set_value('UAV_Trajectory', UAV_Trajectory)

        self.ris_l = globe.get_value('RIS_L')
        cont_dim = 1 + 3 + self.ris_l
        self.action_space = spaces.Dict({
            'continuous': spaces.Box(low=0.0, high=1.0, shape=(cont_dim,), dtype=np.float32),
            'discrete': spaces.MultiBinary(self.ris_l)
        })
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(4,), dtype=np.float32)
        self.Train = Train

    def step(self, action):
        tau = action['continuous'][0]
        power = [mt.pow(10, ((x - 1) * 30 / 10 + 3)) for x in action['continuous'][1:4]]
        theta = action['continuous'][4:] * 2 * np.pi
        omega = action['discrete']
        step = globe.get_value('step')

        reward, state, energy = self.env_state(step, tau, power, theta, omega)
        globe.set_value('step', step + 1)
        done = (step >= globe.get_value('t') - 1)
        state = state / np.sum(state)

        return state, reward / (energy + 1e-10), done, {}

    def reset(self):
        globe.set_value('step', 0)
        return np.random.rand(4)

    def render(self, mode='human'):
        pass

    def pl_BR(self, L_U, L_AP):
        a, b, varphi, alpha = globe.get_value('a'), globe.get_value('b'), globe.get_value('varphi'), globe.get_value('alpha')
        dist = np.linalg.norm(np.array(L_U) - np.array(L_AP))
        theta = (180 / mt.pi) * mt.asin((L_U[2] - L_AP[2]) / dist)
        p_los = 1 / (1 + a * mt.exp(a * b - b * theta))
        p_nlos = 1 - p_los
        return (p_los + p_nlos * varphi) * mt.pow(dist, -alpha)

    def SmallFading_G(self, BS_Z, RIS_L):
        return 1 / np.sqrt(2) * (np.random.randn(BS_Z, RIS_L) + 1j * np.random.randn(BS_Z, RIS_L))

    def Rayleigh_RU(self, RIS_L):
        return 1 / np.sqrt(2) * (np.random.randn(RIS_L, 1) + 1j * np.random.randn(RIS_L, 1))

    def Channel_RU(self, L_U, UT, BS_Z, RIS_L):
        d = np.linalg.norm(np.array(L_U) - np.array(UT))
        PL = np.sqrt(globe.get_value('kappa') * mt.pow(d, -globe.get_value('hat_alpha')))
        return np.ones((RIS_L, 1)) * np.sqrt(5 / 6) * PL + np.sqrt(1 / 6) * PL * self.Rayleigh_RU(RIS_L)

    def EH(self, tau, p1, p2, p3, theta, L_U, L_AP, omega):
        eta = globe.get_value('eta')
        BS_Z, RIS_L = globe.get_value('BS_Z'), globe.get_value('RIS_L')
        g_BR = self.pl_BR(L_U, L_AP)
        G = np.ones((BS_Z, RIS_L)) * g_BR * self.SmallFading_G(BS_Z, RIS_L)
        total_power = p1 + p2 + p3
        g_norm = np.linalg.norm(G, axis=0)
        E_direct = np.sum(g_norm * total_power)
        E_reflect = np.sum(g_norm * total_power * (1 - omega))
        return tau * eta * E_direct + (1 - tau) * eta * E_reflect, E_direct

    def capacity(self, tau, powers, theta, L_U, L_AP, UTs, omega):
        BW = globe.get_value('BW')
        BS_Z, RIS_L = globe.get_value('BS_Z'), globe.get_value('RIS_L')
        G = np.ones((BS_Z, RIS_L)) * self.pl_BR(L_U, L_AP) * self.SmallFading_G(BS_Z, RIS_L)
        coeff = np.diag(np.exp(1j * theta) * omega)

        results = []
        for i, p in zip(UTs, powers):
            h = self.Channel_RU(L_U, i, BS_Z, RIS_L)
            signal = np.sum(np.abs(G @ coeff @ h)**2) * p
            sinr = signal / globe.get_value('AWGN')
            results.append(BW * mt.log2(1 + sinr) * (1 - tau) if sinr > 0 else 0)
        return results

    def env_state(self, step, tau, powers, theta, omega):
        L_U = globe.get_value('UAV_Trajectory')[min(step, globe.get_value('t') - 1)]
        L_AP = globe.get_value('L_AP')
        UTs = [globe.get_value(f'UT_{i}')[min(step, globe.get_value('t') - 1)] for i in range(3)]

        EH_val, recv_energy = self.EH(tau, *powers, theta, L_U, L_AP, omega)
        throughput = self.capacity(tau, powers, theta, L_U, L_AP, UTs, omega)
        reward = EH_val if all(tp > 7e7 for tp in throughput) else 0

        dists = [np.linalg.norm(np.array(L_U) - np.array(L_AP))] + [np.linalg.norm(np.array(L_U) - np.array(u)) for u in UTs]
        return reward, np.array(dists), recv_energy
