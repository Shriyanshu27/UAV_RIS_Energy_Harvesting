import numpy as np
from gym_foo.envs.foo_env import FooEnv

class HybridSplitEnv:
    """
    This wrapper environment separates the action interfaces for PPO (continuous)
    and DQN (discrete), allowing them to jointly control FooEnv with a shared reward.
    """

    def __init__(self, LoadData=True, Train=True):
        self.env = FooEnv(LoadData=LoadData, Train=Train)

        # Expose separate obs/action spaces to both agents
        self.continuous_action_space = self.env.action_space['continuous']
        self.discrete_action_space = self.env.action_space['discrete']
        self.observation_space = self.env.observation_space

        # Use same obs for both agents
        self.continuous_observation_space = self.observation_space
        self.discrete_observation_space = self.observation_space

    def reset(self):
        """
        Returns a tuple of (obs_for_ppo, obs_for_dqn)
        """
        obs = self.env.reset()
        return obs, obs  # Both agents get the same state

    def step(self, cont_action, disc_action):
        """
        Accepts action from PPO and DQN separately, combines them and calls FooEnv.step()

        Returns:
        - (next_obs_for_ppo, next_obs_for_dqn)
        - shared reward
        - done
        - info dict
        """
        combined_action = {
            "continuous": cont_action,
            "discrete": disc_action
        }

        next_obs, reward, done, info = self.env.step(combined_action)
        return (next_obs, next_obs), reward, done, info

    def render(self, mode="human"):
        self.env.render(mode)
