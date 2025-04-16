import numpy as np
from stable_baselines3 import PPO, DQN
from hybrid_split_env import HybridSplitEnv

# Load hybrid environment
env = HybridSplitEnv(LoadData=True, Train=False)

# Load trained agents
ppo_model = PPO.load("models/HybridPPO_Agent")
dqn_model = DQN.load("models/HybridDQN_Agent")

# Reset environment
obs_cont, obs_disc = env.reset()
done = False
total_reward = 0
step_count = 0

print("\n🧪 Running evaluation episode...\n")

while not done:
    action_cont, _ = ppo_model.predict(obs_cont, deterministic=True)
    action_disc, _ = dqn_model.predict(obs_disc, deterministic=True)

    (next_obs_cont, next_obs_disc), reward, done, info = env.step(action_cont, action_disc)

    total_reward += reward
    obs_cont, obs_disc = next_obs_cont, next_obs_disc
    step_count += 1

    print(f"Step {step_count:02d} | Reward: {reward:.4f}")

print(f"\n✅ Evaluation complete. Total Episode Reward: {total_reward:.4f}\n")
