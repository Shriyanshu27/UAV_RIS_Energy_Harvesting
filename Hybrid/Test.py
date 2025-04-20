import numpy as np
from stable_baselines3 import PPO, DQN
from hybrid_split_env import HybridSplitEnv

# === Load Trained Models ===
ppo_model = PPO.load("models/HybridPPO_Agent.zip")
dqn_model = DQN.load("models/HybridDQN_Agent.zip")

# === Initialize Hybrid Environment in Test Mode ===
env = HybridSplitEnv(LoadData=True, Train=False)

# === Reset Environment ===
obs_cont, obs_disc = env.reset()
done = False
total_reward = 0.0
step_count = 0

# === Track RIS state manually ===
RIS_L = 16
ris_state = np.zeros(RIS_L, dtype=np.float32)

print("🚀 Running Hybrid PPO + DQN Agent in Test Mode...\n")

# === Run Until Episode Ends ===
while not done:
    # Predict continuous action from PPO
    action_cont, _ = ppo_model.predict(obs_cont, deterministic=True)

    # Construct full DQN observation: [distances (4) + RIS state (16)]
    dqn_obs = np.concatenate([obs_disc, ris_state], dtype=np.float32)

    # Predict RIS bit to flip from DQN
    flip_index, _ = dqn_model.predict(dqn_obs.reshape(1, -1), deterministic=True)
    ris_state[flip_index] = 1 - ris_state[flip_index]  # toggle reflect/harvest bit

    # Step environment with PPO and updated RIS config
    (next_obs_cont, next_obs_disc), reward, done, info = env.step(action_cont, ris_state)

    total_reward += reward
    step_count += 1

    # Logging
    print(f"Step {step_count:02d} | Reward: {reward:.2f} | EH: {info.get('eh', 0):.2f} | Tau: {info.get('tau', 0):.2f}")

    # Update observations
    obs_cont = next_obs_cont
    obs_disc = next_obs_disc

# === Summary ===
print("\n✅ Test Complete!")
print(f"📦 Total Steps: {step_count}")
print(f"💰 Total Reward: {total_reward:.2f}")
print(f"⚡ Final Energy Harvested: {info.get('eh_total', 'N/A')} units")
