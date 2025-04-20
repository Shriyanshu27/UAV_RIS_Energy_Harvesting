import numpy as np
from stable_baselines3 import PPO, DQN
from gym_foo.envs.foo_env import FooEnv

# Load trained PPO and DQN models
ppo_model = PPO.load("models/PPO_Hybrid.zip")
dqn_model = DQN.load("models/DQN_RIS_Agent.zip")

# Create the environment
env = FooEnv(LoadData=True, Train=False)

# Reset the environment
obs = env.reset()
done = False
total_reward = 0

print("🔁 Running PPO + DQN Hybrid Test Episode...\n")

while not done:
    # Get continuous action from PPO
    ppo_action, _ = ppo_model.predict(obs, deterministic=True)

    # Get discrete RIS config from DQN
    dqn_action, _ = dqn_model.predict(obs, deterministic=True)

    # Set the DQN-generated RIS config inside env
    env.set_ris_config(dqn_action)

    # Merge actions into dict format expected by hybrid env
    hybrid_action = {
        "continuous": np.array(ppo_action),
        "discrete": np.array(dqn_action)
    }

    # Step the environment
    obs, reward, done, info = env.step(hybrid_action)
    total_reward += reward

    print(f"Step Reward: {reward:.4f} | Done: {done}")

print(f"\n✅ Test Complete. Total Episode Reward: {total_reward:.4f}")
