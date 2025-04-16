import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from hybrid_split_env import HybridSplitEnv

# Initialize environment
hybrid_env = HybridSplitEnv(LoadData=True, Train=True)

# PPO agent: continuous actions
ppo_model = PPO(
    "MlpPolicy",
    DummyVecEnv([lambda: hybrid_env.env]),
    verbose=1,
    tensorboard_log="./ppo_tensorboard"
)

# DQN agent: discrete binary RIS control
dqn_model = DQN(
    "MlpPolicy",
    DummyVecEnv([lambda: hybrid_env.env]),
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    train_freq=1,
    target_update_interval=250,
    verbose=1,
    tensorboard_log="./dqn_tensorboard"
)

# Training loop parameters
total_timesteps = 50000
eval_every = 1000

# Initial reset
obs_cont, obs_disc = hybrid_env.reset()

for step in range(1, total_timesteps + 1):
    # Predict actions
    action_cont, _ = ppo_model.predict(obs_cont, deterministic=False)
    action_disc, _ = dqn_model.predict(obs_disc, deterministic=False)

    # Step in env
    (next_obs_cont, next_obs_disc), reward, done, info = hybrid_env.step(action_cont, action_disc)

    # PPO (on-policy): use rollout buffer
    ppo_model.rollout_buffer.add(obs_cont, action_cont, reward, 0.0, done, {}, next_obs_cont)

    # DQN (off-policy): store in replay buffer manually
    dqn_model.replay_buffer.add(obs_disc, action_disc, reward, next_obs_disc, done, {})

    # Training updates
    if step % 64 == 0:
        ppo_model.train()
        dqn_model.train(batch_size=32, gradient_steps=1)

    # Reset if episode ends
    if done:
        obs_cont, obs_disc = hybrid_env.reset()
    else:
        obs_cont, obs_disc = next_obs_cont, next_obs_disc

    # Logging
    if step % eval_every == 0:
        print(f"[Step {step}] Shared reward: {reward:.4f}")

# Save trained models
ppo_model.save("models/HybridPPO_Agent")
dqn_model.save("models/HybridDQN_Agent")

print("✅ Training complete and models saved.")
