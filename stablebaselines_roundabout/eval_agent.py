from stable_baselines3 import PPO
import highway_env
import gymnasium as gym

# Load original env (no custom reward)
eval_env = gym.make("roundabout-v0", render_mode ="human")

# Load trained model
model = PPO.load("checkpoints/ppo_roundabout_20000_steps")

n_episodes = 100
mean_reward =0
# Evaluate
for ep in range(n_episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated or not eval_env.unwrapped.vehicle.on_road
        total_reward += reward
        eval_env.render()

    print(f"Evaluation reward for episode {ep} with original reward function:", total_reward)
    mean_reward +=total_reward

print("Mean reward:", mean_reward/n_episodes)