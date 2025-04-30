import numpy as np
import matplotlib.pyplot as plt
import highway_env
from stable_baselines3 import PPO
import json
from utils import make_eval_env, RandomAgent

"""Trial 10 finished with value: 17.174999785423278 and parameters: {'learning_rate': 0.0009500406286558071, 'gamma': 0.9011753333009546, 'gae_lambda': 0.8030797886480995, 'clip_range': 0.31492582624818505, 'ent_coef': 1.134957570170251e-06}. Best is trial 10 with value: 17.174999785423278."""
display=True
# Load the trained model and environment
wrapped_env = make_eval_env()
#model = RandomAgent(wrapped_env)
model = PPO.load("stablebaselines_roundabout/roundabout_checkpoints/ppo_roundabout_custom_reward")

# Number of evaluation episodes
n_episodes = 100

# Metrics initialization
total_rewards = []
collisions = 0
successful_episodes = 0

# Evaluate the model over multiple episodes
for episode in range(n_episodes):
    obs = wrapped_env.reset()
    done = False
    collision, entered_roundabout = False, False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = wrapped_env.step(action)
        episode_reward += reward[0]
        
        vehicle = wrapped_env.envs[0].env.unwrapped.vehicle
        lane_index = vehicle.lane_index
        lane = wrapped_env.envs[0].env.unwrapped.road.network.get_lane(lane_index)
        exited_roundabout = isinstance(lane, highway_env.road.lane.CircularLane)
        if exited_roundabout: # à un moment on est entré dans le roundabout
            entered_roundabout = True
        
        if info[0]['crashed']:  # If collision occurs
            collision=True
            collisions += 1

        if display:
            wrapped_env.render()
    
    # Check if the agent successfully exited the roundabout (success condition)
    vehicle = wrapped_env.envs[0].env.unwrapped.vehicle
    lane_index = vehicle.lane_index
    lane = wrapped_env.envs[0].env.unwrapped.road.network.get_lane(lane_index)
    exited_roundabout = isinstance(lane, highway_env.road.lane.StraightLane) and entered_roundabout

    if not collision and exited_roundabout:
        successful_episodes += 1
    print(100*successful_episodes/(episode+1), "% successful episodes")
    total_rewards.append(episode_reward)

# Calculate success rate
success_rate = successful_episodes / n_episodes * 100

# Min and Max reward
min_reward = np.min(total_rewards)
max_reward = np.max(total_rewards)
median_reward = np.median(total_rewards)

# Reward distribution plot
plt.figure(figsize=(10, 5))
plt.hist(total_rewards, bins=30, color='skyblue', edgecolor='black')
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.show()

# Print the metrics
print(f"Success Rate: {success_rate:.2f}%")
print(f"Min Reward: {min_reward:.2f}")
print(f"Max Reward: {max_reward:.2f}")
print(f"Median Reward: {median_reward:.2f}")  # Print the median reward
print(f"Percentage of Collisions: {collisions / n_episodes * 100:.2f}%")

# Save the metrics
metrics = {
    'success_rate': success_rate,
    'min_reward': min_reward,
    'max_reward': max_reward,
    'median_reward': median_reward,  # Include the median reward in the saved metrics
    'collisions_percentage': collisions / n_episodes * 100,
    'rewards': total_rewards
}


with open("stablebaselines_roundabout/roundabout_checkpoints/evaluation_metrics.json", 'w') as jsonfile:
    json.dump(metrics, jsonfile, indent=4)
