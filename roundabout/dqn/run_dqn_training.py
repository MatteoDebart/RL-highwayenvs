# Set up environment and agent
import highway_env
import gymnasium as gym
import time
import numpy as np

from dqn import DQN
from copy import deepcopy

def custom_eval_agent(agent, env, n_sim = 10, render = False):
    #print(f"Evaluating agent {n_sim} times")
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        entered_roundabout = False
        while not done: 
            additional_reward = 0
            action = agent.get_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
            
            lane = env.unwrapped.road.network.get_lane(env.unwrapped.vehicle.lane_index)

            if isinstance(lane, highway_env.road.lane.CircularLane):
                entered_roundabout = True
            
            if isinstance(lane, highway_env.road.lane.StraightLane) and entered_roundabout:
                entered_roundabout = False
                additional_reward += 5

            if not env.unwrapped.vehicle.on_road:
                done = True
                additional_reward += 5

            env.render()
        episode_rewards[i] = reward_sum + additional_reward
    return episode_rewards

render = True
if render:
    env = gym.make("roundabout-v0", render_mode="human")
else:
    env = gym.make("roundabout-v0")
env.unwrapped.configure({
    "duration": 25,  # 👈 Increase from default (usually 11) to 40 steps
})
env.reset()
# Create the DQN agent
agent = DQN(
    action_space=env.action_space,
    observation_space=env.observation_space,
    gamma=0.99,
    batch_size = 16,
    buffer_capacity=10000,
    update_target_every=10,
    epsilon_start=1.0,
    decrease_epsilon_factor=200,
    epsilon_min=0.01,
    learning_rate=0.0001
)

# Train the agent
n_episodes = 1000

episodes = []
rewards= []
max = 0


for episode in range(n_episodes):
    print("Episode: ", episode)
    state, _ = env.reset()
    terminated = False
    total_reward = 0
    
    done = False
    #print()

    entered_roundabout = False

    while not done:
        additional_reward = 0
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated or not env.unwrapped.vehicle.on_road

        lane = env.unwrapped.road.network.get_lane(env.unwrapped.vehicle.lane_index)

        if isinstance(lane, highway_env.road.lane.CircularLane):
            entered_roundabout = True
        
        if isinstance(lane, highway_env.road.lane.StraightLane) and entered_roundabout:
            entered_roundabout = False
            additional_reward += 5

        if not env.unwrapped.vehicle.on_road:
            done = True
            additional_reward += 5
        
        agent.update(state, action, reward + additional_reward, terminated, next_state)
        if render:
            env.render()

        total_reward += reward + additional_reward
        state = next_state
    if total_reward>max:
        max = total_reward
    
    if episode % 25 == 0:
        mean_reward= np.mean(custom_eval_agent(agent, env, 10, render=render))
        episodes.append(episode)
        rewards.append(mean_reward)
        print(f"Episode {episode} finished with mean reward: {mean_reward}")


import matplotlib.pyplot as plt

plt.plot(episodes, rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward over Episodes")
plt.grid(True)
plt.show()