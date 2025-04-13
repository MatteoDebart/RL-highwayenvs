from copy import deepcopy
import numpy as np
import highway_env
from tqdm import tqdm

def custom_eval_agent(agent, env, n_sim = 10):
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


        episode_rewards[i] = reward_sum + additional_reward
    return episode_rewards


def eval_agent(agent, env, n_sim=10):
    """    
    Monte Carlo evaluation of the agent.

    Repeat n_sim times:
        * Run the agent policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    print(f"Evaluating agent {n_sim} times")
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        #print(i)
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        while not done: 
            action = agent.get_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards