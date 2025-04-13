from eval import eval_agent, custom_eval_agent
import numpy as np
import highway_env
from tqdm import tqdm

def train(env, agent, N_episodes, eval_every=100, reward_threshold=300, n_eval=10):
    total_time = 0
    rewards = []
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        total_reward = 0
        entered_roundabout = False
        while not done: 
            additional_reward = 0
            action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            done = terminated or truncated or not env.unwrapped.vehicle.on_road

            

            lane = env.unwrapped.road.network.get_lane(env.unwrapped.vehicle.lane_index)

            if isinstance(lane, highway_env.road.lane.CircularLane):
                entered_roundabout = True
            
            if isinstance(lane, highway_env.road.lane.StraightLane) and entered_roundabout:
                entered_roundabout = False
                additional_reward += 5

            if not env.unwrapped.vehicle.on_road:
                additional_reward += 10

            agent.update(state, action, reward +additional_reward, done, next_state) 
            state = next_state

            total_time += 1
            total_reward += reward + additional_reward
        print(f"Episode {ep} ended with reward {total_reward}")

        '''if ((ep+1)% eval_every == 0):
            mean_reward = np.mean(custom_eval_agent(agent, env, n_sim=n_eval))
            print("episode =", ep+1, ", reward = ", mean_reward)
            rewards.append(mean_reward)
            if mean_reward >= reward_threshold:
                break'''
                
    return rewards