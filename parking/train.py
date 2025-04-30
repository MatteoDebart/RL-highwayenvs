from eval import eval_agent, custom_eval_agent
import numpy as np
import highway_env

def train(env, agent, N_episodes, eval_every=100, reward_threshold=300, n_eval=10):
    total_time = 0
    rewards = []
    episodes = []
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        state = {"observation" : state["observation"], "desired_goal" : state["desired_goal"]}        

        total_reward = 0
        while not done: 
            action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = {"observation" : next_state["observation"], "desired_goal" : next_state["desired_goal"]}
            agent.update(state, action, reward, done, next_state)
            state = next_state
            
            total_time += 1
            total_reward += reward 


        if ((ep+1)% eval_every == 0):
            mean_reward = np.mean(eval_agent(agent, env, n_sim=n_eval))
            print("episode =", ep+1, ", reward = ", mean_reward)
            rewards.append(mean_reward)
            episodes.append(ep+1)
            if mean_reward >= reward_threshold:
                break
    return rewards, episodes