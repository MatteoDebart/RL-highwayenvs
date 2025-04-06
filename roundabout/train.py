from eval import eval_agent
import numpy as np
from tqdm import tqdm

def train(env, agent, N_episodes, eval_every=100, reward_threshold=300, n_eval=10):
    total_time = 0
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        while not done: 
            action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, terminated, next_state)

            state = next_state

            done = terminated or truncated
            total_time += 1

        if ((ep+1)% eval_every == 0):
            mean_reward = np.mean(eval_agent(agent, env, n_sim=n_eval))
            print("episode =", ep+1, ", reward = ", mean_reward)
            if mean_reward >= reward_threshold:
                break
                
    return 