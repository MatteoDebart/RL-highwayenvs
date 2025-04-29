import numpy as np

def eval_agent(agent, env, n_sim=5):
    episode_rewards = np.zeros(n_sim)
    agent.eval()

    for i in range(n_sim):
        done = False
        truncated = False
        state, _ = env.reset()

        rewards = 0

        while not (done or truncated):
            action = agent.get_action(state, epsilon=0)
            state, reward, done, truncated, _ = env.step(action)
            rewards += reward

        episode_rewards[i] = rewards

    return episode_rewards

def train(env, agent, N_episodes, eval_every=10, reward_threshold=300):
    agent.train()
    total_time = 0
    state, _ = env.reset()
    losses = []
    rewards = []
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        tot_loss = 0
        tot_reward = 0
        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            loss_val = agent.update(state, action, reward, terminated, next_state)

            state = next_state
            tot_loss += loss_val
            tot_reward += reward

            done = terminated or truncated
            total_time += 1

        losses.append(tot_loss)
        rewards.append(tot_reward)

        if (ep + 1) % eval_every == 0:
            reward = eval_agent(agent, env)
            print("episode =", ep + 1, ", reward = ", np.mean(reward))
            if np.mean(reward) >= reward_threshold:
                break

    return losses, rewards