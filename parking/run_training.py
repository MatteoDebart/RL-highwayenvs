import gymnasium as gym
import numpy as np
import highway_env
import time
import pickle
import matplotlib.pyplot as plt

from ppo import PPO
from eval import eval_agent
from train import train

with open('parking/config_task2.pkl', 'rb') as f:
    config = pickle.load(f)

env = gym.make("parking-v0", render_mode = "rgb_array", config = config)


action_space = env.action_space
observation_space = {"observation" : env.observation_space["observation"], "desired_goal" : env.observation_space["desired_goal"],}
gamma = .99
episode_batch_size = 16
learning_rate = 1e-3


agent = PPO(action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        learning_rate, 
)
N_episodes = 2000


start = time.time()
print("mean reward before training = ", np.mean(eval_agent(agent, env, 10)))

# Run the training loop
rewards, eps = train(env, agent, N_episodes, eval_every=episode_batch_size, n_eval=25)


# Evaluate the final policy
print("mean reward after training = ", np.mean(eval_agent(agent, env, 10)))


agent.save("parking/ppo_parking_agent.pth") #Save the agent at the end of the training


plt.plot(eps, rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward over Episodes")
plt.grid(True)
plt.show()