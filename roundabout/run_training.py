import gymnasium as gym
import numpy as np
import highway_env
import time

from reinforce_agent import  REINFORCE
from eval import eval_agent
from train import train

env = gym.make("roundabout-v0", render_mode='rgb_array')

action_space = env.action_space
observation_space = env.observation_space

gamma = .95
episode_batch_size = 16
learning_rate = 2e-4

agent = REINFORCE(
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        )
N_episodes = 1000


start = time.time()
print("mean reward before training = ", np.mean(eval_agent(agent, env, 100)))
# Run the training loop
train(env, agent, N_episodes, eval_every=50,)

# Evaluate the final policy
print("mean reward after training = ", np.mean(eval_agent(agent, env, 100)))

print(f"Training length: {time.time() - start} secondes")