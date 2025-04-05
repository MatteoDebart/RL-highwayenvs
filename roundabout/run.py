import gymnasium as gym
import highway_env
import time
import numpy as np

from eval import eval_agent
from reinforce_agent import REINFORCE_SKELETON
env = gym.make("roundabout-v0", render_mode = "human")
env.reset()

action_space = env.action_space
observation_space = env.observation_space
gamma = 0.99
episode_batch_size = 1
learning_rate = 1e-2

agent = REINFORCE_SKELETON(action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,)

print(f'Average over 5 runs : {np.mean(eval_agent(agent, env))}')