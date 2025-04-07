import gymnasium as gym
import numpy as np
import highway_env
import time

from reinforce_agent import  REINFORCE
from actor_critic import ActorCritic
from eval import eval_agent
from train import train

env = gym.make("roundabout-v0", render_mode='rgb_array')
action_space = env.action_space
observation_space = env.observation_space

gamma = .97
episode_batch_size = 10
learning_rate = 1e-3

agent = ActorCritic(
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        )
N_episodes = 1000


start = time.time()
print("mean reward before training = ", np.mean(eval_agent(agent, env, 10)))
# Run the training loop
train(env, agent, N_episodes, eval_every=10,)

# Evaluate the final policy
print("mean reward after training = ", np.mean(eval_agent(agent, env, 10)))

print(f"Training length: {time.time() - start} secondes")