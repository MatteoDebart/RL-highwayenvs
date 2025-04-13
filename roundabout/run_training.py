import gymnasium as gym
import numpy as np
import highway_env
import time

from reinforce_agent import  REINFORCE
from actor_critic import ActorCritic, TDActorCriticBasic
from ppo import PPO
from eval import eval_agent
from train import train

from torch.utils.tensorboard import SummaryWriter

env = gym.make("roundabout-v0", render_mode='human')
env.unwrapped.configure({
    "duration": 20})

action_space = env.action_space
observation_space = env.observation_space

gamma = .97
episode_batch_size = 16
learning_rate = 1e-3

'''agent = ActorCritic(
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        )

agent = TDActorCriticBasic(action_space,
        observation_space,
        gamma,
        learning_rate,
        learning_rate)'''

counter = 0
writer = SummaryWriter(log_dir=f"runs/exp{counter}")


agent = PPO(action_space,
        observation_space,
        gamma,
        16,
        learning_rate,
        learning_rate, 
        writer=writer)
N_episodes = 1000


start = time.time()
#print("mean reward before training = ", np.mean(eval_agent(agent, env, 10)))
# Run the training loop
rewards = train(env, agent, N_episodes, eval_every=16)

# Evaluate the final policy
#print("mean reward after training = ", np.mean(eval_agent(agent, env, 10)))

print(f"Training length: {time.time() - start} secondes")

import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward over Episodes")
plt.grid(True)
plt.show()