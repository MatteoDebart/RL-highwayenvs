import gymnasium as gym
import numpy as np
import highway_env


from reinforce_agent import  REINFORCE_SKELETON
from eval import eval_agent
env = gym.make("roundabout-v0", render_mode='rgb_array')

action_space = env.action_space
observation_space = env.observation_space

gamma = .95
episode_batch_size = 1
learning_rate = 1e-2

agent = REINFORCE_SKELETON(
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        )
N_episodes = 300


print("mean reward before training = ", np.mean(eval_agent(agent, env, 200)))
# Run the training loop
#train(env, agent, N_episodes, eval_every=50,)

# Evaluate the final policy
print("mean reward after training = ", np.mean(eval_agent(agent, env, 200)))