import pickle
import numpy as np
import yaml
import gymnasium as gym
import highway_env
from DQN import DQN, DQN_collaboration, DQN_noTargetNetwork
import matplotlib.pyplot as plt
from utils import train, eval_agent

if __name__ == "__main__":
    parameters_dict = yaml.load("parameters.yaml")

    with open(parameters_dict["config_pickle"], "rb") as pickle_file:
        config_dict = pickle.load(pickle_file)

    env = gym.make("highway-fast-v0", render_mode=None, config=config_dict)

    action_space = env.action_space
    observation_space = env.observation_space

    arguments = [
        action_space,
        observation_space,
        parameters_dict["gamma"],
        parameters_dict["batch_size"],
        parameters_dict["buffer_capacity"],
        parameters_dict["epsilon_start"],
        parameters_dict["decrease_epsilon_factor"],
        parameters_dict["epsilon_min"],
        parameters_dict["learning_rate"],
    ]

    N_episodes = parameters_dict["N_episodes"]
    if parameters_dict["model_to_use"] == "DQN":
        arguments.append(parameters_dict["update_target_every"])

    model_to_use = {
        "DQN": DQN,
        "DQN_noTargetNetwork": DQN_noTargetNetwork,
        "DQN_collaboration": DQN_collaboration,
    }

    agent = model_to_use[parameters_dict["model_to_use"]](*arguments)

    # Run the training loop
    losses, rewards = train(env, agent, N_episodes)

    plt.plot(losses)
    plt.title("Evolution de la Loss")
    plt.show()

    plt.plot(rewards)
    plt.title("Evolution de la Reward")
    plt.show()

    plt.hist(rewards, bins=20)
    plt.title("Distribution de la Reward")
    plt.show()

    # Evaluate the final policy
    rewards = eval_agent(agent, env, 20)
    print("")
    print("mean reward after training = ", np.mean(rewards))
