import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random


class DQN_Base:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        update_target_every=0,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.reset()

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return self.action_space.sample()  # Explore
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.get_q(state)
            return np.argmax(q_values)  # Exploit

    def update(self, state, action, reward, terminated, next_state):
        pass

    def get_q(self, state):
        state_tensor = state
        with torch.no_grad():
            output = self.q_net(state_tensor)  # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )

    def reset(self):
        obs_shape = self.observation_space.shape
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = Net(obs_shape, n_actions)
        self.target_net = Net(obs_shape, n_actions)

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )
        self.optimizer2 = optim.Adam(
            params=self.target_net.parameters(), lr=self.learning_rate
        )  # Used in collaboration DQN

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def train(self):
        self.q_net.train()

    def eval(self):
        self.q_net.eval()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


def mult(list):
    result = 1
    for i in list:
        result *= i

    return result


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mult(obs_size), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
