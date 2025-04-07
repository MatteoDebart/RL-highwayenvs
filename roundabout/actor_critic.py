import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class Net2(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # First hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),  # Output layer
        )

    def forward(self, x):
        return self.net(x)

class ActorCritic:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.learning_rate = learning_rate

        self.reset()

    def _gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into the list of returns * gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T-t-1] + gamma * G
            full_gamma /= gamma
            returns_list.append(full_gamma * G)
        return torch.tensor(returns_list[::-1], dtype=torch.float32)


    def update(self, state, action, reward, terminated, next_state):
        self.current_episode.append((
            torch.tensor(state).flatten().unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor(next_state).flatten(0).unsqueeze(0)
        )
        )

        if terminated:
            self.n_eps += 1

            states, actions, rewards, next_states = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            state_values = self.value_net.forward(states).squeeze()
            next_state_values = self.value_net.forward(next_states).squeeze()

            # Compute TD error for each step
            td_target = rewards + self.gamma * next_state_values
            td_error = td_target - state_values


            log_probs = self.policy_net.forward(states)
            log_probs = log_probs - torch.log(torch.sum(torch.exp(log_probs), dim=1)).unsqueeze(1)

            actor_loss = -torch.sum(log_probs.gather(1, actions) * td_error)
            critic_loss = torch.mean(td_error ** 2) # MSE


            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            

            actor_loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)


            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.current_episode = []
            self.n_eps += 1

    def get_action(self, state, epsilon=None):

        state_tensor = torch.tensor(state).flatten().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net.forward(state_tensor)
            action_probs = torch.softmax(logits, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        return action


        return action

    def reset(self):
        hidden_size = 128

        obs_size = self.observation_space.shape[0] * self.observation_space.shape[1]
        n_actions = self.action_space.n

        self.policy_net = Net(obs_size, hidden_size, n_actions)
        self.value_net = Net(obs_size, hidden_size, 1)

        self.scores = []
        self.current_episode = []

        self.actor_optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            params=self.value_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0
