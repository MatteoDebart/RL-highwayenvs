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


class REINFORCE_SKELETON:
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

        self.episode_batch_size = episode_batch_size
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
        )
        )

        if terminated: 
            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self._gradient_returns(rewards, self.gamma)
            unn_log_probs = self.policy_net.forward(states)
            log_probs = unn_log_probs - torch.log(torch.sum(torch.exp(unn_log_probs), dim=1)).unsqueeze(1)


            full_neg_score = - torch.dot(log_probs.gather(1, actions).squeeze(), current_episode_returns).unsqueeze(0)#).sum()


            self.current_episode = []

            self.optimizer.zero_grad()
            full_neg_score.backward()
            self.optimizer.step()
    

    def get_action(self, state, epsilon=None):

        state_tensor = torch.tensor(state).flatten().unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net.forward(state_tensor)
            action_probs = torch.softmax(logits, dim=1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
        return action


    def reset(self):
        hidden_size = 128

        obs_size = self.observation_space.shape[0] * self.observation_space.shape[1]
        n_actions = self.action_space.n

        self.policy_net = Net(obs_size, hidden_size, n_actions)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0
