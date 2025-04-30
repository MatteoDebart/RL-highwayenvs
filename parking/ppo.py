import torch
import torch.nn as nn
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

class NetContinousActions(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_dim, hidden_size, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.log_std = nn.Parameter(
            torch.zeros(act_dim)
        ) 

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp()
        return mean, std
    

class PPO:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        actor_learning_rate,
        critic_learning_rate,
        lambda_=0.95,
        writer=None,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = 0.2

        self.episode_batch_size = episode_batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.loss_function = nn.MSELoss()

        hidden_size = 128

        obs_size = sum([space.shape[0] for space in self.observation_space.values()])
        act_dim = self.action_space.shape[0]  

        self.actor = NetContinousActions(obs_size, hidden_size, act_dim)
        self.critic = Net(obs_size, hidden_size, 1)

        self.optimizer = optim.Adam(
            params=self.actor.parameters(), lr=self.actor_learning_rate
        )

        self.critic_optimizer = optim.Adam(
            params=self.critic.parameters(), lr=self.critic_learning_rate
        )
        self.current_episode = []
        self.episode_reward = 0

        self.scores = []

        self.n_eps = 0
        self.total_steps = 0
        self.critic_updates = 0

    def get_action(self, state, epsilon=None):
        """
        Sample action from Gaussian distribution (for continuous action space).
        """
        state_tensor = torch.cat([torch.tensor(state[key]).float() for key in state.keys()]).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample() 

            return action.squeeze(0).numpy()

    def compute_GAE(self, rewards, terminateds, advantages):
        GAE = 0
        GAE_list = []
        for t in reversed(range(len(rewards))):
            GAE = (1 - terminateds[t]) * GAE
            GAE = advantages[t] + self.gamma * self.lambda_ * GAE
            GAE_list.append(GAE)
        return torch.tensor(GAE_list[::-1], dtype=torch.float32)

    def compute_ppo_score(self):
        states, actions, rewards, terminals, next_states, old_log_probs = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            target_values = (
                rewards
                + self.gamma * (1 - terminals) * self.critic(next_states).squeeze()
            )
            values = self.critic(states).squeeze(1)
            advantages = target_values - values

        GAEs = self.compute_GAE(rewards, terminals, advantages)
        if GAEs.numel() > 1:
            GAEs = (GAEs - GAEs.mean()) / GAEs.std()
        else:
            GAEs = GAEs - GAEs.mean()

        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        ppo_clip_obj = torch.min(ratio * GAEs, clipped_ratio * GAEs)

        return ppo_clip_obj.sum().unsqueeze(0)


    def train_reset(self):
        self.current_episode = []
        self.episode_reward = 0
        self.scores = []

    def update_critic(self, transition):
        state, _, reward, terminated, next_state, _ = transition

        values = self.critic.forward(state)
        with torch.no_grad():
            next_state_values = (1 - terminated) * self.critic(next_state)
            targets = next_state_values * self.gamma + reward

        loss = self.loss_function(values, targets)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

        self.critic_optimizer.step()

    def update(self, state, action, reward, terminated, next_state):
        """
        Update both the actor and critic.
        """
        state_tensor = torch.cat([torch.tensor(state[key]).float() for key in state.keys()]).unsqueeze(0)
        next_state_tensor = torch.cat([torch.tensor(next_state[key]).float() for key in state.keys()]).unsqueeze(0)
        action_tensor = torch.tensor(action)
        
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            old_log_probs = dist.log_prob(action_tensor).sum(dim=-1).unsqueeze(0)

        transition = (
            state_tensor,
            action_tensor.unsqueeze(0),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.int64),
            next_state_tensor,
            old_log_probs,
        )

        self.total_steps += 1
        self.episode_reward += reward

        self.current_episode.append(transition)
        self.update_critic(transition)

        if terminated:
            self.episode_reward = 0
            self.n_eps += 1

            self.scores.append(self.compute_ppo_score())
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.optimizer.step()
                if self.writer:
                    self.writer.add_scalar(
                        "loss/actor", full_neg_score.item(), self.n_eps
                    )

                self.scores = []

    def save(self, path):
        torch.save({
        'actor_state_dict': self.actor.state_dict(),
        'critic_state_dict': self.critic.state_dict(),
        'actor_optimizer_state_dict': self.optimizer.state_dict(),
        'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        'total_steps': self.total_steps,
        'n_eps': self.n_eps
    }, path)


    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.n_eps = checkpoint.get('n_eps', 0)