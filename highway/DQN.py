import torch
import numpy as np
from DQN_Base import DQN_Base


class DQN(DQN_Base):
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
        update_target_every,
    ):
        super(DQN_Base).__init__(
            action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            epsilon_start,
            decrease_epsilon_factor,
            epsilon_min,
            learning_rate,
            update_target_every,
        )

    def update(self, state, action, reward, terminated, next_state):
        self.buffer.push(
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        batch = self.buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*batch)]
        )

        # Compute Q-values for current states
        q_values = self.q_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (
                1 - done_batch
            )

        loss = self.loss_function(q_values.float(), target_q_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()
        self.n_steps += 1
        return loss.detach().numpy()


class DQN_collaboration(DQN_Base):
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
    ):
        super(DQN_Base).__init__(
            action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            epsilon_start,
            decrease_epsilon_factor,
            epsilon_min,
            learning_rate,
        )

    def update(self, state, action, reward, terminated, next_state):
        self.buffer.push(
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        batch = self.buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*batch)]
        )

        # Compute Q-values for current states
        q_values = self.q_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (
                1 - done_batch
            )

        loss = self.loss_function(q_values.float(), target_q_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute Q-values for current states
        q_values2 = self.target_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values2 = self.q_net(next_state_batch).max(1)[0]
            target_q_values2 = reward_batch + self.gamma * max_next_q_values2 * (
                1 - done_batch
            )

        loss = self.loss_function(q_values2.float(), target_q_values2.float())
        self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer2.step()

        self.decrease_epsilon()
        self.n_steps += 1
        return loss.detach().numpy()


class DQN_noTargetNetwork(DQN_Base):
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
    ):
        super(DQN_Base).__init__(
            action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            epsilon_start,
            decrease_epsilon_factor,
            epsilon_min,
            learning_rate,
        )

    def update(self, state, action, reward, terminated, next_state):
        self.buffer.push(
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        batch = self.buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*batch)]
        )

        # Compute Q-values for current states
        q_values = self.q_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the network
        max_next_q_values = self.q_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + self.gamma * max_next_q_values * (
            1 - done_batch
        )

        loss = self.loss_function(q_values.float(), target_q_values.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.decrease_epsilon()
        self.n_steps += 1
        return loss.detach().numpy()
