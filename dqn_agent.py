import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    next_actions: list[int]


class ReplayBuffer:
    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def add(self, experience):
        self._buffer.append(experience)

    def sample(self, batch_size, rng):
        return rng.sample(list(self._buffer), batch_size)

    def __len__(self):
        return len(self._buffer)


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(128, 64)):
        super().__init__()
        hidden_one, hidden_two = hidden_sizes
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, output_size),
        )

    def forward(self, state):
        return self.layers(state)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        seed=None,
        hidden_sizes=(128, 64),
        gamma=0.99,
        learning_rate=0.001,
        batch_size=32,
        replay_capacity=10000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_sync_steps=100,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_sync_steps = target_sync_steps
        self.device = torch.device(device or self._default_device())

        self._rng = random.Random(seed)
        self._train_steps = 0

        if seed is not None:
            torch.manual_seed(seed)

        self.q_network = QNetwork(state_size, action_size, self.hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_size, action_size, self.hidden_sizes).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity)

    def _default_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def select_action(self, state, available_actions, explore=True):
        if not available_actions:
            raise ValueError("available_actions must not be empty")

        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(available_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
        self.q_network.train()

        best_action = available_actions[0]
        best_value = q_values[best_action].item()
        for action in available_actions[1:]:
            value = q_values[action].item()
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def remember(self, state, action, reward, next_state, done, next_actions):
        self.replay_buffer.add(
            Experience(
                state=list(state),
                action=action,
                reward=reward,
                next_state=list(next_state),
                done=done,
                next_actions=list(next_actions),
            )
        )

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self._rng)
        states = torch.tensor(
            [experience.state for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [experience.action for experience in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [experience.reward for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [float(experience.done) for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            [experience.next_state for experience in batch],
            dtype=torch.float32,
            device=self.device,
        )

        predicted_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            masked_next_q = []
            for row, experience in zip(next_q_values, batch):
                if experience.done or not experience.next_actions:
                    masked_next_q.append(0.0)
                    continue
                action_indexes = torch.tensor(
                    experience.next_actions,
                    dtype=torch.long,
                    device=self.device,
                )
                masked_next_q.append(torch.max(row[action_indexes]).item())

            max_next_q_values = torch.tensor(
                masked_next_q,
                dtype=torch.float32,
                device=self.device,
            )
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        loss = functional.mse_loss(predicted_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_sync_steps == 0:
            self.sync_target_network()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def sync_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(
            {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "hidden_sizes": list(self.hidden_sizes),
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "target_sync_steps": self.target_sync_steps,
                "model_state_dict": self.q_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location=device or "cpu")
        agent = cls(
            payload["state_size"],
            payload["action_size"],
            hidden_sizes=tuple(payload["hidden_sizes"]),
            gamma=payload["gamma"],
            learning_rate=payload["learning_rate"],
            batch_size=payload["batch_size"],
            epsilon_start=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            target_sync_steps=payload["target_sync_steps"],
            device=device,
        )
        agent.q_network.load_state_dict(payload["model_state_dict"])
        agent.target_network.load_state_dict(payload["model_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return agent
