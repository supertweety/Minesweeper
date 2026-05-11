"""Utilities for the deep-RL Minesweeper exercise notebook.

This module keeps the notebook short enough to read in class.
You can treat it as infrastructure: the main places to experiment are
the notebook cells that choose network sizes, epsilon schedules, and
learning-rate schedules.
"""

from __future__ import annotations

import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from mine_board import MineBoard

DEFAULT_PLOT_WINDOW = 50


def _plt():
    """Import matplotlib only when a plotting helper is actually used."""
    import matplotlib.pyplot as plt

    return plt


def _markdown_display():
    from IPython.display import Markdown, display

    return Markdown, display


def action_to_rc(action, size):
    return divmod(action, size)


def rc_to_action(row, col, size):
    return row * size + col


def moving_average(values, window):
    if not values:
        return []
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def clone_board(board):
    return [row[:] for row in board]


def plot_board(board, title="", highlight=None, ax=None):
    """Draw a visible Minesweeper board in a notebook-friendly style."""
    plt = _plt()
    rows = len(board)
    cols = len(board[0]) if board else 0
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(2.8, 2.8))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    for row in range(rows):
        for col in range(cols):
            cell = board[row][col]
            facecolor = "#d9dde3"
            text = cell
            text_color = "#222222"

            if cell == "?":
                facecolor = "#7f8fa6"
                text_color = "white"
            elif cell == " ":
                facecolor = "#f4f5f7"
                text = ""

            if highlight == (row, col):
                edgecolor = "#d9534f"
                linewidth = 3.0
            else:
                edgecolor = "black"
                linewidth = 1.5

            rect = plt.Rectangle(
                (col, row),
                1,
                1,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
            )
            ax.add_patch(rect)
            ax.text(
                col + 0.5,
                row + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=15,
                color=text_color,
            )

    if title:
        ax.set_title(title, fontsize=10)

    if created_ax:
        plt.show()


@dataclass
class StepInfo:
    won: bool
    hit_mine: bool
    invalid_action: bool
    row: int
    col: int
    revealed: int
    steps: int


class MinesweeperEnv:
    """Single-agent environment used by the deep-RL exercise.

    State representation:
    - each cell is encoded with a one-hot vector of length 10
    - the whole board is flattened into one long vector

    That representation is not the only possible choice. Part of the
    exercise is to think about whether a different representation or a
    different network architecture would fit the task better.
    """

    HIDDEN_CHANNEL = 0
    EMPTY_CHANNEL = 1
    NUMBER_OFFSET = 1
    CHANNELS_PER_CELL = 10

    def __init__(
        self,
        size=4,
        num_mines=3,
        reveal_reward=0.1,
        win_reward=1.0,
        lose_reward=-1.0,
        invalid_reward=-0.25,
        safe_first_move=True,
        seed=0,
    ):
        self._size = size
        self._num_mines = num_mines
        self._reveal_reward = reveal_reward
        self._win_reward = win_reward
        self._lose_reward = lose_reward
        self._invalid_reward = invalid_reward
        self._safe_first_move = safe_first_move
        self._seed = seed
        self._rng = random.Random(seed)
        self._board = None
        self._moves_taken = 0

    def size(self):
        return self._size

    def action_space_size(self):
        return self._size * self._size

    def state_size(self):
        return self.action_space_size() * self.CHANNELS_PER_CELL

    def reset(self, seed=None):
        if seed is not None:
            self._rng = random.Random(seed)
        self._board = MineBoard(self._size, self._num_mines, rng=self._rng)
        self._moves_taken = 0
        return self.state()

    def legal_actions(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before legal_actions()")
        actions = []
        for row in range(self._size):
            for col in range(self._size):
                if self._board.is_hidden(row, col):
                    actions.append(rc_to_action(row, col, self._size))
        return actions

    def state(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before state()")

        encoded = []
        for row in range(self._size):
            for col in range(self._size):
                channels = [0.0] * self.CHANNELS_PER_CELL
                visible_value = self._board.visible_value(row, col)
                if visible_value == MineBoard.HIDDEN:
                    channels[self.HIDDEN_CHANNEL] = 1.0
                elif visible_value == MineBoard.UNCOVERED:
                    channels[self.EMPTY_CHANNEL] = 1.0
                else:
                    number = int(visible_value)
                    channels[self.NUMBER_OFFSET + number] = 1.0
                encoded.extend(channels)
        return encoded

    def step(self, action):
        if self._board is None:
            raise RuntimeError("reset() must be called before step()")
        if action < 0 or action >= self.action_space_size():
            raise ValueError("action is out of range")

        row, col = action_to_rc(action, self._size)

        # The safe-first-move rule removes some unlucky starts so training
        # does not spend too much time on trivial one-step failures.
        if self._safe_first_move and self._moves_taken == 0 and self._board.has_mine(row, col):
            while self._board.has_mine(row, col):
                self._board = MineBoard(self._size, self._num_mines, rng=self._rng)

        if not self._board.is_hidden(row, col):
            info = StepInfo(
                won=False,
                hit_mine=False,
                invalid_action=True,
                row=row,
                col=col,
                revealed=0,
                steps=self._moves_taken,
            )
            return self.state(), self._invalid_reward, False, info

        revealed_before = self._board.revealed_safe_cells()
        survived = self._board.perform_action(row, col)
        self._moves_taken += 1

        if not survived:
            info = StepInfo(
                won=False,
                hit_mine=True,
                invalid_action=False,
                row=row,
                col=col,
                revealed=0,
                steps=self._moves_taken,
            )
            return self.state(), self._lose_reward, True, info

        revealed = self._board.revealed_safe_cells() - revealed_before
        reward = revealed * self._reveal_reward
        won = self._board.is_solved()
        if won:
            reward += self._win_reward

        info = StepInfo(
            won=won,
            hit_mine=False,
            invalid_action=False,
            row=row,
            col=col,
            revealed=revealed,
            steps=self._moves_taken,
        )
        return self.state(), reward, won, info

    def render(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before render()")
        return self._board.board()


class ConstantSchedule:
    """Always return the same value."""

    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value


class LinearSchedule:
    """Linearly interpolate from start to end over a fixed duration."""

    def __init__(self, start, end, duration):
        self.start = start
        self.end = end
        self.duration = max(1, duration)

    def __call__(self, step):
        mix = min(1.0, step / self.duration)
        return self.start + mix * (self.end - self.start)


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


class CNNQNetwork(nn.Module):
    """Convolutional baseline for DQN.

    The environment still emits one flattened vector, but this network
    reshapes it back into a board so it can use spatial structure.
    The selected action is encoded as one extra input channel, and the
    network predicts one scalar value Q(s, a).
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(128, 64),
        channels_per_cell=MinesweeperEnv.CHANNELS_PER_CELL,
        conv_channels=(32, 64),
    ):
        super().__init__()
        hidden_one, hidden_two = hidden_sizes
        conv_one, conv_two = conv_channels

        self.channels_per_cell = channels_per_cell
        cells = input_size // channels_per_cell
        self.board_size = math.isqrt(cells)
        if self.board_size * self.board_size * channels_per_cell != input_size:
            raise ValueError(
                "input_size must match board_size * board_size * channels_per_cell"
            )

        self.features = nn.Sequential(
            nn.Conv2d(channels_per_cell + 1, conv_one, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_one, conv_two, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_output_size = conv_two * self.board_size * self.board_size
        self.head = nn.Sequential(
            nn.Linear(conv_output_size, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, 1),
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.long, device=state.device)
        else:
            action = action.to(device=state.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        batch_size = state.shape[0]
        if action.shape[0] != batch_size:
            raise ValueError("state batch and action batch must have the same size")

        x = state.reshape(
            batch_size,
            self.board_size,
            self.board_size,
            self.channels_per_cell,
        )
        x = x.permute(0, 3, 1, 2)
        action_channel = torch.zeros(
            batch_size,
            1,
            self.board_size,
            self.board_size,
            dtype=state.dtype,
            device=state.device,
        )
        rows = torch.div(action, self.board_size, rounding_mode="floor")
        cols = action % self.board_size
        action_channel[torch.arange(batch_size, device=state.device), 0, rows, cols] = 1.0
        x = torch.cat([x, action_channel], dim=1)
        x = self.features(x)
        x = x.reshape(batch_size, -1)
        return self.head(x).squeeze(-1)


class MLPPolicyNetwork(nn.Module):
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


class MLPValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=(128, 64)):
        super().__init__()
        hidden_one, hidden_two = hidden_sizes
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, 1),
        )

    def forward(self, state):
        return self.layers(state).squeeze(-1)


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


class DQNAgent:
    """A small DQN baseline.

    The implementation is intentionally straightforward so that the tuning
    questions remain visible. This is not meant to be an industrial-strength
    DQN implementation.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=(128, 64),
        gamma=0.99,
        batch_size=64,
        replay_capacity=5000,
        target_sync_every=100,
        epsilon_schedule=None,
        learning_rate_schedule=None,
        seed=0,
        device=None,
        name="dqn",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.hidden_sizes = tuple(hidden_sizes)
        self.name = name
        self.device = torch.device(device or self._default_device())
        self._rng = random.Random(seed)
        torch.manual_seed(seed)

        self.epsilon_schedule = epsilon_schedule or LinearSchedule(1.0, 0.05, 200)
        self.learning_rate_schedule = learning_rate_schedule or ConstantSchedule(1e-3)
        self.current_epsilon = self.epsilon_schedule(0)
        self.current_learning_rate = self.learning_rate_schedule(0)
        self.episode_index = 0
        self.train_steps = 0

        self.q_network = CNNQNetwork(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.target_network = CNNQNetwork(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.current_learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity)

    def _default_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start_episode(self, training=True):
        if training:
            self.current_epsilon = self.epsilon_schedule(self.episode_index)
            self.current_learning_rate = self.learning_rate_schedule(self.episode_index)
            set_optimizer_lr(self.optimizer, self.current_learning_rate)
            self.episode_index += 1
        else:
            self.current_epsilon = 0.0

    def _q_values_for_actions(self, network, state, actions):
        if not actions:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(device=self.device, dtype=torch.float32)

        if state.dim() == 1:
            states = state.unsqueeze(0).repeat(len(actions), 1)
        elif state.dim() == 2 and state.shape[0] == 1:
            states = state.repeat(len(actions), 1)
        else:
            states = state

        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        return network(states, action_tensor)

    def select_action(self, state, legal_actions, training=True):
        legal_actions = list(legal_actions)
        if training and self._rng.random() < self.current_epsilon:
            return self._rng.choice(legal_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self._q_values_for_actions(self.q_network, state_tensor, legal_actions)

        best_index = int(torch.argmax(q_values).item())
        return legal_actions[best_index]

    def observe_transition(self, state, action, reward, next_state, next_legal_actions, done, training=True):
        if not training:
            return None

        self.replay_buffer.add(
            Experience(
                state=list(state),
                action=action,
                reward=reward,
                next_state=list(next_state),
                done=done,
                next_actions=list(next_legal_actions),
            )
        )

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self._rng)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(item.done) for item in batch], dtype=torch.float32, device=self.device)

        predicted_q = self.q_network(states, actions)

        with torch.no_grad():
            max_next_q_values = []
            for item in batch:
                if item.done or not item.next_actions:
                    max_next_q_values.append(0.0)
                else:
                    next_state_tensor = torch.tensor(item.next_state, dtype=torch.float32, device=self.device)
                    next_values = self._q_values_for_actions(
                        self.target_network,
                        next_state_tensor,
                        item.next_actions,
                    )
                    max_next_q_values.append(torch.max(next_values).item())
            max_next_q_values = torch.tensor(max_next_q_values, dtype=torch.float32, device=self.device)
            targets = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        loss = torch.nn.functional.mse_loss(predicted_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_sync_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "policy_loss": None,
            "value_loss": None,
            "entropy": None,
        }

    def end_episode(self, training=True):
        return {}


def masked_categorical(logits, legal_actions, device):
    masked_logits = torch.full_like(logits, -1e9)
    indexes = torch.tensor(list(legal_actions), dtype=torch.long, device=device)
    masked_logits[indexes] = logits[indexes]
    return torch.distributions.Categorical(logits=masked_logits)


class REINFORCEAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=(128, 64),
        gamma=0.99,
        entropy_coef=0.01,
        learning_rate_schedule=None,
        seed=0,
        device=None,
        name="reinforce",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.name = name
        self.device = torch.device(device or self._default_device())
        self._rng = random.Random(seed)
        torch.manual_seed(seed)

        self.learning_rate_schedule = learning_rate_schedule or ConstantSchedule(5e-4)
        self.current_learning_rate = self.learning_rate_schedule(0)
        self.episode_index = 0

        self.policy_network = MLPPolicyNetwork(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.current_learning_rate)

    def _default_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start_episode(self, training=True):
        if training:
            self.current_learning_rate = self.learning_rate_schedule(self.episode_index)
            set_optimizer_lr(self.optimizer, self.current_learning_rate)
            self.episode_index += 1
        self._log_probs = []
        self._rewards = []
        self._entropies = []

    def select_action(self, state, legal_actions, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_network(state_tensor).squeeze(0)
        distribution = masked_categorical(logits, legal_actions, self.device)

        if training:
            action_tensor = distribution.sample()
            self._log_probs.append(distribution.log_prob(action_tensor))
            self._entropies.append(distribution.entropy())
        else:
            action_tensor = torch.argmax(distribution.probs)

        return int(action_tensor.item())

    def observe_transition(self, state, action, reward, next_state, next_legal_actions, done, training=True):
        if training:
            self._rewards.append(reward)
        return None

    def _discounted_returns(self, rewards):
        returns = []
        running_total = 0.0
        for reward in reversed(rewards):
            running_total = reward + self.gamma * running_total
            returns.append(running_total)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def end_episode(self, training=True):
        if not training or not self._rewards:
            return {}

        returns = self._discounted_returns(self._rewards)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)
        policy_loss = -(log_probs * returns).sum() - self.entropy_coef * entropies.sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": None,
            "policy_loss": float(policy_loss.item()),
            "value_loss": None,
            "entropy": float(entropies.mean().item()),
        }


class ActorCriticAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=(128, 64),
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        learning_rate_schedule=None,
        seed=0,
        device=None,
        name="actor_critic",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.name = name
        self.device = torch.device(device or self._default_device())
        self._rng = random.Random(seed)
        torch.manual_seed(seed)

        self.learning_rate_schedule = learning_rate_schedule or ConstantSchedule(5e-4)
        self.current_learning_rate = self.learning_rate_schedule(0)
        self.episode_index = 0

        self.policy_network = MLPPolicyNetwork(state_size, action_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.value_network = MLPValueNetwork(state_size, hidden_sizes=self.hidden_sizes).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.current_learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.current_learning_rate)

    def _default_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start_episode(self, training=True):
        if training:
            self.current_learning_rate = self.learning_rate_schedule(self.episode_index)
            set_optimizer_lr(self.policy_optimizer, self.current_learning_rate)
            set_optimizer_lr(self.value_optimizer, self.current_learning_rate)
            self.episode_index += 1
        self._last_log_prob = None
        self._last_entropy = None
        self._last_value = None

    def select_action(self, state, legal_actions, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_network(state_tensor).squeeze(0)
        distribution = masked_categorical(logits, legal_actions, self.device)

        if training:
            action_tensor = distribution.sample()
            self._last_log_prob = distribution.log_prob(action_tensor)
            self._last_entropy = distribution.entropy()
            self._last_value = self.value_network(state_tensor).squeeze(0)
        else:
            action_tensor = torch.argmax(distribution.probs)

        return int(action_tensor.item())

    def observe_transition(self, state, action, reward, next_state, next_legal_actions, done, training=True):
        if not training:
            return None

        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if done:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_value = self.value_network(next_state_tensor).squeeze(0)

        target = reward_tensor + self.gamma * next_value
        advantage = target - self._last_value

        policy_loss = -self._last_log_prob * advantage.detach()
        value_loss = advantage.pow(2)
        entropy_bonus = self._last_entropy

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)

        self.policy_optimizer.step()
        self.value_optimizer.step()

        return {
            "loss": None,
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }

    def end_episode(self, training=True):
        return {}


def run_episode(env, agent, training=True, max_steps=None, capture=False):
    max_steps = max_steps or env.action_space_size() * 2
    state = env.reset()
    agent.start_episode(training=training)
    total_reward = 0.0
    won = False
    metrics = []
    trajectory = []

    if capture:
        trajectory.append(
            {
                "board": clone_board(env.render()),
                "action": None,
                "reward": 0.0,
                "total_reward": 0.0,
                "note": "Initial board",
                "highlight": None,
            }
        )

    for step_index in range(max_steps):
        legal_actions = env.legal_actions()
        action = agent.select_action(state, legal_actions, training=training)
        next_state, reward, done, info = env.step(action)
        next_legal_actions = [] if done else env.legal_actions()

        metric = agent.observe_transition(
            state,
            action,
            reward,
            next_state,
            next_legal_actions,
            done,
            training=training,
        )
        if metric:
            metrics.append(metric)

        total_reward += reward

        if capture:
            if info.hit_mine:
                note = "The agent hit a mine."
            elif info.won:
                note = "The board is solved."
            elif info.invalid_action:
                note = "Invalid action."
            else:
                note = f"Revealed {info.revealed} safe cell(s)."

            trajectory.append(
                {
                    "board": clone_board(env.render()),
                    "action": action,
                    "reward": reward,
                    "total_reward": total_reward,
                    "note": note,
                    "highlight": (info.row, info.col),
                }
            )

        state = next_state
        if done:
            won = info.won
            break

    final_metric = agent.end_episode(training=training)
    if final_metric:
        metrics.append(final_metric)

    return {
        "reward": total_reward,
        "won": won,
        "steps": step_index + 1,
        "metrics": metrics,
        "trajectory": trajectory,
        "board": clone_board(env.render()),
        "epsilon": getattr(agent, "current_epsilon", None),
        "learning_rate": getattr(agent, "current_learning_rate", None),
    }


def train_agent(
    agent_builder,
    env_config,
    episodes=1000,
    report_every=25,
    max_steps=None,
    base_seed=0,
    plot_window=DEFAULT_PLOT_WINDOW,
):
    env = MinesweeperEnv(**env_config, seed=base_seed)
    agent = agent_builder()
    max_steps = max_steps or env.action_space_size() * 2

    rewards = []
    wins = []
    steps = []
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    epsilons = []
    learning_rates = []
    recent_rewards = deque(maxlen=report_every)
    recent_wins = deque(maxlen=report_every)

    for episode in range(episodes):
        outcome = run_episode(env, agent, training=True, max_steps=max_steps, capture=False)
        rewards.append(outcome["reward"])
        wins.append(int(outcome["won"]))
        steps.append(outcome["steps"])
        epsilons.append(outcome["epsilon"])
        learning_rates.append(outcome["learning_rate"])
        recent_rewards.append(outcome["reward"])
        recent_wins.append(int(outcome["won"]))

        for metric in outcome["metrics"]:
            if metric.get("loss") is not None:
                losses.append(metric["loss"])
            if metric.get("policy_loss") is not None:
                policy_losses.append(metric["policy_loss"])
            if metric.get("value_loss") is not None:
                value_losses.append(metric["value_loss"])
            if metric.get("entropy") is not None:
                entropies.append(metric["entropy"])

        if (episode + 1) % report_every == 0:
            print(
                f"Episode {episode + 1:4d} | avg reward "
                f"{sum(recent_rewards) / len(recent_rewards):6.3f} | "
                f"win rate {sum(recent_wins) / len(recent_wins):5.1%}",
                flush=True,
            )

    return {
        "agent": agent,
        "episode_rewards": rewards,
        "smoothed_rewards": moving_average(rewards, window=plot_window),
        "rolling_win_rates": moving_average(wins, window=plot_window),
        "steps": steps,
        "losses": losses,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies,
        "epsilons": epsilons,
        "learning_rates": learning_rates,
    }


def evaluate_agent(agent, env_config, episodes=30, max_steps=None, capture=False, base_seed=10_000):
    env = MinesweeperEnv(**env_config, seed=base_seed)
    max_steps = max_steps or env.action_space_size() * 2
    rewards = []
    wins = 0
    rollout = None
    last_board = None
    lengths = []

    for episode in range(episodes):
        outcome = run_episode(
            env,
            agent,
            training=False,
            max_steps=max_steps,
            capture=capture and episode == episodes - 1,
        )
        rewards.append(outcome["reward"])
        wins += int(outcome["won"])
        lengths.append(outcome["steps"])
        last_board = outcome["board"]
        if capture and episode == episodes - 1:
            rollout = outcome["trajectory"]

    return {
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "win_rate": wins / episodes if episodes else 0.0,
        "average_steps": sum(lengths) / len(lengths) if lengths else 0.0,
        "last_board": last_board,
        "rollout": rollout,
    }


def compare_agents(
    agent_builders,
    env_config,
    train_episodes=1000,
    eval_episodes=30,
    max_steps=None,
    plot_window=DEFAULT_PLOT_WINDOW,
    report_every=25,
):
    results = {}
    for offset, (name, builder) in enumerate(agent_builders.items()):
        print(f"\nTraining {name} ...", flush=True)
        training = train_agent(
            builder,
            env_config,
            episodes=train_episodes,
            report_every=report_every,
            max_steps=max_steps,
            base_seed=1_000 * offset,
            plot_window=plot_window,
        )
        evaluation = evaluate_agent(
            training["agent"],
            env_config,
            episodes=eval_episodes,
            max_steps=max_steps,
            capture=True,
            base_seed=20_000 + 1_000 * offset,
        )
        results[name] = {
            **training,
            **evaluation,
        }
    return results


def plot_training_rewards(results, window=DEFAULT_PLOT_WINDOW):
    plt = _plt()
    plt.figure(figsize=(8, 4.5))
    for name, result in results.items():
        plt.plot(result["smoothed_rewards"], label=name)
    plt.xlabel("Episode")
    plt.ylabel(f"Moving average reward (window={window})")
    plt.title("Training reward by agent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_win_rates(results, window=DEFAULT_PLOT_WINDOW):
    plt = _plt()
    plt.figure(figsize=(8, 4.5))
    for name, result in results.items():
        plt.plot(result["rolling_win_rates"], label=name)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling win rate (window={window})")
    plt.title("Training win rate by agent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_schedule_traces(results):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    plotted_any = False

    for name, result in results.items():
        epsilons = [value for value in result["epsilons"] if value is not None]
        learning_rates = [value for value in result["learning_rates"] if value is not None]
        if epsilons:
            axes[0].plot(epsilons, label=name)
            plotted_any = True
        if learning_rates:
            axes[1].plot(learning_rates, label=name)
            plotted_any = True

    axes[0].set_title("Epsilon by episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Epsilon")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Learning rate by episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    if plotted_any:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
        print("No schedule traces available.")


def show_results_table(results):
    Markdown, display = _markdown_display()
    lines = [
        "| agent | eval avg reward | eval win rate | eval avg steps |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['average_reward']:.2f} | "
            f"{result['win_rate']:.2f} | {result['average_steps']:.2f} |"
        )
    display(Markdown("\n".join(lines)))


def show_rollout(results, agent_name, size):
    plt = _plt()
    trajectory = results[agent_name]["rollout"]
    if not trajectory:
        print("No rollout stored for this agent.")
        return

    columns = len(trajectory)
    fig, axes = plt.subplots(1, columns, figsize=(3.0 * columns, 3.0))
    if columns == 1:
        axes = [axes]

    for ax, frame in zip(axes, trajectory):
        plot_board(
            frame["board"],
            title=frame["title"] if "title" in frame else frame["note"],
            highlight=frame.get("highlight"),
            ax=ax,
        )

    fig.suptitle(f"Sample evaluation rollout: {agent_name}", fontsize=12)
    plt.tight_layout()
    plt.show()


__all__ = [
    "ActorCriticAgent",
    "ConstantSchedule",
    "DEFAULT_PLOT_WINDOW",
    "DQNAgent",
    "LinearSchedule",
    "MinesweeperEnv",
    "REINFORCEAgent",
    "compare_agents",
    "plot_training_rewards",
    "plot_win_rates",
    "plot_schedule_traces",
    "show_results_table",
    "show_rollout",
]
