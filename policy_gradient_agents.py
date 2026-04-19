import random
from collections import deque

import torch
import torch.nn.functional as functional
from torch import nn


class PolicyNetwork(nn.Module):
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


class ValueNetwork(nn.Module):
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


class PolicyGradientAgent:
    def __init__(
        self,
        state_size,
        action_size,
        seed=None,
        hidden_sizes=(128, 64),
        gamma=0.99,
        learning_rate=0.001,
        entropy_coef=0.01,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device or self._default_device())
        self._rng = random.Random(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.policy_network = PolicyNetwork(
            state_size,
            action_size,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

    def _default_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start_episode(self):
        pass

    def _masked_distribution(self, state, available_actions):
        if not available_actions:
            raise ValueError("available_actions must not be empty")

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        logits = self.policy_network(state_tensor).squeeze(0)
        masked_logits = torch.full_like(logits, -1e9)
        action_indexes = torch.tensor(
            available_actions,
            dtype=torch.long,
            device=self.device,
        )
        masked_logits[action_indexes] = logits[action_indexes]
        distribution = torch.distributions.Categorical(logits=masked_logits)
        return state_tensor, distribution

    def select_action(self, state, available_actions, explore=True, track=True):
        _, distribution = self._masked_distribution(state, available_actions)
        if explore:
            action_tensor = distribution.sample()
        else:
            action_tensor = torch.argmax(distribution.probs)
        action = int(action_tensor.item())
        if track:
            self._record_action(state, distribution, action_tensor)
        return action

    def _record_action(self, state, distribution, action_tensor):
        raise NotImplementedError

    def observe(self, reward, next_state, next_available_actions, done, training=True):
        raise NotImplementedError

    def finish_episode(self, training=True):
        return {}

    def _discounted_returns(self, rewards):
        returns = []
        running_total = 0.0
        for reward in reversed(rewards):
            running_total = reward + self.gamma * running_total
            returns.append(running_total)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)


class REINFORCEAgent(PolicyGradientAgent):
    def start_episode(self):
        self._log_probs = []
        self._rewards = []
        self._entropies = []

    def _record_action(self, state, distribution, action_tensor):
        self._log_probs.append(distribution.log_prob(action_tensor))
        self._entropies.append(distribution.entropy())

    def observe(self, reward, next_state, next_available_actions, done, training=True):
        self._rewards.append(reward)
        return None

    def finish_episode(self, training=True):
        if not training or not self._rewards:
            return {}

        returns = self._discounted_returns(self._rewards)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)
        policy_loss = -(log_probs * returns).sum() - self.entropy_coef * entropies.sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": None,
            "entropy": float(entropies.mean().item()),
        }


class ReinforceBaselineAgent(PolicyGradientAgent):
    def __init__(
        self,
        state_size,
        action_size,
        seed=None,
        hidden_sizes=(128, 64),
        gamma=0.99,
        learning_rate=0.001,
        value_learning_rate=0.001,
        entropy_coef=0.01,
        device=None,
    ):
        super().__init__(
            state_size,
            action_size,
            seed=seed,
            hidden_sizes=hidden_sizes,
            gamma=gamma,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
            device=device,
        )
        self.value_network = ValueNetwork(state_size, hidden_sizes=hidden_sizes).to(
            self.device
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=value_learning_rate,
        )

    def start_episode(self):
        self._log_probs = []
        self._rewards = []
        self._entropies = []
        self._values = []

    def _record_action(self, state, distribution, action_tensor):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._log_probs.append(distribution.log_prob(action_tensor))
        self._entropies.append(distribution.entropy())
        self._values.append(self.value_network(state_tensor).squeeze(0))

    def observe(self, reward, next_state, next_available_actions, done, training=True):
        self._rewards.append(reward)
        return None

    def finish_episode(self, training=True):
        if not training or not self._rewards:
            return {}

        returns = self._discounted_returns(self._rewards)
        values = torch.stack(self._values)
        advantages = returns - values.detach()
        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)

        policy_loss = -(log_probs * advantages).sum() - self.entropy_coef * entropies.sum()
        value_loss = functional.mse_loss(values, returns)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropies.mean().item()),
        }


class ActorCriticAgent(PolicyGradientAgent):
    def __init__(
        self,
        state_size,
        action_size,
        seed=None,
        hidden_sizes=(128, 64),
        gamma=0.99,
        learning_rate=0.001,
        value_coef=0.5,
        entropy_coef=0.01,
        device=None,
    ):
        super().__init__(
            state_size,
            action_size,
            seed=seed,
            hidden_sizes=hidden_sizes,
            gamma=gamma,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
            device=device,
        )
        self.value_network = ValueNetwork(state_size, hidden_sizes=hidden_sizes).to(
            self.device
        )
        self.value_coef = value_coef
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate,
        )

    def start_episode(self):
        self._last_log_prob = None
        self._last_entropy = None
        self._last_value = None

    def _record_action(self, state, distribution, action_tensor):
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._last_log_prob = distribution.log_prob(action_tensor)
        self._last_entropy = distribution.entropy()
        self._last_value = self.value_network(state_tensor).squeeze(0)

    def observe(self, reward, next_state, next_available_actions, done, training=True):
        if not training:
            return None

        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if done:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                next_state_tensor = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
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
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }

    def finish_episode(self, training=True):
        return {}


def clone_board(board):
    return [row[:] for row in board]


def run_policy_gradient_episode(env, agent, training, max_steps=None, capture=False):
    max_steps = max_steps or env.action_space_size() * 2
    state = env.reset()
    agent.start_episode()
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

    for _ in range(max_steps):
        available_actions = env.available_actions()
        action = agent.select_action(
            state,
            available_actions,
            explore=training,
            track=training,
        )
        next_state, reward, done, info = env.step(action)
        next_actions = [] if done else env.available_actions()
        metric = agent.observe(
            reward,
            next_state,
            next_actions,
            done,
            training=training,
        )
        if metric:
            metrics.append(metric)

        total_reward += reward
        if capture:
            row, col = divmod(action, env.size())
            if info.get("hit_mine", False):
                note = "The agent hit a mine."
            elif done:
                note = "The board is solved."
            else:
                note = f"Revealed {info.get('revealed', 0)} safe cell(s)."
            trajectory.append(
                {
                    "board": clone_board(env.render()),
                    "action": action,
                    "reward": reward,
                    "total_reward": total_reward,
                    "note": note,
                    "highlight": (row, col),
                }
            )

        state = next_state
        if done:
            won = not info.get("hit_mine", False)
            break

    final_metric = agent.finish_episode(training=training)
    if final_metric:
        metrics.append(final_metric)

    return {
        "reward": total_reward,
        "won": won,
        "metrics": metrics,
        "trajectory": trajectory,
        "board": clone_board(env.render()),
    }


def train_policy_gradient_agent(
    env,
    agent,
    episodes,
    report_every=25,
    max_steps=None,
):
    max_steps = max_steps or env.action_space_size() * 2
    episode_rewards = []
    rolling_win_rates = []
    policy_losses = []
    value_losses = []
    entropies = []
    recent_rewards = deque(maxlen=report_every)
    recent_wins = deque(maxlen=report_every)

    for episode in range(1, episodes + 1):
        outcome = run_policy_gradient_episode(
            env,
            agent,
            training=True,
            max_steps=max_steps,
            capture=False,
        )
        episode_rewards.append(outcome["reward"])
        recent_rewards.append(outcome["reward"])
        recent_wins.append(int(outcome["won"]))
        rolling_win_rates.append(sum(recent_wins) / len(recent_wins))

        for metric in outcome["metrics"]:
            if metric.get("policy_loss") is not None:
                policy_losses.append(metric["policy_loss"])
            if metric.get("value_loss") is not None:
                value_losses.append(metric["value_loss"])
            if metric.get("entropy") is not None:
                entropies.append(metric["entropy"])

        if episode % report_every == 0:
            print(
                f"Episode {episode:4d} | avg reward "
                f"{sum(recent_rewards) / len(recent_rewards):6.3f} | "
                f"win rate {sum(recent_wins) / len(recent_wins):5.1%}"
            )

    return {
        "episode_rewards": episode_rewards,
        "smoothed_rewards": _moving_average(episode_rewards, window=20),
        "rolling_win_rates": rolling_win_rates,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies,
    }


def evaluate_policy_gradient_agent(
    env,
    agent,
    episodes=30,
    max_steps=None,
    capture=False,
):
    max_steps = max_steps or env.action_space_size() * 2
    wins = 0
    rewards = []
    trajectory = None
    last_board = None

    for episode in range(episodes):
        outcome = run_policy_gradient_episode(
            env,
            agent,
            training=False,
            max_steps=max_steps,
            capture=capture and episode == episodes - 1,
        )
        wins += int(outcome["won"])
        rewards.append(outcome["reward"])
        last_board = outcome["board"]
        if capture and episode == episodes - 1:
            trajectory = outcome["trajectory"]

    return {
        "win_rate": wins / episodes if episodes else 0.0,
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "last_board": last_board,
        "trajectory": trajectory,
    }


def _moving_average(values, window=20):
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averaged.append(sum(values[start : index + 1]) / (index - start + 1))
    return averaged
