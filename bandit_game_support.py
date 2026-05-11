"""Utilities for the bandit game notebook.

This module keeps the notebook short and focused on playing the game.
The hidden reward distributions stay inside this file so that the game
does not accidentally reveal too much information.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

REWARD_VALUES = list(range(11))


def _plt():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _display_tools():
    try:
        from IPython.display import Markdown, display

        return Markdown, display
    except Exception:
        def _markdown(text):
            return text

        def _display(text):
            print(text)

        return _markdown, _display


def _sample_dirichlet(rng, alpha):
    draws = [rng.gammavariate(max(a, 1e-9), 1.0) for a in alpha]
    total = sum(draws)
    return [value / total for value in draws]


def _expected_reward(probabilities):
    return sum(reward * prob for reward, prob in zip(REWARD_VALUES, probabilities))


def _reward_variance(probabilities):
    mean = _expected_reward(probabilities)
    return sum(prob * (reward - mean) ** 2 for reward, prob in zip(REWARD_VALUES, probabilities))


def _categorical_sample(rng, probabilities):
    threshold = rng.random()
    running = 0.0
    for reward, prob in zip(REWARD_VALUES, probabilities):
        running += prob
        if threshold <= running:
            return reward
    return REWARD_VALUES[-1]


class DiscreteRewardBandit:
    """A multi-armed bandit with hidden rewards in {0, 1, ..., 10}."""

    def __init__(self, n_arms=5, seed=None):
        self.n_arms = n_arms
        self._rng = random.Random(seed)
        self._distributions = self._generate_interesting_distributions()
        self._means = [_expected_reward(distribution) for distribution in self._distributions]

    def pull(self, arm, rng=None):
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("arm is out of range")
        return _categorical_sample(rng or self._rng, self._distributions[arm])

    def _generate_interesting_distributions(self):
        while True:
            distributions = [self._sample_arm_distribution() for _ in range(self.n_arms)]
            means = [_expected_reward(distribution) for distribution in distributions]
            variances = [_reward_variance(distribution) for distribution in distributions]
            ordered_means = sorted(means, reverse=True)

            gap = ordered_means[0] - ordered_means[1]
            spread = max(means) - min(means)
            variance_spread = max(variances) - min(variances)

            if spread < 1.25:
                continue
            if not (0.25 <= gap <= 2.5):
                continue
            if variance_spread < 2.0:
                continue
            return distributions

    def _sample_arm_distribution(self):
        center = self._rng.uniform(1.0, 9.0)
        spread = self._rng.choice([0.8, 1.2, 1.8, 2.4])
        style = self._rng.choice(["smooth", "spiky", "risky"])

        weights = []
        for reward in REWARD_VALUES:
            base = math.exp(-((reward - center) ** 2) / (2.0 * spread * spread))
            if style == "spiky":
                if reward in {round(center), max(0, round(center) - 1), min(10, round(center) + 1)}:
                    base *= 2.2
            elif style == "risky":
                distance = abs(reward - center)
                base *= 0.6
                if reward in {0, 10}:
                    base += 0.7 + 0.2 * distance
            weights.append(base + 0.05)

        concentration = self._rng.choice([3.0, 6.0, 10.0])
        alpha = [concentration * weight for weight in weights]
        return _sample_dirichlet(self._rng, alpha)


class UCBPlayer:
    """UCB1 for bounded rewards.

    The arm score is the empirical mean plus the standard UCB1 bonus
    sqrt(2 log t / N_t(a)).
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.reward_sums = [0.0] * n_arms
        self.total_reward = 0.0
        self.total_pulls = 0

    def preview(self):
        means = [
            self.reward_sums[arm] / self.counts[arm] if self.counts[arm] else 0.0
            for arm in range(self.n_arms)
        ]
        bonuses = []
        scores = []
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                bonuses.append(float("inf"))
                scores.append(float("inf"))
            else:
                bonus = math.sqrt((2.0 * math.log(self.total_pulls + 1.0)) / self.counts[arm])
                bonuses.append(bonus)
                scores.append(means[arm] + bonus)

        untried = [arm for arm, count in enumerate(self.counts) if count == 0]
        if untried:
            chosen_arm = untried[0]
        else:
            chosen_arm = max(range(self.n_arms), key=lambda arm: scores[arm])

        finite_bonuses = [0.0 if not math.isfinite(bonus) else bonus for bonus in bonuses]
        finite_scores = [means[arm] + finite_bonuses[arm] for arm in range(self.n_arms)]

        return {
            "chosen_arm": chosen_arm,
            "means": means,
            "bonuses": finite_bonuses,
            "scores": finite_scores,
            "counts": list(self.counts),
        }

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.total_reward += reward
        self.total_pulls += 1


class ThompsonSamplingPlayer:
    def __init__(self, n_arms, seed=0):
        self.n_arms = n_arms
        self.posterior = [[1.0] * len(REWARD_VALUES) for _ in range(n_arms)]
        self.total_reward = 0.0
        self.total_pulls = 0
        self._rng = random.Random(seed)

    def preview(self):
        sampled_means = []
        posterior_means = []
        evidence = []

        for arm in range(self.n_arms):
            probabilities = _sample_dirichlet(self._rng, self.posterior[arm])
            sampled_means.append(_expected_reward(probabilities))

            total_alpha = sum(self.posterior[arm])
            mean_probabilities = [alpha / total_alpha for alpha in self.posterior[arm]]
            posterior_means.append(_expected_reward(mean_probabilities))
            evidence.append(total_alpha - len(REWARD_VALUES))

        chosen_arm = max(range(self.n_arms), key=lambda arm: sampled_means[arm])

        return {
            "chosen_arm": chosen_arm,
            "sampled_means": sampled_means,
            "posterior_means": posterior_means,
            "evidence": evidence,
        }

    def update(self, arm, reward):
        self.posterior[arm][reward] += 1.0
        self.total_reward += reward
        self.total_pulls += 1


@dataclass
class RoundResult:
    round_index: int
    user_arm: int
    user_reward: int
    ucb_arm: int
    ucb_reward: int
    thompson_arm: int
    thompson_reward: int


class BanditGame:
    """A three-player game: you, UCB, and Thompson sampling."""

    def __init__(self, n_arms=5, seed=None, user_name="You"):
        self.n_arms = n_arms
        self.user_name = user_name
        self._rng = random.Random(seed)
        bandit_seed = self._rng.randrange(10**9)
        thompson_seed = self._rng.randrange(10**9)

        self.bandit = DiscreteRewardBandit(n_arms=n_arms, seed=bandit_seed)
        self.ucb = UCBPlayer(n_arms=n_arms)
        self.thompson = ThompsonSamplingPlayer(n_arms=n_arms, seed=thompson_seed)

        self.round_index = 0
        self.user_total_reward = 0.0
        self.round_history = []
        self.cumulative_rewards = {
            self.user_name: [0.0],
            "UCB": [0.0],
            "Thompson": [0.0],
        }
        self._cached_preview = None

    def preview(self):
        if self._cached_preview is None:
            self._cached_preview = {
                "round_index": self.round_index + 1,
                "ucb": self.ucb.preview(),
                "thompson": self.thompson.preview(),
            }
        return self._cached_preview

    def play_round(self, user_choice):
        preview = self.preview()
        user_arm = self._normalize_user_choice(user_choice, preview)
        ucb_arm = preview["ucb"]["chosen_arm"]
        thompson_arm = preview["thompson"]["chosen_arm"]

        user_reward = self.bandit.pull(user_arm, self._rng)
        ucb_reward = self.bandit.pull(ucb_arm, self._rng)
        thompson_reward = self.bandit.pull(thompson_arm, self._rng)

        self.user_total_reward += user_reward
        self.ucb.update(ucb_arm, ucb_reward)
        self.thompson.update(thompson_arm, thompson_reward)

        self.round_index += 1
        self.cumulative_rewards[self.user_name].append(self.user_total_reward)
        self.cumulative_rewards["UCB"].append(self.ucb.total_reward)
        self.cumulative_rewards["Thompson"].append(self.thompson.total_reward)

        result = RoundResult(
            round_index=self.round_index,
            user_arm=user_arm,
            user_reward=user_reward,
            ucb_arm=ucb_arm,
            ucb_reward=ucb_reward,
            thompson_arm=thompson_arm,
            thompson_reward=thompson_reward,
        )
        self.round_history.append(result)
        self._cached_preview = None
        return result

    def _normalize_user_choice(self, user_choice, preview):
        if isinstance(user_choice, str):
            normalized = user_choice.strip().lower()
            if normalized == "ucb":
                return preview["ucb"]["chosen_arm"]
            if normalized in {"thompson", "ts"}:
                return preview["thompson"]["chosen_arm"]
            raise ValueError("string choices must be 'ucb' or 'thompson'")

        arm = int(user_choice)
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"arm must be between 0 and {self.n_arms - 1}")
        return arm


def _score_table_lines(game):
    latest = game.round_history[-1] if game.round_history else None
    lines = [
        f"### Round {game.preview()['round_index']}",
        "",
        "| player | cumulative reward | latest arm | latest reward |",
        "|---|---:|---:|---:|",
    ]

    latest_values = {
        game.user_name: ("-", "-"),
        "UCB": ("-", "-"),
        "Thompson": ("-", "-"),
    }
    if latest is not None:
        latest_values = {
            game.user_name: (latest.user_arm, latest.user_reward),
            "UCB": (latest.ucb_arm, latest.ucb_reward),
            "Thompson": (latest.thompson_arm, latest.thompson_reward),
        }

    for name in [game.user_name, "UCB", "Thompson"]:
        arm, reward = latest_values[name]
        lines.append(
            f"| {name} | {game.cumulative_rewards[name][-1]:.0f} | {arm} | {reward} |"
        )
    return lines


def show_dashboard(game):
    Markdown, display = _display_tools()
    preview = game.preview()
    lines = _score_table_lines(game)
    lines.extend(
        [
            "",
            f"- UCB would choose arm **{preview['ucb']['chosen_arm']}**.",
            "- UCB bonus: `sqrt(2 log t / N_t(a))`",
            f"- Thompson sampling would choose arm **{preview['thompson']['chosen_arm']}**.",
            f"- Your options are arms `0` to `{game.n_arms - 1}`, or the shortcuts `'ucb'` and `'thompson'`.",
        ]
    )
    display(Markdown("\n".join(lines)))
    plot_cumulative_rewards(game)
    plot_ucb_state(game, preview)
    plot_thompson_state(game, preview)


def show_round_result(game, result):
    Markdown, display = _display_tools()
    lines = [
        f"### Results for round {result.round_index}",
        "",
        f"- {game.user_name} chose arm **{result.user_arm}** and got **{result.user_reward}**.",
        f"- UCB chose arm **{result.ucb_arm}** and got **{result.ucb_reward}**.",
        f"- Thompson sampling chose arm **{result.thompson_arm}** and got **{result.thompson_reward}**.",
    ]
    display(Markdown("\n".join(lines)))


def play_round_and_show(game, user_choice):
    result = game.play_round(user_choice)
    show_round_result(game, result)
    show_dashboard(game)
    return result


def prompt_and_play_round(game):
    preview = game.preview()
    user_choice = input(
        f"Round {preview['round_index']}: choose an arm 0-{game.n_arms - 1}, "
        "or type 'ucb' / 'thompson': "
    ).strip()
    if user_choice.lower() not in {"ucb", "thompson", "ts"}:
        user_choice = int(user_choice)
    return play_round_and_show(game, user_choice)


def interactive_game_loop(game, max_rounds=None):
    """Keep prompting until the player decides to stop.

    Enter:
    - an arm number such as 0, 1, 2, ...
    - 'ucb'
    - 'thompson'
    - 'q' to stop
    """
    rounds_played = 0
    while max_rounds is None or rounds_played < max_rounds:
        preview = game.preview()
        user_choice = input(
            f"Round {preview['round_index']}: choose an arm 0-{game.n_arms - 1}, "
            "type 'ucb' / 'thompson', or 'q' to stop: "
        ).strip()
        lowered = user_choice.lower()
        if lowered in {"q", "quit", "exit", ""}:
            print("Stopped the game loop.")
            break
        if lowered not in {"ucb", "thompson", "ts"}:
            user_choice = int(user_choice)
        play_round_and_show(game, user_choice)
        rounds_played += 1


def plot_cumulative_rewards(game):
    plt = _plt()
    if plt is None:
        print("matplotlib is not installed, so plots are skipped.")
        return
    plt.figure(figsize=(8, 4.5))
    for name, rewards in game.cumulative_rewards.items():
        plt.plot(range(len(rewards)), rewards, marker="o", label=name)
    plt.xlabel("Round")
    plt.ylabel("Cumulative reward")
    plt.title("Scoreboard")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_ucb_state(game, preview=None):
    plt = _plt()
    if plt is None:
        print("matplotlib is not installed, so plots are skipped.")
        return
    preview = preview or game.preview()
    state = preview["ucb"]
    arms = list(range(game.n_arms))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(arms, state["means"], color="#4e79a7")
    axes[0].set_title("UCB empirical means")
    axes[0].set_xlabel("Arm")
    axes[0].set_ylabel("Average observed reward")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(arms, state["bonuses"], color="#f28e2b", alpha=0.8, label="bonus")
    axes[1].plot(arms, state["scores"], marker="o", color="#e15759", label="ucb score")
    axes[1].axvline(state["chosen_arm"], color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("UCB bonuses and scores")
    axes[1].set_xlabel("Arm")
    axes[1].set_ylabel("Score")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_thompson_state(game, preview=None):
    plt = _plt()
    if plt is None:
        print("matplotlib is not installed, so plots are skipped.")
        return
    preview = preview or game.preview()
    state = preview["thompson"]
    arms = list(range(game.n_arms))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(arms, state["posterior_means"], color="#59a14f", alpha=0.8, label="posterior mean")
    axes[0].plot(arms, state["sampled_means"], marker="o", color="#af7aa1", label="sampled mean")
    axes[0].axvline(state["chosen_arm"], color="black", linestyle="--", alpha=0.5)
    axes[0].set_title("Thompson expected rewards")
    axes[0].set_xlabel("Arm")
    axes[0].set_ylabel("Expected reward")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(arms, state["evidence"], color="#76b7b2")
    axes[1].set_title("Thompson evidence by arm")
    axes[1].set_xlabel("Arm")
    axes[1].set_ylabel("Observed pulls")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


__all__ = [
    "BanditGame",
    "interactive_game_loop",
    "play_round_and_show",
    "prompt_and_play_round",
    "show_dashboard",
]
