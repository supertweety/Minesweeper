import argparse

from dqn_agent import DQNAgent
from minesweeper_env import MinesweeperEnv


def format_board(rows):
    return "\n".join(" ".join(cell if cell != " " else "." for cell in row) for row in rows)


def run_episode(env, agent, training, max_steps):
    state = env.reset()
    total_reward = 0.0
    won = False
    losses = []

    for _ in range(max_steps):
        available_actions = env.available_actions()
        action = agent.select_action(state, available_actions, explore=training)
        next_state, reward, done, info = env.step(action)
        next_actions = [] if done else env.available_actions()

        if training:
            agent.remember(state, action, reward, next_state, done, next_actions)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        total_reward += reward
        state = next_state

        if done:
            won = not info.get("hit_mine", False)
            break

    average_loss = sum(losses) / len(losses) if losses else None
    return total_reward, won, average_loss, env.render()


def evaluate(env, agent, episodes, max_steps):
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    wins = 0
    rewards = []
    last_board = None

    for _ in range(episodes):
        reward, won, _, board = run_episode(env, agent, training=False, max_steps=max_steps)
        rewards.append(reward)
        wins += int(won)
        last_board = board

    agent.epsilon = original_epsilon
    return {
        "win_rate": wins / episodes if episodes else 0.0,
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "last_board": last_board,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Minesweeper.")
    parser.add_argument("--size", type=int, default=5, help="Board width and height.")
    parser.add_argument(
        "--num-mines",
        type=int,
        default=5,
        help="Number of mines on the board.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Training episodes.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Evaluation episodes after training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum moves per episode. Defaults to twice the number of cells.",
    )
    parser.add_argument(
        "--safe-first-move",
        action="store_true",
        help="Regenerate the board if the first chosen square is a mine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the environment and agent.",
    )
    parser.add_argument(
        "--model-path",
        default="dqn_minesweeper_model.pt",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="How often to print training progress.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Replay batch size used for gradient updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay applied after each training step.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cpu, cuda, or mps.",
    )
    args = parser.parse_args()

    env = MinesweeperEnv(
        size=args.size,
        num_mines=args.num_mines,
        safe_first_move=args.safe_first_move,
        seed=args.seed,
    )
    initial_state = env.reset(seed=args.seed)
    agent = DQNAgent(
        state_size=len(initial_state),
        action_size=env.action_space_size(),
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        device=args.device,
    )
    max_steps = args.max_steps or env.action_space_size() * 2

    print(f"Training on device: {agent.device}")

    recent_rewards = []
    recent_wins = []

    for episode in range(1, args.episodes + 1):
        reward, won, loss, _ = run_episode(env, agent, training=True, max_steps=max_steps)
        recent_rewards.append(reward)
        recent_wins.append(int(won))
        if len(recent_rewards) > args.report_every:
            recent_rewards.pop(0)
            recent_wins.pop(0)

        if episode % args.report_every == 0:
            average_reward = sum(recent_rewards) / len(recent_rewards)
            win_rate = sum(recent_wins) / len(recent_wins)
            loss_text = f"{loss:.4f}" if loss is not None else "warming up"
            print(
                f"Episode {episode:4d} | avg reward {average_reward:6.3f} "
                f"| win rate {win_rate:5.1%} | epsilon {agent.epsilon:5.3f} "
                f"| loss {loss_text}"
            )

    evaluation = evaluate(env, agent, args.eval_episodes, max_steps=max_steps)
    agent.save(args.model_path)

    print()
    print("Evaluation")
    print(f"  Win rate:        {evaluation['win_rate']:.1%}")
    print(f"  Average reward:  {evaluation['average_reward']:.3f}")
    print(f"  Model saved to:  {args.model_path}")
    print("  Final board from last evaluation episode:")
    print(format_board(evaluation["last_board"]))


if __name__ == "__main__":
    main()
