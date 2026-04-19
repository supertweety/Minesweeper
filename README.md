# Minesweeper Tournament Framework

This repository now includes a small tournament package for turn-based Minesweeper agents.
It is aimed at agents that reason about the board, for example with SAT solving or other symbolic methods.

The package lives in [`minesweeper_tournament`](./minesweeper_tournament).

## What the Tournament Does

Agents play on the same hidden Minesweeper board, one after another, one move at a time.

For each round:

1. A new random board is generated.
2. Agents are ordered randomly by default.
3. They take turns selecting one hidden tile.
4. If an agent clicks a mine, that agent is removed from the current round and every other active agent gets 1 point.
5. If an agent times out, that agent is removed from the current round, but is allowed to return in the next round.
6. If an agent crashes, that agent is removed from the current round and returns in the next round as a fresh instance.
7. When all non-mine tiles are uncovered, the round ends.

The tournament runs for a predefined number of rounds.

## Package Overview

- [`minesweeper_tournament/api.py`](./minesweeper_tournament/api.py): public data structures and the base agent class
- [`minesweeper_tournament/engine.py`](./minesweeper_tournament/engine.py): tournament runner and process isolation
- [`minesweeper_tournament/dummy_agents.py`](./minesweeper_tournament/dummy_agents.py): sample agents for testing
- [`mine_board.py`](./mine_board.py): existing board logic reused by the tournament engine

## Agent API

Subclass `MinesweeperAgent` and implement `choose_action`.

```python
from minesweeper_tournament import Action, MinesweeperAgent, Observation, RoundContext


class MyAgent(MinesweeperAgent):
    def new_round(self, context: RoundContext) -> None:
        self._something = {}

    def choose_action(self, observation: Observation) -> Action:
        return observation.legal_actions[0]
```

### Lifecycle

- `new_round(context)`: called once at the start of each round
- `choose_action(observation) -> Action`: called once on each turn when the agent is active
- `round_finished(outcome)`: called at the end of a round for agents that are still alive in that round

### Observation Fields

`Observation` contains:

- `round_index`: current round number, starting at 1
- `turn_index`: current turn number inside the round
- `board`: visible board as a tuple of tuples
- `legal_actions`: all currently legal hidden tiles as `Action(row, col)`
- `scores`: current tournament score table
- `active_agents`: agents still in the current round
- `disqualified_agents`: agents removed from the whole tournament, if a custom runner chooses to use permanent disqualification

The visible board uses the same symbols as the existing board logic:

- `?`: hidden tile
- `' '` (space): uncovered empty tile
- `'1'` to `'8'`: uncovered clue numbers

## Rules and Edge Cases

### Clicking a Mine

If an agent clicks on a mine:

- that agent is removed from the current round
- each other active agent gets 1 point

### Thinking Too Long

Each agent call is protected by a timeout.

If an agent exceeds the configured time limit:

- the agent is removed from the current round
- the agent is allowed back into the next round

### Crashes

Agents run in separate Python processes.

If an agent raises an exception or its process fails:

- it is removed from the current round
- it is started again as a fresh instance in the next round

### Invalid Moves

If an agent returns:

- something other than `Action`
- coordinates outside the board
- an already revealed tile

then the move is treated as invalid and the agent is removed from the current round.

Invalid moves do not award points to other agents.

## Dummy Agents

The package includes a few ready-made test agents:

- `RandomAgent`: chooses uniformly from legal moves
- `RowMajorAgent`: always picks the first legal tile in row-major order
- `LocalInferenceAgent`: uses a small set of local clue-based rules to avoid obviously dangerous moves and uncover obviously safe ones
- `SlowAgent`: sleeps on purpose to test timeout handling
- `CrashAgent`: raises an exception on purpose to test crash handling

## How `LocalInferenceAgent` Works

`LocalInferenceAgent` is meant to be a simple baseline that you can read in one sitting and then improve on.

It uses two familiar Minesweeper rules:

1. If a clue already has all of its mines accounted for, every other hidden neighbor is safe.
2. If a clue still needs exactly as many mines as it has hidden neighbors, all of those hidden neighbors must be mines.

That is the whole idea. The agent keeps a small internal set of positions that it currently believes are mines, repeatedly scans the visible clues, and extracts any safe moves that follow immediately from those two rules.

This is the core pattern:

```python
if clue == known_mine_neighbors:
    for position in hidden_neighbors:
        safe_moves.add(position)

remaining_mines = clue - known_mine_neighbors
if remaining_mines == len(hidden_neighbors):
    for position in hidden_neighbors:
        self._known_mines.add(position)
```

The full agent applies that logic in a loop until no new information appears. That matters because one clue can reveal a mine, which then makes a neighboring clue expose a safe tile, which can then unlock another clue.

Its action selection is intentionally small and easy to debug:

```python
safe_moves = self._infer_safe_moves(board, legal_positions)

if safe_moves:
    row, col = sorted(safe_moves)[0]
    return Action(row=row, col=col)

fallback = [
    action
    for action in observation.legal_actions
    if (action.row, action.col) not in self._known_mines
]
```

If no safe move can be proved, the agent falls back to a random legal action that is not currently marked as a likely mine, if such a move exists.

This makes `LocalInferenceAgent` useful as a reference point:

- it is stronger than a purely random baseline
- its behavior is still transparent enough to inspect by hand
- it shows how much mileage you can get from very small pieces of logical inference

It is also deliberately limited. The agent only uses local deductions around visible clues, so it will miss situations where several clues must be combined at once.

## Where SAT Solving Fits

SAT-based agents push exactly that next step: instead of reasoning about one clue at a time, they combine all currently visible constraints into one global logical model.

A standard way to think about it is:

- every hidden tile gets a Boolean variable
- `True` means "this tile contains a mine"
- every visible clue creates a counting constraint over its hidden neighbors

For example, if a visible `2` touches three still-hidden tiles and one tile you already know is a mine, then those three hidden variables must contain exactly one more mine in total.

In pseudocode, the SAT view looks something like this:

```python
variables = {(r, c): new_bool_var() for (r, c) in hidden_tiles}

for clue_cell in visible_clues:
    neighbors = hidden_neighbors_of(clue_cell)
    required = clue_value(clue_cell) - known_mines_around(clue_cell)
    add_exactly_k_constraint([variables[p] for p in neighbors], required)
```

Once those constraints are in the solver, you can ask stronger questions than the local baseline can answer:

- Is a tile definitely safe?
- Is a tile definitely a mine?
- Are there multiple consistent worlds left, which means a guess is unavoidable?

One common pattern is:

1. Assume a tile is a mine and ask whether the constraints are still satisfiable.
2. Assume the same tile is safe and ask again.
3. If one assumption is impossible, the other one is forced.

That gives a direct bridge between the local baseline and a symbolic agent:

- `LocalInferenceAgent` captures the easiest deductions people often make by eye
- a SAT-based agent captures those same deductions and also combines information across the whole frontier of visible clues

For tournament work, a good progression is often:

1. Start from `LocalInferenceAgent` so you have a clean, readable baseline.
2. Replace its handcrafted rules with a constraint model over the current hidden frontier.
3. Use the tournament logs to compare how often the SAT-based agent finds safe moves that the local baseline misses.

## Minimal Tournament Example

Use top-level callables or classes as factories. Avoid lambdas because the tournament uses `multiprocessing`.

```python
from functools import partial

from minesweeper_tournament import (
    LocalInferenceAgent,
    RandomAgent,
    RowMajorAgent,
    TournamentConfig,
    TournamentRunner,
)


runner = TournamentRunner(
    agent_factories={
        "local": partial(LocalInferenceAgent, seed=7, name="local"),
        "random": partial(RandomAgent, seed=1, name="random"),
        "row_major": partial(RowMajorAgent, name="row_major"),
    },
    config=TournamentConfig(
        board_size=5,
        num_mines=5,
        num_rounds=20,
        turn_timeout_seconds=0.5,
        random_seed=123,
    ),
)

result = runner.run()
print(result.scores)
```

## Inspecting Results

Each tournament returns a `TournamentResult`.

Useful fields:

- `result.scores`: final score table
- `result.rounds`: per-round summaries
- `result.disqualified_agents`: agents removed from the tournament, if a custom runner uses permanent disqualification

Each round contains detailed turn logs:

```python
for turn in result.rounds[0].turn_records:
    print(turn.turn_index, turn.agent_name, turn.outcome, turn.message)
```

## Debugging Agents

The easiest way to debug your agent is:

1. Start with a tiny board, for example `4x4` with `2` mines.
2. Run it against `RowMajorAgent`, `RandomAgent`, and `LocalInferenceAgent`.
3. Look at `result.rounds[*].turn_records` to see exactly when it timed out, crashed, or made an invalid move.
4. Only then scale up to larger boards.

If you want to debug the reasoning logic directly, you can also instantiate your agent in a normal Python session and call `choose_action` manually with a handcrafted `Observation`.

## Notes

- Return an `Action(row, col)`.
- Only choose from `observation.legal_actions`.
- Keep your agent class importable at module top level.
- Do not rely on global mutable state.
- Assume the engine may recreate your agent between rounds.

That last point is important: a timeout or crash removes the agent from the current round, and the engine starts a fresh agent instance in the next round.
