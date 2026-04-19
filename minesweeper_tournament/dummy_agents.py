from __future__ import annotations

import random
import time

from .api import Action, MinesweeperAgent, Observation


class RandomAgent(MinesweeperAgent):
    def __init__(self, seed: int | None = None, name: str | None = None):
        super().__init__(name=name)
        self._rng = random.Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        return self._rng.choice(list(observation.legal_actions))


class RowMajorAgent(MinesweeperAgent):
    def choose_action(self, observation: Observation) -> Action:
        return observation.legal_actions[0]


class LocalInferenceAgent(MinesweeperAgent):
    """
    A small rule-based baseline that uses two local Minesweeper ideas:

    1. If a clue equals the number of currently hidden neighbors, all of those
       neighbors must be mines.
    2. If a clue already has that many known mine neighbors, all remaining
       hidden neighbors must be safe.

    When no safe move can be inferred, the agent falls back to a random legal
    move that is not currently marked as a mine guess, if possible.
    """

    def __init__(self, seed: int | None = None, name: str | None = None):
        super().__init__(name=name)
        self._rng = random.Random(seed)
        self._known_mines: set[tuple[int, int]] = set()

    def new_round(self, context) -> None:
        self._known_mines = set()

    def choose_action(self, observation: Observation) -> Action:
        board = observation.board
        legal_positions = {(action.row, action.col) for action in observation.legal_actions}
        safe_moves = self._infer_safe_moves(board, legal_positions)

        if safe_moves:
            row, col = sorted(safe_moves)[0]
            return Action(row=row, col=col)

        fallback = [
            action
            for action in observation.legal_actions
            if (action.row, action.col) not in self._known_mines
        ]
        if fallback:
            return self._rng.choice(fallback)
        return self._rng.choice(list(observation.legal_actions))

    def _infer_safe_moves(
        self,
        board: tuple[tuple[str, ...], ...],
        legal_positions: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        safe_moves: set[tuple[int, int]] = set()
        changed = True
        while changed:
            changed = False
            for row in range(len(board)):
                for col in range(len(board[row])):
                    cell = board[row][col]
                    if not cell.isdigit():
                        continue

                    clue = int(cell)
                    hidden_neighbors = []
                    known_mine_neighbors = 0

                    for neighbor_row, neighbor_col in self._neighbors(board, row, col):
                        if (neighbor_row, neighbor_col) in self._known_mines:
                            known_mine_neighbors += 1
                        elif (neighbor_row, neighbor_col) in legal_positions:
                            hidden_neighbors.append((neighbor_row, neighbor_col))

                    if not hidden_neighbors:
                        continue

                    if clue == known_mine_neighbors:
                        for position in hidden_neighbors:
                            safe_moves.add(position)

                    remaining_mines = clue - known_mine_neighbors
                    if remaining_mines == len(hidden_neighbors):
                        for position in hidden_neighbors:
                            if position not in self._known_mines:
                                self._known_mines.add(position)
                                changed = True
                                if position in safe_moves:
                                    safe_moves.remove(position)

        return {
            position for position in safe_moves if position in legal_positions and position not in self._known_mines
        }

    def _neighbors(
        self,
        board: tuple[tuple[str, ...], ...],
        row: int,
        col: int,
    ) -> list[tuple[int, int]]:
        neighbors = []
        for neighbor_row in range(max(0, row - 1), min(len(board), row + 2)):
            for neighbor_col in range(max(0, col - 1), min(len(board[row]), col + 2)):
                if neighbor_row != row or neighbor_col != col:
                    neighbors.append((neighbor_row, neighbor_col))
        return neighbors


class SlowAgent(MinesweeperAgent):
    def __init__(self, sleep_seconds: float = 2.0, name: str | None = None):
        super().__init__(name=name)
        self._sleep_seconds = sleep_seconds

    def choose_action(self, observation: Observation) -> Action:
        time.sleep(self._sleep_seconds)
        return observation.legal_actions[0]


class CrashAgent(MinesweeperAgent):
    def choose_action(self, observation: Observation) -> Action:
        raise RuntimeError("intentional crash for tournament testing")
