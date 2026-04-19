import random

from mine_board import MineBoard


class MinesweeperEnv:
    HIDDEN_CHANNEL = 0
    EMPTY_CHANNEL = 1
    NUMBER_OFFSET = 2
    CHANNELS_PER_CELL = 10

    def __init__(
        self,
        size=5,
        num_mines=10,
        reveal_reward=0.1,
        win_reward=1.0,
        lose_reward=-1.0,
        invalid_reward=-0.25,
        safe_first_move=False,
        seed=None,
    ):
        self._size = size
        self._num_mines = num_mines
        self._reveal_reward = reveal_reward
        self._win_reward = win_reward
        self._lose_reward = lose_reward
        self._invalid_reward = invalid_reward
        self._safe_first_move = safe_first_move
        self._rng = random.Random(seed)
        self._board = None
        self._moves_taken = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        self._board = MineBoard(self._size, self._num_mines, rng=self._rng)
        self._moves_taken = 0
        return self.state()

    def size(self):
        return self._size

    def action_space_size(self):
        return self._size * self._size

    def available_actions(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before available_actions()")
        actions = []
        for i in range(self._size):
            for j in range(self._size):
                if self._board.is_hidden(i, j):
                    actions.append(i * self._size + j)
        return actions

    def state(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before state()")
        encoded = []
        for i in range(self._size):
            for j in range(self._size):
                channels = [0.0] * self.CHANNELS_PER_CELL
                visible_value = self._board.visible_value(i, j)
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

        row, col = divmod(action, self._size)
        if self._safe_first_move and self._moves_taken == 0 and self._board.has_mine(row, col):
            while self._board.has_mine(row, col):
                self._board = MineBoard(self._size, self._num_mines, rng=self._rng)

        if not self._board.is_hidden(row, col):
            return self.state(), self._invalid_reward, False, {
                "invalid_action": True,
                "row": row,
                "col": col,
                "revealed": 0,
            }

        revealed_before = self._board.revealed_safe_cells()
        survived = self._board.perform_action(row, col)
        self._moves_taken += 1

        if not survived:
            return self.state(), self._lose_reward, True, {
                "hit_mine": True,
                "row": row,
                "col": col,
                "revealed": 0,
            }

        revealed = self._board.revealed_safe_cells() - revealed_before
        reward = revealed * self._reveal_reward
        done = self._board.is_solved()
        if done:
            reward += self._win_reward

        return self.state(), reward, done, {
            "hit_mine": False,
            "invalid_action": False,
            "row": row,
            "col": col,
            "revealed": revealed,
        }

    def render(self):
        if self._board is None:
            raise RuntimeError("reset() must be called before render()")
        return self._board.board()
