import itertools
import random


class MineBoard:
    MINE = "@"
    UNCOVERED = " "
    HIDDEN = "?"

    def __init__(self, size=5, num_mines=10, rng=None):
        if size <= 0:
            raise ValueError("size must be positive")
        if num_mines < 0 or num_mines >= size * size:
            raise ValueError("num_mines must be between 0 and size * size - 1")
        self._board = [[self.HIDDEN] * size for i in range(size)]
        self._size = size
        self._num_mines = num_mines
        self._rng = rng or random
        mine_positions = self._rng.sample(
            list(itertools.product(range(size), range(size))), num_mines
        )
        for i, j in mine_positions:
            self._board[i][j] = self.MINE
        self._neighbouring_mines = [[0] * size for i in range(size)]
        for i in range(size):
            for j in range(size):
                self._neighbouring_mines[i][j] = self._count_neighbouring_mines(i, j)

    def print(self):
        for row in self.board():
            print(row)

    def board(self):
        retval = []
        for i in range(len(self._board)):
            new_row = []
            for j in range(len(self._board[i])):
                if self._board[i][j] == self.MINE or self._board[i][j] == self.HIDDEN:
                    new_row.append(self.HIDDEN)
                else:
                    if self._neighbouring_mines[i][j] == 0:
                        new_row.append(self.UNCOVERED)
                    else:
                        new_row.append(str(self._neighbouring_mines[i][j]))
            retval.append(new_row)
        return retval

    def size(self):
        return self._size

    def num_mines(self):
        return self._num_mines

    def safe_cells(self):
        return self._size * self._size - self._num_mines

    def revealed_safe_cells(self):
        count = 0
        for i in range(len(self._board)):
            for j in range(len(self._board[i])):
                if self._board[i][j] == self.UNCOVERED:
                    count += 1
        return count

    def has_mine(self, i, j):
        return self._board[i][j] == self.MINE

    def is_hidden(self, i, j):
        return self._board[i][j] == self.HIDDEN or self._board[i][j] == self.MINE

    def visible_value(self, i, j):
        if self.is_hidden(i, j):
            return self.HIDDEN
        if self._neighbouring_mines[i][j] == 0:
            return self.UNCOVERED
        return str(self._neighbouring_mines[i][j])

    def perform_action(self, i, j):
        if self._board[i][j] == self.MINE:
            return False
        if self._board[i][j] == self.HIDDEN:
            self._board[i][j] = self.UNCOVERED
            if self._neighbouring_mines[i][j] == 0:
                for neighbour_i, neighbour_j in self._neighbours(i, j):
                    self.perform_action(neighbour_i, neighbour_j)
        return True

    def _count_neighbouring_mines(self, i, j):
        count = 0
        for neighbour_i, neighbour_j in self._neighbours(i, j):
            if self._board[neighbour_i][neighbour_j] == self.MINE:
                count += 1
        return count

    def _neighbours(self, i, j):
        retval = []
        for neighbour_i in range(max(0, i - 1), min(len(self._board), i + 2)):
            for neighbour_j in range(
                max(0, j - 1), min(len(self._board[i]), j + 2)
            ):
                if neighbour_i != i or neighbour_j != j:
                    retval.append((neighbour_i, neighbour_j))
        return retval

    def is_solved(self):
        for i in range(len(self._board)):
            for j in range(len(self._board[i])):
                if self._board[i][j] == self.HIDDEN:
                    return False
        return True
