from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


Position = Tuple[int, int]


class QuoridorBoard:
    """Implements the movement and wall placement rules for Quoridor."""

    def __init__(self, size: int = 9, walls_per_player: int = 10):
        if size < 3:
            raise ValueError("Quoridor requires a board of at least 3x3 squares")
        if walls_per_player < 0:
            raise ValueError("Each player must start with a non-negative wall count")

        self.size = size
        self.walls_per_player = walls_per_player

        middle = size // 2
        self.pawns: List[Position] = [(0, middle), (size - 1, middle)]
        self.horizontal_walls = np.zeros((size - 1, size - 1), dtype=np.int8)
        self.vertical_walls = np.zeros((size - 1, size - 1), dtype=np.int8)
        self.remaining_walls = [walls_per_player, walls_per_player]

        self._action_size = size * size + 2 * (size - 1) * (size - 1)

    # ------------------------------------------------------------------
    # Board serialisation helpers
    # ------------------------------------------------------------------
    def copy(self) -> "QuoridorBoard":
        new_board = QuoridorBoard(self.size, self.walls_per_player)
        new_board.pawns = list(self.pawns)
        new_board.horizontal_walls = self.horizontal_walls.copy()
        new_board.vertical_walls = self.vertical_walls.copy()
        new_board.remaining_walls = list(self.remaining_walls)
        return new_board

    def to_state(self) -> np.ndarray:
        state = np.zeros((6, self.size, self.size), dtype=np.int8)
        first_row, first_col = self.pawns[0]
        second_row, second_col = self.pawns[1]
        state[0, first_row, first_col] = 1
        state[1, second_row, second_col] = 1
        state[2, : self.size - 1, : self.size - 1] = self.horizontal_walls
        state[3, : self.size - 1, : self.size - 1] = self.vertical_walls
        state[4, :, :] = self.remaining_walls[0]
        state[5, :, :] = self.remaining_walls[1]
        return state

    @classmethod
    def from_state(cls, state: np.ndarray, walls_per_player: int = 10) -> "QuoridorBoard":
        size = state.shape[1]
        board = cls(size=size, walls_per_player=walls_per_player)

        first_pos = np.argwhere(state[0] == 1)
        second_pos = np.argwhere(state[1] == 1)
        if len(first_pos) != 1 or len(second_pos) != 1:
            raise ValueError("State does not contain exactly one pawn for each player")
        board.pawns = [tuple(first_pos[0]), tuple(second_pos[0])]
        board.horizontal_walls = state[2, : size - 1, : size - 1].astype(np.int8)
        board.vertical_walls = state[3, : size - 1, : size - 1].astype(np.int8)
        board.remaining_walls = [int(state[4, 0, 0]), int(state[5, 0, 0])]
        return board

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def action_size(self) -> int:
        return self._action_size

    def legal_actions(self, player_index: int) -> np.ndarray:
        valids = np.zeros(self._action_size, dtype=np.int8)
        for move in self._pawn_moves(player_index):
            idx = move[0] * self.size + move[1]
            valids[idx] = 1

        if self.remaining_walls[player_index] <= 0:
            return valids

        offset = self.size * self.size
        span = (self.size - 1) * (self.size - 1)
        for row in range(self.size - 1):
            for col in range(self.size - 1):
                if self._can_place_wall("h", row, col, player_index):
                    valids[offset + row * (self.size - 1) + col] = 1
                if self._can_place_wall("v", row, col, player_index):
                    valids[offset + span + row * (self.size - 1) + col] = 1
        return valids

    def execute_action(self, player_index: int, action: int) -> None:
        move_count = self.size * self.size
        wall_span = (self.size - 1) * (self.size - 1)

        if action < move_count:
            target = (action // self.size, action % self.size)
            legal_moves = {tuple(m) for m in self._pawn_moves(player_index)}
            if target not in legal_moves:
                raise ValueError("Attempted to execute an illegal pawn move")
            self.pawns[player_index] = target
            return

        if self.remaining_walls[player_index] <= 0:
            raise ValueError("Player has no remaining walls to place")

        wall_index = action - move_count
        orientation = "h"
        if wall_index >= wall_span:
            orientation = "v"
            wall_index -= wall_span

        row = wall_index // (self.size - 1)
        col = wall_index % (self.size - 1)

        if not self._can_place_wall(orientation, row, col, player_index):
            raise ValueError("Attempted to place an illegal wall")

        self._set_wall(orientation, row, col, 1)
        self.remaining_walls[player_index] -= 1

    def winner(self) -> int | None:
        if self.pawns[0][0] == self.size - 1:
            return 1
        if self.pawns[1][0] == 0:
            return -1
        return None

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def _pawn_moves(self, player_index: int) -> List[Position]:
        current = self.pawns[player_index]
        opponent = self.pawns[1 - player_index]
        moves: List[Position] = []

        for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (current[0] + delta[0], current[1] + delta[1])
            if not self._is_on_board(adj):
                continue
            if self._is_blocked(current, adj):
                continue
            if adj != opponent:
                moves.append(adj)
                continue

            jump = (adj[0] + delta[0], adj[1] + delta[1])
            if self._is_on_board(jump) and not self._is_blocked(adj, jump):
                moves.append(jump)
            else:
                if delta[0] != 0:
                    candidates = [(adj[0], adj[1] - 1), (adj[0], adj[1] + 1)]
                else:
                    candidates = [(adj[0] - 1, adj[1]), (adj[0] + 1, adj[1])]
                for diag in candidates:
                    if self._is_on_board(diag) and not self._is_blocked(adj, diag):
                        moves.append(diag)
        # Remove duplicates while preserving order
        seen = set()
        unique_moves = []
        for move in moves:
            if move not in seen:
                unique_moves.append(move)
                seen.add(move)
        return unique_moves

    def _is_on_board(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _is_blocked(self, start: Position, end: Position) -> bool:
        sr, sc = start
        er, ec = end
        dr = er - sr
        dc = ec - sc

        if abs(dr) + abs(dc) != 1:
            raise ValueError("_is_blocked expects orthogonally adjacent squares")

        if dr == 1:  # moving down
            if sr >= self.size - 1:
                return True
            if sc < self.size - 1 and self.horizontal_walls[sr, sc]:
                return True
            if sc > 0 and self.horizontal_walls[sr, sc - 1]:
                return True
            return False
        if dr == -1:  # moving up
            if sr <= 0:
                return True
            if sc < self.size - 1 and self.horizontal_walls[sr - 1, sc]:
                return True
            if sc > 0 and self.horizontal_walls[sr - 1, sc - 1]:
                return True
            return False
        if dc == 1:  # moving right
            if sc >= self.size - 1:
                return True
            if sr < self.size - 1 and self.vertical_walls[sr, sc]:
                return True
            if sr > 0 and self.vertical_walls[sr - 1, sc]:
                return True
            return False
        if dc == -1:  # moving left
            if sc <= 0:
                return True
            if sr < self.size - 1 and self.vertical_walls[sr, sc - 1]:
                return True
            if sr > 0 and self.vertical_walls[sr - 1, sc - 1]:
                return True
            return False
        return False

    # ------------------------------------------------------------------
    # Wall placement helpers
    # ------------------------------------------------------------------
    def _can_place_wall(self, orientation: str, row: int, col: int, player_index: int) -> bool:
        if not (0 <= row < self.size - 1 and 0 <= col < self.size - 1):
            return False

        if orientation == "h":
            if self.horizontal_walls[row, col]:
                return False
            if self.vertical_walls[row, col]:
                return False
        else:
            if self.vertical_walls[row, col]:
                return False
            if self.horizontal_walls[row, col]:
                return False

        self._set_wall(orientation, row, col, 1)
        try:
            has_paths = self._has_path(0) and self._has_path(1)
        finally:
            self._set_wall(orientation, row, col, 0)
        return has_paths

    def _set_wall(self, orientation: str, row: int, col: int, value: int) -> None:
        if orientation == "h":
            self.horizontal_walls[row, col] = value
        elif orientation == "v":
            self.vertical_walls[row, col] = value
        else:
            raise ValueError("Unknown wall orientation: expected 'h' or 'v'")

    def _has_path(self, player_index: int) -> bool:
        start = self.pawns[player_index]
        target_row = self.size - 1 if player_index == 0 else 0
        visited = set([start])
        queue: deque[Position] = deque([start])

        while queue:
            current = queue.popleft()
            if current[0] == target_row:
                return True
            for neighbour in self._adjacent_positions(current):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return False

    def _adjacent_positions(self, position: Position) -> Iterable[Position]:
        r, c = position
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in deltas:
            candidate = (r + dr, c + dc)
            if self._is_on_board(candidate) and not self._is_blocked(position, candidate):
                yield candidate
