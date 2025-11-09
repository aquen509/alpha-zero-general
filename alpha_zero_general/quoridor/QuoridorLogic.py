from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


Position = Tuple[int, int]


class QuoridorBoard:
    """Implements the movement and wall placement rules for Quoridor."""

    def __init__(self, size: int = 9, walls_per_player: int = 10):
        """Create a new Quoridor board with the default starting layout.

        Args:
            size (int): Dimension of the square board. Must be at least ``3`` to
                allow a path between start and goal rows.
            walls_per_player (int): Number of walls each player has available at
                the start of the game. Must be non-negative.

        Raises:
            ValueError: If ``size`` is less than ``3`` or ``walls_per_player``
                is negative.

        Assumptions:
            ``size`` is typically odd so that the initial pawns are centred, but
            any value ``>=3`` is accepted.
        """
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
        """Return a deep copy of the board state.

        Returns:
            QuoridorBoard: New instance containing identical pawn positions,
            wall placement, and remaining wall counts.

        Assumptions:
            The source board is internally consistent; the copy shares no
            mutable numpy arrays with the original.
        """
        new_board = QuoridorBoard(self.size, self.walls_per_player)
        new_board.pawns = list(self.pawns)
        new_board.horizontal_walls = self.horizontal_walls.copy()
        new_board.vertical_walls = self.vertical_walls.copy()
        new_board.remaining_walls = list(self.remaining_walls)
        return new_board

    def to_state(self) -> np.ndarray:
        """Serialise the board into the tensor representation.

        Returns:
            np.ndarray: A `(6, size, size)` tensor where each plane encodes
            pawn locations, wall placement, and remaining wall counts.

        Notes:
            Planes 0-1 store pawn positions, 2-3 store horizontal and vertical
            walls respectively, and 4-5 replicate the remaining wall counts for
            each player.

        Assumptions:
            Pawn coordinates and wall arrays reside within the board bounds.
        """
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
        """Construct a board instance from a tensor representation.

        Args:
            state (np.ndarray): `(6, size, size)` tensor produced by
                :meth:`to_state`.
            walls_per_player (int): Default number of walls available for each
                player if the tensor omits this information.

        Returns:
            QuoridorBoard: Board matching the supplied tensor.

        Raises:
            ValueError: If the tensor does not contain exactly one pawn for each
                player.

        Assumptions:
            The tensor follows the same layout as produced by
            :meth:`to_state`.
        """
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
        """Total number of encoded actions available in this position.

        Returns:
            int: Number of pawn moves plus horizontal and vertical wall
            placements.

        Assumptions:
            Derived from the board dimensions configured at initialisation.
        """
        return self._action_size

    def legal_actions(self, player_index: int) -> np.ndarray:
        """Compute the legal action mask for a player.

        Args:
            player_index (int): Index of the active player (0 for player 1,
                1 for player -1).

        Returns:
            np.ndarray: Binary vector of length :attr:`action_size` where entries
            are ``1`` for legal actions and ``0`` otherwise.

        Assumptions:
            ``player_index`` is either 0 or 1 and the board encodes a valid
            position.
        """
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
        """Mutate the board by applying the specified action.

        Args:
            player_index (int): Active player executing the action (0 or 1).
            action (int): Encoded pawn move or wall placement index.

        Raises:
            ValueError: If the action is illegal, attempts to place a wall when
                none remain, or references an invalid orientation.

        Assumptions:
            ``action`` follows the encoding produced by
            :meth:`legal_actions` and :attr:`action_size`.
        """
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
        """Identify whether a player has reached the goal row.

        Returns:
            int | None: ``1`` if player 1 reached the bottom row, ``-1`` if
            player -1 reached the top row, or ``None`` if the game is ongoing.

        Assumptions:
            Pawn coordinates are inside the board boundaries.
        """
        if self.pawns[0][0] == self.size - 1:
            return 1
        if self.pawns[1][0] == 0:
            return -1
        return None

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def _pawn_moves(self, player_index: int) -> List[Position]:
        """Generate the legal pawn moves for a player.

        Args:
            player_index (int): Player whose pawn moves are desired (0 or 1).

        Returns:
            List[Position]: Sequence of destination coordinates ``(row, col)``.

        Notes:
            Handles forward, backward, lateral, jump, and diagonal moves as
            permitted by Quoridor rules. Order of moves is deterministic but not
            otherwise significant.

        Assumptions:
            Pawn coordinates and wall placements are consistent with the game's
            rules.
        """
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
        """Determine whether a coordinate lies within the board boundaries.

        Args:
            pos (Tuple[int, int]): Coordinate to evaluate.

        Returns:
            bool: ``True`` if ``pos`` is within the board, otherwise ``False``.

        Assumptions:
            ``pos`` comprises integer row and column indices.
        """
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _is_blocked(self, start: Position, end: Position) -> bool:
        """Check whether movement between orthogonally adjacent squares is blocked.

        Args:
            start (Tuple[int, int]): Origin square.
            end (Tuple[int, int]): Destination square; must be orthogonally
                adjacent to ``start``.

        Returns:
            bool: ``True`` if a wall blocks the movement or the destination is
            outside the board; otherwise ``False``.

        Raises:
            ValueError: If ``start`` and ``end`` are not orthogonally adjacent.

        Assumptions:
            ``start`` and ``end`` reference valid board coordinates.
        """
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
        """Test whether a wall can be placed without violating rules.

        Args:
            orientation (str): ``"h"`` for horizontal or ``"v"`` for vertical.
            row (int): Row index of the wall segment.
            col (int): Column index of the wall segment.
            player_index (int): Player attempting the placement (unused but
                retained for parity with action validation).

        Returns:
            bool: ``True`` if the wall placement is legal; otherwise ``False``.

        Notes:
            A placement is legal only if it does not overlap existing walls and
            both players retain a path to their respective goal rows.

        Assumptions:
            ``orientation`` is either ``"h"`` or ``"v"`` and ``row``/``col``
            refer to valid wall indices.
        """
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
        """Mutate the internal wall arrays for a specific segment.

        Args:
            orientation (str): ``"h"`` or ``"v"`` orientation flag.
            row (int): Wall row index.
            col (int): Wall column index.
            value (int): Numeric value to assign (typically ``0`` or ``1``).

        Raises:
            ValueError: If ``orientation`` is neither ``"h"`` nor ``"v"``.

        Assumptions:
            ``row`` and ``col`` fall within the range ``[0, size - 2]``.
        """
        if orientation == "h":
            self.horizontal_walls[row, col] = value
        elif orientation == "v":
            self.vertical_walls[row, col] = value
        else:
            raise ValueError("Unknown wall orientation: expected 'h' or 'v'")

    def _has_path(self, player_index: int) -> bool:
        """Determine whether a player still has a path to their goal row.

        Args:
            player_index (int): Player whose path accessibility is tested.

        Returns:
            bool: ``True`` if at least one path exists from the current pawn
            position to the player's goal row; ``False`` otherwise.

        Notes:
            Uses a breadth-first search limited by wall constraints.

        Assumptions:
            The board configuration adheres to movement and wall placement
            rules.
        """
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
        """Yield orthogonally adjacent squares not blocked by walls.

        Args:
            position (Tuple[int, int]): Source coordinate.

        Yields:
            Tuple[int, int]: Coordinates of each reachable adjacent square.

        Assumptions:
            ``position`` is on the board and wall arrays match board dimensions.
        """
        r, c = position
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in deltas:
            candidate = (r + dr, c + dc)
            if self._is_on_board(candidate) and not self._is_blocked(position, candidate):
                yield candidate
