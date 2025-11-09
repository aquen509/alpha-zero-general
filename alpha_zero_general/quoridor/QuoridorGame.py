import numpy as np

from ..Game import Game
from .QuoridorLogic import QuoridorBoard


class QuoridorGame(Game):
    """AlphaZero-general Game wrapper for Quoridor."""

    def __init__(self, size: int = 9, walls_per_player: int = 10):
        """Initialise the Quoridor AlphaZero game wrapper.

        Args:
            size (int): Length of one edge of the square board. Must be at
                least 3 and is typically an odd number so that the starting
                pawns can be centred.
            walls_per_player (int): Number of walls available to each player at
                the beginning of the game. Must be non-negative.

        Assumptions:
            The caller provides a valid board size and wall count compatible
            with :class:`QuoridorBoard`.
        """
        super().__init__()
        self.size = size
        self.walls_per_player = walls_per_player
        self._base_board = QuoridorBoard(size=size, walls_per_player=walls_per_player)

    def getInitBoard(self) -> np.ndarray:
        """Return a fresh Quoridor board state.

        Returns:
            np.ndarray: A `(6, size, size)` tensor describing the initial board
            layout, matching :meth:`QuoridorBoard.to_state`.

        Assumptions:
            The underlying :class:`QuoridorBoard` instance uses the canonical
            starting positions for both players and contains no walls.
        """
        return self._base_board.to_state()

    def getBoardSize(self):
        """Return the tensor shape for the Quoridor board representation.

        Returns:
            Tuple[int, int, int]: Dimensions of the numpy tensor used to store
            the board state `(planes, rows, cols)`.

        Assumptions:
            The base board follows the AlphaZero tensor encoding of
            :class:`QuoridorBoard`.
        """
        return self._base_board.to_state().shape

    def getActionSize(self) -> int:
        """Return the number of discrete actions supported by the game.

        Returns:
            int: Count of pawn moves plus wall placements as encoded by
            :class:`QuoridorBoard`.

        Assumptions:
            Derived from the base board instance constructed during
            initialisation.
        """
        return self._base_board.action_size

    def getNextState(self, board, player, action):
        """Apply an action to a board state and swap the active player.

        Args:
            board (np.ndarray): A `(6, size, size)` tensor describing the
                current position.
            player (int): The perspective to move from. Uses 1 for the first
                player and -1 for the second.
            action (int): Encoded move or wall placement index, as produced by
                :meth:`getValidMoves`.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the new board tensor and
            the next player indicator (-player).

        Assumptions:
            ``board`` originated from :class:`QuoridorBoard`. The provided
            ``action`` is legal for the ``player`` given the ``board`` state.
        """
        working = QuoridorBoard.from_state(np.copy(board), walls_per_player=self.walls_per_player)
        player_index = 0 if player == 1 else 1
        working.execute_action(player_index, action)
        next_player = -player
        return working.to_state(), next_player

    def getValidMoves(self, board, player):
        """List legal actions for the supplied position.

        Args:
            board (np.ndarray): A `(6, size, size)` tensor for the position to
                query.
            player (int): Perspective whose legal actions should be returned.

        Returns:
            np.ndarray: A flat vector of length :meth:`getActionSize`, where a
            value of 1 marks a legal action and 0 marks an illegal action.

        Assumptions:
            ``board`` corresponds to a valid Quoridor position and ``player`` is
            either 1 or -1.
        """
        working = QuoridorBoard.from_state(board, walls_per_player=self.walls_per_player)
        player_index = 0 if player == 1 else 1
        return working.legal_actions(player_index)

    def getGameEnded(self, board, player):
        """Evaluate the game termination status from the player's perspective.

        Args:
            board (np.ndarray): The `(6, size, size)` tensor for the current
                position.
            player (int): Perspective in which the result should be reported.

        Returns:
            int: ``1`` if the supplied player has won, ``-1`` if the opponent
            has won, otherwise ``0`` when the game is ongoing.

        Assumptions:
            ``board`` is a valid Quoridor state.
        """
        working = QuoridorBoard.from_state(board, walls_per_player=self.walls_per_player)
        winner = working.winner()
        if winner is None:
            return 0
        if winner == player:
            return 1
        if winner == -player:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        """Canonicalise a board to the perspective of the given player.

        Args:
            board (np.ndarray): The position tensor to canonicalise.
            player (int): ``1`` returns the board unchanged, ``-1`` swaps pawn
                and wall planes to mirror the opponent's perspective.

        Returns:
            np.ndarray: Board tensor expressed from the viewpoint of ``player``.

        Assumptions:
            ``board`` originated from :class:`QuoridorBoard` and contains player
            planes in the standard order.
        """
        state = np.copy(board)
        if player == 1:
            return state

        state[[0, 1]] = state[[1, 0]]
        state[[4, 5]] = state[[5, 4]]
        return state

    def getSymmetries(self, board, pi):
        """Return the symmetry-equivalent board-policy pairs.

        Args:
            board (np.ndarray): The current position tensor.
            pi (np.ndarray): Probability distribution over actions for ``board``.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: The AlphaZero symmetry list.
            Quoridor currently exposes no non-trivial symmetries, so the list
            contains a single `(board, pi)` pair.

        Assumptions:
            ``board`` and ``pi`` already align with each other and respect the
            action encoding.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """Render the board tensor as a human-readable string.

        Args:
            board (np.ndarray): Position tensor to render.

        Returns:
            str: Multi-line grid featuring pawn markers and wall segments.

        Assumptions:
            ``board`` is a valid state produced by :class:`QuoridorBoard`.
        """
        return self._render_state(board)

    @staticmethod
    def display(board):
        """Print a textual representation of the supplied board tensor.

        Args:
            board (np.ndarray): Position tensor compatible with
                :meth:`stringRepresentation`.

        Assumptions:
            ``board`` is a valid Quoridor state tensor.
        """
        print(QuoridorGame._render_state(board))

    @staticmethod
    def _render_state(board: np.ndarray) -> str:
        """Build a human-readable view of the board state.

        Args:
            board (np.ndarray): A `(6, size, size)` tensor representing the
                current game state.

        Returns:
            str: Newline-delimited grid that marks pawns (``X``/``O``), empty
            squares (``.``), and the segments of walls between squares.

        Assumptions:
            The board tensor contains exactly one pawn per player and wall
            planes aligned with the encoding defined in
            :class:`QuoridorBoard`.
        """
        size = board.shape[1]
        p1 = tuple(np.argwhere(board[0] == 1)[0])
        p2 = tuple(np.argwhere(board[1] == 1)[0])
        horizontal = board[2]
        vertical = board[3]

        lines: list[str] = []
        for row in range(size):
            row_chars: list[str] = []
            for col in range(size):
                if (row, col) == p1:
                    row_chars.append("X")
                elif (row, col) == p2:
                    row_chars.append("O")
                else:
                    row_chars.append(".")

                if col < size - 1:
                    has_wall = False
                    if row < size - 1 and vertical[row, col]:
                        has_wall = True
                    if row > 0 and vertical[row - 1, col]:
                        has_wall = True
                    row_chars.append("|" if has_wall else " ")
            lines.append("".join(row_chars))

            if row < size - 1:
                divider: list[str] = []
                for col in range(size):
                    has_wall = False
                    if col < size - 1 and horizontal[row, col]:
                        has_wall = True
                    if col > 0 and horizontal[row, col - 1]:
                        has_wall = True
                    divider.append("-" if has_wall else " ")

                    if col < size - 1:
                        divider.append("-" if horizontal[row, col] else " ")

                lines.append("".join(divider).rstrip())

        return "\n".join(lines)
