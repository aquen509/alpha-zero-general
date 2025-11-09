import numpy as np

from ..Game import Game
from .QuoridorLogic import QuoridorBoard


class QuoridorGame(Game):
    """AlphaZero-general Game wrapper for Quoridor."""

    def __init__(self, size: int = 9, walls_per_player: int = 10):
        super().__init__()
        self.size = size
        self.walls_per_player = walls_per_player
        self._base_board = QuoridorBoard(size=size, walls_per_player=walls_per_player)

    def getInitBoard(self) -> np.ndarray:
        return self._base_board.to_state()

    def getBoardSize(self):
        return self._base_board.to_state().shape

    def getActionSize(self) -> int:
        return self._base_board.action_size

    def getNextState(self, board, player, action):
        working = QuoridorBoard.from_state(np.copy(board), walls_per_player=self.walls_per_player)
        player_index = 0 if player == 1 else 1
        working.execute_action(player_index, action)
        next_player = -player
        return working.to_state(), next_player

    def getValidMoves(self, board, player):
        working = QuoridorBoard.from_state(board, walls_per_player=self.walls_per_player)
        player_index = 0 if player == 1 else 1
        return working.legal_actions(player_index)

    def getGameEnded(self, board, player):
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
        state = np.copy(board)
        if player == 1:
            return state

        state[[0, 1]] = state[[1, 0]]
        state[[4, 5]] = state[[5, 4]]
        return state

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        size = board.shape[1]
        p1 = tuple(np.argwhere(board[0] == 1)[0])
        p2 = tuple(np.argwhere(board[1] == 1)[0])
        horizontal = board[2]
        vertical = board[3]

        for row in range(size):
            line = []
            for col in range(size):
                if (row, col) == p1:
                    line.append('A')
                elif (row, col) == p2:
                    line.append('B')
                else:
                    line.append('.')
                if col < size - 1:
                    has_wall = False
                    if row < size - 1 and col < size - 1 and vertical[row, col]:
                        has_wall = True
                    if row > 0 and col < size - 1 and vertical[row - 1, col]:
                        has_wall = True
                    line.append('|' if has_wall else ' ')
            print(''.join(line))
            if row < size - 1:
                divider = []
                for col in range(size - 1):
                    filled = row < size - 1 and col < size - 1 and horizontal[row, col]
                    divider.append('-' if filled else ' ')
                    divider.append('-' if filled else ' ')
                print(''.join(divider))
