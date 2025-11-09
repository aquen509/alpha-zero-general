import numpy as np
import pytest

from .QuoridorGame import QuoridorGame
from .QuoridorLogic import QuoridorBoard


def positions_from_valids(size, valids):
    """Translate legal-move mask into board coordinates.

    Args:
        size (int): Dimension of the square board used for indexing.
        valids (np.ndarray): Flat binary vector where a value of ``1`` marks a
            legal action. Only the first ``size * size`` entries (pawn moves) are
            considered.

    Returns:
        Set[Tuple[int, int]]: Coordinates corresponding to legal pawn moves.

    Assumptions:
        ``valids`` has length at least ``size * size`` and represents actions in
        the encoding returned by :meth:`QuoridorBoard.legal_actions`.
    """
    move_count = size * size
    indices = np.where(valids[:move_count] == 1)[0]
    return {(idx // size, idx % size) for idx in indices}


def test_initial_board_setup():
    """Ensure newly created boards place pawns and walls correctly.

    Assumptions:
        The default constructor for :class:`QuoridorBoard` mirrors the standard
        Quoridor starting setup without pre-placed walls.
    """
    board = QuoridorBoard(size=5, walls_per_player=5)
    state = board.to_state()

    p1_pos = tuple(np.argwhere(state[0] == 1)[0])
    p2_pos = tuple(np.argwhere(state[1] == 1)[0])
    assert p1_pos == (0, 2)
    assert p2_pos == (4, 2)

    assert np.count_nonzero(state[2]) == 0
    assert np.count_nonzero(state[3]) == 0

    assert state[4, 0, 0] == 5
    assert state[5, 0, 0] == 5


def test_next_state_after_pawn_move():
    """Verify pawn moves update the board tensor and player turn.

    Assumptions:
        :class:`QuoridorGame` wraps :class:`QuoridorBoard` without modifying the
        action encoding.
    """
    game = QuoridorGame(size=5, walls_per_player=5)
    board = game.getInitBoard()

    down_action = 1 * game.size + (game.size // 2)
    valids = game.getValidMoves(board, 1)
    assert valids[down_action] == 1

    next_board, next_player = game.getNextState(board, 1, down_action)
    assert tuple(np.argwhere(next_board[0] == 1)[0]) == (1, 2)
    assert next_player == -1


def test_wall_placement_decrements_remaining_walls():
    """Check wall placement reduces the remaining wall count.

    Assumptions:
        Placing the first horizontal wall at index 0 is legal on an empty board.
    """
    board = QuoridorBoard(size=3, walls_per_player=2)
    valids = board.legal_actions(0)

    first_horizontal = board.size * board.size
    assert valids[first_horizontal] == 1

    board.execute_action(0, first_horizontal)
    assert board.remaining_walls[0] == 1
    assert board.horizontal_walls[0, 0] == 1


def test_wall_placement_cannot_block_all_paths():
    """Confirm wall placement validation forbids blocking every path.

    Assumptions:
        The second adjacent horizontal wall would block the only path on a 3x3
        board and should therefore be rejected.
    """
    board = QuoridorBoard(size=3, walls_per_player=3)
    first_horizontal = board.size * board.size
    second_horizontal = first_horizontal + 1

    board.execute_action(0, first_horizontal)
    valids = board.legal_actions(0)
    assert valids[second_horizontal] == 0

    with pytest.raises(ValueError):
        board.execute_action(0, second_horizontal)


def test_diagonal_moves_when_jump_blocked():
    """Ensure diagonal moves become available when direct jumps are blocked.

    Assumptions:
        Placing a vertical wall behind the opponent prevents a straight jump,
        triggering diagonal move options per Quoridor rules.
    """
    board = QuoridorBoard(size=5, walls_per_player=5)
    board.pawns = [(2, 2), (2, 3)]
    board._set_wall("v", 2, 3, 1)

    moves = positions_from_valids(board.size, board.legal_actions(0))

    assert (1, 3) in moves
    assert (3, 3) in moves


def test_canonical_form_swaps_perspective():
    """Validate canonical form swaps player-specific planes.

    Assumptions:
        Canonicalisation performs an in-place swap of pawn and wall planes for
        the opposing perspective without additional transformations.
    """
    board = QuoridorBoard(size=5, walls_per_player=5)
    board.pawns = [(1, 2), (3, 1)]
    board.remaining_walls = [4, 7]
    board.horizontal_walls[0, 2] = 1
    board.vertical_walls[1, 1] = 1

    game = QuoridorGame(size=5, walls_per_player=5)
    state = board.to_state()
    canonical = game.getCanonicalForm(state, player=-1)

    assert np.array_equal(canonical[0], state[1])
    assert np.array_equal(canonical[1], state[0])
    assert np.array_equal(canonical[4], state[5])
    assert np.array_equal(canonical[5], state[4])


def test_game_ended_respects_player_perspective():
    """Confirm termination scores are reported relative to the caller.

    Assumptions:
        The winner check exclusively depends on pawn rows relative to the goal
        line for each player.
    """
    board = QuoridorBoard(size=5, walls_per_player=5)
    board.pawns[0] = (4, 2)
    state = board.to_state()

    game = QuoridorGame(size=5, walls_per_player=5)
    assert game.getGameEnded(state, 1) == 1
    assert game.getGameEnded(state, -1) == -1

    board.pawns[0] = (0, 2)
    board.pawns[1] = (0, 1)
    state = board.to_state()
    assert game.getGameEnded(state, 1) == -1
    assert game.getGameEnded(state, -1) == 1


def test_string_representation_initial_board():
    """Check textual rendering for an empty starting board.

    Assumptions:
        Rendering uses the same shared helper for display and string output.
    """
    game = QuoridorGame(size=3, walls_per_player=5)
    board = game.getInitBoard()

    expected = ". X .\n\n. . .\n\n. O ."
    assert game.stringRepresentation(board) == expected


def test_string_representation_includes_walls():
    """Ensure rendered text includes horizontal and vertical walls.

    Assumptions:
        Wall planes encode horizontal segments before vertical segments in the
        state tensor.
    """
    board = QuoridorBoard(size=3, walls_per_player=5)
    board.pawns = [(0, 0), (2, 2)]
    board.horizontal_walls[0, 0] = 1
    board.vertical_walls[0, 1] = 1

    game = QuoridorGame(size=3, walls_per_player=5)
    rendered = game.stringRepresentation(board.to_state())

    expected = "X .|.\n---\n. .|.\n\n. . O"
    assert rendered == expected


def test_string_representation_vertical_wall_bottom_segment():
    """Verify vertical walls render correctly when spanning lower segments.

    Assumptions:
        Rendering inspects both segments adjacent to a vertical wall to place
        the pipe character between squares.
    """
    board = QuoridorBoard(size=3, walls_per_player=5)
    board.pawns = [(0, 1), (2, 2)]
    board.vertical_walls[1, 0] = 1

    game = QuoridorGame(size=3, walls_per_player=5)
    rendered = game.stringRepresentation(board.to_state())

    expected = ". X .\n\n.|. .\n\n.|. O"
    assert rendered == expected
