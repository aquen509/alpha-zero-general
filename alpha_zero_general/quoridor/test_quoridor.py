import numpy as np
import pytest

from .QuoridorGame import QuoridorGame
from .QuoridorLogic import QuoridorBoard


def positions_from_valids(size, valids):
    move_count = size * size
    indices = np.where(valids[:move_count] == 1)[0]
    return {(idx // size, idx % size) for idx in indices}


def test_initial_board_setup():
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
    game = QuoridorGame(size=5, walls_per_player=5)
    board = game.getInitBoard()

    down_action = 1 * game.size + (game.size // 2)
    valids = game.getValidMoves(board, 1)
    assert valids[down_action] == 1

    next_board, next_player = game.getNextState(board, 1, down_action)
    assert tuple(np.argwhere(next_board[0] == 1)[0]) == (1, 2)
    assert next_player == -1


def test_wall_placement_decrements_remaining_walls():
    board = QuoridorBoard(size=3, walls_per_player=2)
    valids = board.legal_actions(0)

    first_horizontal = board.size * board.size
    assert valids[first_horizontal] == 1

    board.execute_action(0, first_horizontal)
    assert board.remaining_walls[0] == 1
    assert board.horizontal_walls[0, 0] == 1


def test_wall_placement_cannot_block_all_paths():
    board = QuoridorBoard(size=3, walls_per_player=3)
    first_horizontal = board.size * board.size
    second_horizontal = first_horizontal + 1

    board.execute_action(0, first_horizontal)
    valids = board.legal_actions(0)
    assert valids[second_horizontal] == 0

    with pytest.raises(ValueError):
        board.execute_action(0, second_horizontal)


def test_diagonal_moves_when_jump_blocked():
    board = QuoridorBoard(size=5, walls_per_player=5)
    board.pawns = [(2, 2), (2, 3)]
    board._set_wall("v", 2, 3, 1)

    moves = positions_from_valids(board.size, board.legal_actions(0))

    assert (1, 3) in moves
    assert (3, 3) in moves


def test_canonical_form_swaps_perspective():
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
    game = QuoridorGame(size=3, walls_per_player=5)
    board = game.getInitBoard()

    expected = ". X .\n\n. . .\n\n. O ."
    assert game.stringRepresentation(board) == expected


def test_string_representation_includes_walls():
    board = QuoridorBoard(size=3, walls_per_player=5)
    board.pawns = [(0, 0), (2, 2)]
    board.horizontal_walls[0, 0] = 1
    board.vertical_walls[0, 1] = 1

    game = QuoridorGame(size=3, walls_per_player=5)
    rendered = game.stringRepresentation(board.to_state())

    expected = "X .|.\n---\n. .|.\n\n. . O"
    assert rendered == expected


def test_string_representation_vertical_wall_bottom_segment():
    board = QuoridorBoard(size=3, walls_per_player=5)
    board.pawns = [(0, 1), (2, 2)]
    board.vertical_walls[1, 0] = 1

    game = QuoridorGame(size=3, walls_per_player=5)
    rendered = game.stringRepresentation(board.to_state())

    expected = ". X .\n\n.|. .\n\n.|. O"
    assert rendered == expected
