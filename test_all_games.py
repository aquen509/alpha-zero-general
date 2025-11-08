"""Regression tests for verifying PyTorch integrations across the bundled games.

Each test plays two quick games using a randomly initialised neural network against a
random player. Ensure the PyTorch dependencies are installed before running the suite.
"""


import unittest

import numpy as np

from alpha_zero_general import Arena
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.connect4.Connect4Game import Connect4Game
from alpha_zero_general.connect4.pytorch.NNet import (
    NNetWrapper as Connect4PytorchNNet,
)
from alpha_zero_general.dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from alpha_zero_general.dotsandboxes.pytorch.NNet import (
    NNetWrapper as DotsAndBoxesPytorchNNet,
)
from alpha_zero_general.gobang.GobangGame import GobangGame
from alpha_zero_general.gobang.pytorch.NNet import NNetWrapper as GobangPytorchNNet
from alpha_zero_general.othello.OthelloGame import OthelloGame
from alpha_zero_general.othello.OthelloPlayers import RandomPlayer
from alpha_zero_general.othello.pytorch.NNet import (
    NNetWrapper as OthelloPytorchNNet,
)
from alpha_zero_general.rts.RTSGame import RTSGame
from alpha_zero_general.rts.pytorch.NNet import NNetWrapper as RTSPytorchNNet
from alpha_zero_general.tafl.TaflGame import TaflGame
from alpha_zero_general.tafl.pytorch.NNet import NNetWrapper as TaflPytorchNNet
from alpha_zero_general.tictactoe.TicTacToeGame import TicTacToeGame
from alpha_zero_general.tictactoe.pytorch.NNet import (
    NNetWrapper as TicTacToePytorchNNet,
)
from alpha_zero_general.tictactoe_3d.TicTacToeGame import (
    TicTacToeGame as TicTacToe3DGame,
)
from alpha_zero_general.tictactoe_3d.pytorch.NNet import (
    NNetWrapper as TicTacToe3DPytorchNNet,
)
from alpha_zero_general.utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))
   
    def test_othello_pytorch(self):
        self.execute_game_test(OthelloGame(6), OthelloPytorchNNet)

    def test_tictactoe_pytorch(self):
        self.execute_game_test(TicTacToeGame(), TicTacToePytorchNNet)

    def test_tictactoe3d_pytorch(self):
        self.execute_game_test(TicTacToe3DGame(3), TicTacToe3DPytorchNNet)

    def test_gobang_pytorch(self):
        self.execute_game_test(GobangGame(), GobangPytorchNNet)

    def test_tafl_pytorch(self):
        self.execute_game_test(TaflGame(5), TaflPytorchNNet)

    def test_connect4_pytorch(self):
        self.execute_game_test(Connect4Game(5), Connect4PytorchNNet)

    def test_rts_pytorch(self):
        self.execute_game_test(RTSGame(), RTSPytorchNNet)

    def test_dotsandboxes_pytorch(self):
        self.execute_game_test(DotsAndBoxesGame(3), DotsAndBoxesPytorchNNet)

if __name__ == '__main__':
    unittest.main()
