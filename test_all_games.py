"""Regression tests for verifying PyTorch integrations across the bundled games.

Each test plays two quick games using a randomly initialised neural network against a
random player. Ensure the PyTorch dependencies are installed before running the suite.
"""


import unittest

import Arena
from MCTS import MCTS

from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import RandomPlayer
from othello.pytorch.NNet import NNetWrapper as OthelloPytorchNNet

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.pytorch.NNet import NNetWrapper as TicTacToePytorchNNet

from tictactoe_3d.TicTacToeGame import TicTacToeGame as TicTacToe3DGame
from tictactoe_3d.pytorch.NNet import NNetWrapper as TicTacToe3DPytorchNNet

from connect4.Connect4Game import Connect4Game
from connect4.pytorch.NNet import NNetWrapper as Connect4PytorchNNet

from gobang.GobangGame import GobangGame
from gobang.pytorch.NNet import NNetWrapper as GobangPytorchNNet

from tafl.TaflGame import TaflGame
from tafl.pytorch.NNet import NNetWrapper as TaflPytorchNNet

from rts.RTSGame import RTSGame
from rts.pytorch.NNet import NNetWrapper as RTSPytorchNNet

from dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from dotsandboxes.pytorch.NNet import NNetWrapper as DotsAndBoxesPytorchNNet

import numpy as np
from utils import *

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
