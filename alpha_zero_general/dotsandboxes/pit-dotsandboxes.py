import os
from pathlib import Path

import numpy as np

from alpha_zero_general import Arena
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.utils import dotdict

from alpha_zero_general.dotsandboxes.DotsAndBoxesGame import DotsAndBoxesGame
from alpha_zero_general.dotsandboxes.DotsAndBoxesPlayers import (
    GreedyRandomPlayer,
    HumanDotsAndBoxesPlayer,
    RandomPlayer,
)
from alpha_zero_general.dotsandboxes.pytorch.NNet import NNetWrapper


g = DotsAndBoxesGame(n=3)

hp1 = HumanDotsAndBoxesPlayer(g).play
hp2 = HumanDotsAndBoxesPlayer(g).play

rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

grp1 = GreedyRandomPlayer(g).play
grp2 = GreedyRandomPlayer(g).play

numMCTSSims = 50
n1 = NNetWrapper(g)
project_root = Path(__file__).resolve().parents[2]
pretrained_dir = project_root / 'pretrained_models' / 'dotsandboxes' / 'pytorch' / '3x3'
pretrained_file = pretrained_dir / 'best.pth.tar'
if pretrained_file.exists():
    n1.load_checkpoint(str(pretrained_dir), 'best.pth.tar')
else:
    print('No PyTorch checkpoint found at {}. Using randomly initialised network.'.format(pretrained_file))
args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNetWrapper(g)
if pretrained_file.exists():
    n2.load_checkpoint(str(pretrained_dir), 'best.pth.tar')
else:
    print('No PyTorch checkpoint found at {}. Using randomly initialised network.'.format(pretrained_file))
args2 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# Play AlphaZero versus Human
p1 = n1p
p2 = hp1
arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
oneWon, twoWon, draws = arena.playGames(2, verbose=True)
print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play Greedy vs Greedy
# p1 = grp1
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(100, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))

# # Play AlphaZero vs Greedy
# p1 = n1p
# p2 = grp2
# arena = Arena.Arena(p1, p2, g, display=DotsAndBoxesGame.display)
# oneWon, twoWon, draws = arena.playGames(2, verbose=False)
# print("oneWon: {}, twoWon: {}, draws: {}".format(oneWon, twoWon, draws))