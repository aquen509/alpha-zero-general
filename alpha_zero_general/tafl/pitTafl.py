
# Note: Run this file from the repository root.

import numpy as np

from alpha_zero_general import Arena
from alpha_zero_general.MCTS import MCTS
from alpha_zero_general.tafl.TaflGame import TaflGame, display
from alpha_zero_general.tafl.TaflPlayers import *
from alpha_zero_general.utils import *
#from alpha_zero_general.tafl.pytorch.NNet import NNetWrapper as NNet

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = TaflGame("Brandubh")

# all players
rp = RandomTaflPlayer(g).play
gp = GreedyTaflPlayer(g).play
hp = HumanTaflPlayer(g).play

# nnet players
#n1 = NNet(g)
#n1.load_checkpoint('./pretrained_models/tafl/pytorch/','6x100x25_best.pth.tar')
#args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


arena = Arena.Arena(hp, gp, g, display=display)
#arena = Arena.Arena(gp, rp, g, display=display)
print(arena.playGames(2, verbose=True))
