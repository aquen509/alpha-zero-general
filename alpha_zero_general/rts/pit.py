from .src.config_class import CONFIG

from .. import Arena
from .RTSGame import RTSGame, display

"""
rts/pit.py

Compares 2 players against each other and outputs num wins p1/ num wins p2/ draws
"""
CONFIG.set_runner('pit')  # set visibility as pit
g = RTSGame()
player1, player2 = CONFIG.pit_args.create_players(g)
arena = Arena.Arena(player1, player2, g, display=display)
print(arena.playGames(CONFIG.pit_args.num_games, verbose=CONFIG.visibility))
