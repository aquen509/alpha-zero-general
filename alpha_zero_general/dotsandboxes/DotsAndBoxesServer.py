from pathlib import Path

import numpy as np
from flask import Flask, request, Response

from ..MCTS import MCTS
from ..utils import dotdict
from .DotsAndBoxesGame import DotsAndBoxesGame
from .DotsAndBoxesPlayers import GreedyRandomPlayer
from .pytorch.NNet import NNetWrapper

app = Flask(__name__)

mcts = None
g = None


# curl -d "board=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" -X POST http://localhost:8888/predict
@app.route('/predict', methods=['POST'])
def predict():
    board = np.fromstring(request.form['board'], sep=',').reshape(g.getBoardSize())

    use_alpha_zero = True
    if use_alpha_zero:
        action = np.argmax(mcts.getActionProb(board, temp=0))
    else:
        action = GreedyRandomPlayer(g).play(board)

    resp = Response(str(action))
    # https://stackoverflow.com/questions/5584923/a-cors-post-request-works-from-plain-javascript-but-why-not-with-jquery
    # https://stackoverflow.com/questions/25860304/how-do-i-set-response-headers-in-flask
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    g = DotsAndBoxesGame(n=3)
    n1 = NNetWrapper(g)
    mcts = MCTS(g, n1, dotdict({'numMCTSSims': 50, 'cpuct': 1.0}))
    project_root = Path(__file__).resolve().parents[2]
    pretrained_dir = project_root / 'pretrained_models' / 'dotsandboxes' / 'pytorch' / '3x3'
    pretrained_file = pretrained_dir / 'best.pth.tar'
    if pretrained_file.exists():
        n1.load_checkpoint(str(pretrained_dir), 'best.pth.tar')
    else:
        print('No PyTorch checkpoint found at {}. Using randomly initialised network.'.format(pretrained_file))
    app.run(debug=False, host='0.0.0.0', port=8888)
