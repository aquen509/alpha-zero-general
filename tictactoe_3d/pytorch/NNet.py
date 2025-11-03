import os
import sys
from typing import Iterable, Tuple

import numpy as np
from tqdm.auto import tqdm

sys.path.append('../../')
from utils import AverageMeter, dotdict
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TicTacToeNNet import TicTacToeNNet as onnet


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
    'num_residual_layers': 4,
    'value_hidden_size': 128,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_z, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def _prepare_boards(self, boards: Iterable[np.ndarray]) -> torch.Tensor:
        tensor = torch.tensor(np.array(boards), dtype=torch.float32)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)
        return tensor

    def train(self, examples: Iterable[Tuple[np.ndarray, np.ndarray, float]]):
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = max(1, int(len(examples) / args.batch_size))

            for _ in tqdm(range(batch_count), desc='Training Net'):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = self._prepare_boards(boards)
                target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32)
                target_vs = torch.tensor(np.array(target_vs), dtype=torch.float32)

                if args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: np.ndarray):
        board = torch.tensor(board.astype(np.float32))
        if board.dim() == 3:
            board = board.unsqueeze(0)
        if args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_z, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    @staticmethod
    def loss_pi(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(targets * outputs) / targets.size(0)

    @staticmethod
    def loss_v(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
