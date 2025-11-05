import os
from typing import Iterable, Tuple

import numpy as np
from tqdm.auto import tqdm

from ...NeuralNet import NeuralNet
from ...utils import AverageMeter, dotdict, get_rng

import torch
import torch.optim as optim

from ..src.config import VERBOSE_MODEL_FIT
from .RTSNNet import RTSNNet


class NNetWrapper(NeuralNet):
    def __init__(self, game, encoder=None):
        from ..src.config_class import CONFIG

        encoder = encoder or CONFIG.nnet_args.encoder
        self.encoder = encoder

        cuda_available = torch.cuda.is_available()
        self.args = dotdict({
            'lr': CONFIG.nnet_args.lr,
            'dropout': CONFIG.nnet_args.dropout,
            'epochs': CONFIG.nnet_args.epochs,
            'batch_size': CONFIG.nnet_args.batch_size,
            'cuda': cuda_available and CONFIG.nnet_args.cuda,
            'num_channels': CONFIG.nnet_args.num_channels,
            'num_residual_layers': getattr(CONFIG.nnet_args, 'num_residual_layers', 4),
            'value_hidden_size': getattr(CONFIG.nnet_args, 'value_hidden_size', 256),
        })

        CONFIG.nnet_args.num_residual_layers = self.args.num_residual_layers
        CONFIG.nnet_args.value_hidden_size = self.args.value_hidden_size

        self.nnet = RTSNNet(game, encoder, self.args)
        self.board_x, self.board_y, _ = game.getBoardSize()
        self.action_size = game.getActionSize()

        if self.args.cuda:
            self.nnet.cuda()

    def _prepare_boards(self, boards: Iterable[np.ndarray]) -> torch.Tensor:
        encoded = self.encoder.encode_multiple(np.array(boards))
        tensor = torch.tensor(encoded, dtype=torch.float32)
        if tensor.dim() == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def train(self, examples: Iterable[Tuple[np.ndarray, np.ndarray, float]]):
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = max(1, int(len(examples) / self.args.batch_size))

            rng = get_rng()
            for _ in tqdm(range(batch_count), desc='Training Net', disable=not VERBOSE_MODEL_FIT):
                sample_ids = rng.integers(len(examples), size=self.args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))

                boards_tensor = self._prepare_boards(boards)
                target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32)
                target_vs = torch.tensor(np.array(target_vs), dtype=torch.float32)

                if self.args.cuda:
                    boards_tensor = boards_tensor.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards_tensor)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards_tensor.size(0))
                v_losses.update(l_v.item(), boards_tensor.size(0))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: np.ndarray, player=None):
        encoded = self.encoder.encode(board)
        tensor = torch.tensor(encoded, dtype=torch.float32)
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        if self.args.cuda:
            tensor = tensor.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(tensor)
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
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
