import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')
from rts.src.config import FORCE_CPU

if FORCE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class RTSNNet(nn.Module):
    def __init__(self, game, encoder, args):
        super().__init__()
        from rts.src.config_class import CONFIG

        self.board_x, self.board_y, _ = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        num_encoders = encoder.num_encoders
        self.input_conv = nn.Conv2d(num_encoders, args.num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(args.num_channels)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(args.num_channels) for _ in range(CONFIG.nnet_args.num_residual_layers)
        ]) if hasattr(CONFIG.nnet_args, 'num_residual_layers') else nn.ModuleList(
            [ResidualBlock(args.num_channels) for _ in range(4)]
        )

        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        hidden_size = getattr(CONFIG.nnet_args, 'value_hidden_size', 256)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, s: torch.Tensor):
        x = s.view(-1, s.size(1), self.board_x, self.board_y)
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.residual_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.board_x * self.board_y)
        p = F.dropout(p, p=self.args.dropout, training=self.training)
        pi = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, self.board_x * self.board_y)
        v = F.dropout(F.relu(self.value_fc1(v)), p=self.args.dropout, training=self.training)
        v = self.value_fc2(v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
