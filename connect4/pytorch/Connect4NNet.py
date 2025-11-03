import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from utils import dotdict


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


class Connect4NNet(nn.Module):
    def __init__(self, game, args: dotdict):
        super().__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.input_conv = nn.Conv2d(1, args.num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(args.num_channels)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(args.num_channels) for _ in range(args.num_residual_layers)
        ])

        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, args.value_hidden_size)
        self.value_fc2 = nn.Linear(args.value_hidden_size, 1)

    def forward(self, s: torch.Tensor):
        x = s.view(-1, 1, self.board_x, self.board_y)
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
