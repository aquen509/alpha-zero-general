from numpy.random import Generator, default_rng

import torch


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


_GLOBAL_RNG: Generator = default_rng()
_TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_rng() -> Generator:
    """Return the shared numpy random number generator used across the project."""

    return _GLOBAL_RNG


def get_device() -> torch.device:
    """Return the preferred torch device (CUDA when available, else CPU)."""

    return _TORCH_DEVICE


def is_cuda_available() -> bool:
    """Check whether the resolved torch device utilises CUDA."""

    return _TORCH_DEVICE.type == 'cuda'


__all__ = ['AverageMeter', 'dotdict', 'get_rng', 'get_device', 'is_cuda_available']
