"""Neural network architectures for spatter classification."""

from .lenet5 import LeNet5, create_lenet5, get_device

__all__ = [
    "LeNet5",
    "create_lenet5",
    "get_device",
]