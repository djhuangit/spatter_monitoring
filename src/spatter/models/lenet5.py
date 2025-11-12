"""LeNet5 Model Architecture for Spatter Classification.

This module implements the winning LeNet5 variant that achieved 98.38% test accuracy.
Key difference from classical LeNet5: No activation after convolutional layers.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LeNet5(nn.Module):
    """LeNet5 for binary spatter classification (98.38% test accuracy).

    Architecture:
        Conv2d(3→6, 5×5) → MaxPool(2×2) →
        Conv2d(6→16, 5×5) → MaxPool(2×2) →
        Flatten(400) →
        Linear(400→120, ReLU) → Linear(120→84, ReLU) → Linear(84→1)

    Key Features:
        - No activation after conv layers (modern variant)
        - MaxPool instead of AvgPool
        - Single output neuron for binary classification
        - Device-agnostic (CPU/MPS/CUDA)

    Args:
        in_channels: Number of input channels (default: 3 for RGB, 1 for grayscale)
        num_classes: Number of output classes (default: 1 for binary)

    Input:
        x: [batch, in_channels, 32, 32]

    Output:
        logits: [batch, num_classes] (use BCEWithLogitsLoss for binary classification)

    Example:
        >>> model = LeNet5(in_channels=3, num_classes=1)
        >>> x = torch.randn(2, 3, 32, 32)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([2, 1])
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.model = nn.Sequential(
            # [b, in_channels, 32, 32] -> [b, 6, 28, 28]
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            # [b, 6, 28, 28] -> [b, 6, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, 14, 14] -> [b, 16, 10, 10]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # [b, 16, 10, 10] -> [b, 16, 5, 5]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 16, 5, 5] -> [b, 400]
            nn.Flatten(),
            # [b, 400] -> [b, 120]
            nn.Linear(400, 120),
            nn.ReLU(),
            # [b, 120] -> [b, 84]
            nn.Linear(120, 84),
            nn.ReLU(),
            # [b, 84] -> [b, num_classes]
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, in_channels, 32, 32]

        Returns:
            Logits [batch, num_classes]
        """
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions for binary classification.

        Args:
            x: Input tensor [batch, in_channels, 32, 32]

        Returns:
            Probabilities [batch, 1] in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get class predictions for binary classification.

        Args:
            x: Input tensor [batch, in_channels, 32, 32]
            threshold: Classification threshold (default: 0.5)

        Returns:
            Class predictions [batch] with values {0, 1}
        """
        proba = self.predict_proba(x)
        return (proba > threshold).long().squeeze()

    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device

    def to_device(self, device: Optional[torch.device] = None) -> "LeNet5":
        """Move model to specified device (CPU/MPS/CUDA).

        Args:
            device: Target device. If None, auto-selects best available.

        Returns:
            Self for chaining
        """
        if device is None:
            device = get_device()

        self.to(device)
        logger.info(f"Model moved to {device}")
        return self


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def create_lenet5(
    in_channels: int = 3,
    num_classes: int = 1,
    device: Optional[torch.device] = None,
    pretrained_path: Optional[str] = None,
) -> LeNet5:
    """Factory function to create LeNet5 model.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes (1 for binary classification)
        device: Target device (None = auto-select)
        pretrained_path: Path to pretrained weights (.pth file)

    Returns:
        LeNet5 model on specified device

    Example:
        >>> model = create_lenet5(in_channels=1, device=torch.device("cpu"))
        >>> print(model.get_device())  # cpu
    """
    model = LeNet5(in_channels=in_channels, num_classes=num_classes)

    if pretrained_path:
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to_device(device)
    return model


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)

    # Test with RGB input
    model_rgb = create_lenet5(in_channels=3, device=torch.device("cpu"))
    x_rgb = torch.randn(2, 3, 32, 32)
    logits_rgb = model_rgb(x_rgb)
    print(f"RGB Input: {x_rgb.shape} -> Output: {logits_rgb.shape}")

    # Test with grayscale input (replicated to 3 channels)
    model_gray = create_lenet5(in_channels=3, device=torch.device("cpu"))
    x_gray = torch.randn(2, 1, 32, 32)
    x_gray_replicated = torch.cat([x_gray, x_gray, x_gray], dim=1)
    logits_gray = model_gray(x_gray_replicated)
    print(f"Grayscale Input: {x_gray.shape} -> Replicated: {x_gray_replicated.shape} -> Output: {logits_gray.shape}")

    # Test predictions
    predictions = model_rgb.predict(x_rgb)
    probabilities = model_rgb.predict_proba(x_rgb)
    print(f"Predictions: {predictions}")
    print(f"Probabilities: {probabilities.squeeze()}")
