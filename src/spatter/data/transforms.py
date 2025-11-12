"""Image transforms for spatter data.

This module provides configurable transform pipelines for training and inference.
"""

from typing import Optional

from torchvision import transforms


def get_train_transforms(
    resize: int = 32,
    rotation_degrees: int = 15,
    normalize: bool = False,
) -> transforms.Compose:
    """Get training transforms with augmentation.

    The 1.25x resize factor provides a buffer for rotation to prevent black corners.

    Args:
        resize: Target output size (default: 32)
        rotation_degrees: Random rotation range in degrees (default: 15)
        normalize: Whether to apply ImageNet normalization (default: False)

    Returns:
        Composed transform pipeline
    """
    transform_list = [
        transforms.Resize((int(resize * 1.25), int(resize * 1.25))),
        transforms.RandomRotation(rotation_degrees),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(transform_list)


def get_eval_transforms(
    resize: int = 32,
    normalize: bool = False,
) -> transforms.Compose:
    """Get evaluation transforms without augmentation.

    Args:
        resize: Target output size (default: 32)
        normalize: Whether to apply ImageNet normalization (default: False)

    Returns:
        Composed transform pipeline
    """
    transform_list = [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(transform_list)
