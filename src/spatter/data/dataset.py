"""Spatter Dataset Implementation.

This module provides PyTorch Dataset classes for loading spatter images from HDF5 files
with lazy loading for memory efficiency.

Key Features:
- Lazy loading: Images decoded on-the-fly from HDF5
- CSV indexing: Maps indices to (file, dataset, index, label)
- Grayscale support: Single-channel output for efficient training
- Train/val/test splits: 60/20/20 split support
"""

import csv
import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

SplitMode = Literal["train", "val", "test", "full"]


def decode_jpeg_bytes(io_buf: bytes) -> np.ndarray:
    """Decode JPEG bytes from HDF5 to numpy array.

    Args:
        io_buf: JPEG-encoded bytes from HDF5

    Returns:
        Decoded image as numpy array
    """
    return cv2.imdecode(np.frombuffer(io_buf, np.uint8), -1)


class SpatterDataset(Dataset):
    """RGB Spatter Dataset with lazy loading from HDF5.

    This dataset loads spatter images in RGB format with configurable transforms.
    Images are decoded on-the-fly from JPEG-compressed HDF5 files.

    Args:
        file_paths: List of HDF5 filenames (relative to root)
        labels: List of labels corresponding to each file
        resize: Target image size (will be resized to resize x resize)
        mode: Dataset split mode ("train", "val", "test", "full")
        root: Root directory containing HDF5 files
        csv_filename: Name of CSV index file (default: "images.csv")
        transform: Optional custom transform (overrides default)

    Example:
        >>> dataset = SpatterDataset(
        ...     file_paths=["left_low_0.h5", "left_high_2.h5"],
        ...     labels=[0, 1],
        ...     resize=32,
        ...     mode="train",
        ...     root="data/processed/hdf5"
        ... )
        >>> img, label = dataset[0]
        >>> print(img.shape, label)  # torch.Size([3, 32, 32]) tensor(0)
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        resize: int,
        mode: SplitMode = "train",
        root: str = "",
        csv_filename: str = "images.csv",
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()

        self.root = Path(root)
        self.file_paths = file_paths
        self.labels_list = labels
        self.resize = resize
        self.mode = mode
        self.csv_filename = csv_filename

        if len(self.file_paths) != len(self.labels_list):
            raise ValueError(
                f"Mismatch between file_paths ({len(self.file_paths)}) "
                f"and labels ({len(self.labels_list)})"
            )

        # Load CSV index
        self.images, self.labels = self._load_csv(csv_filename)

        # Apply split
        self._apply_split()

        # Setup transform
        self.transform = transform or self._default_transform()

        logger.info(
            f"Loaded {mode} dataset: {len(self)} samples "
            f"from {len(self.file_paths)} files"
        )

    def _default_transform(self) -> transforms.Compose:
        """Default transform pipeline for RGB images.

        Returns:
            Composed transforms: Resize (1.25x) → RandomRotation → CenterCrop → ToTensor
        """
        return transforms.Compose([
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
        ])

    def _apply_split(self) -> None:
        """Apply train/val/test split to loaded data."""
        total_len = len(self.images)

        if self.mode == "train":  # 60%
            self.images = self.images[:int(0.6 * total_len)]
            self.labels = self.labels[:int(0.6 * total_len)]
        elif self.mode == "val":  # 20% (60-80%)
            start_idx = int(0.6 * total_len)
            end_idx = int(0.8 * total_len)
            self.images = self.images[start_idx:end_idx]
            self.labels = self.labels[start_idx:end_idx]
        elif self.mode == "test":  # 20% (80-100%)
            start_idx = int(0.8 * total_len)
            self.images = self.images[start_idx:]
            self.labels = self.labels[start_idx:]
        # mode == "full": use all data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index with lazy loading from HDF5.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (image_tensor, label_tensor)
            - image_tensor: [C, H, W] RGB image
            - label_tensor: Scalar label
        """
        # Get metadata from CSV index
        img_meta, label = self.images[idx], self.labels[idx]
        file_path = self.root / img_meta[0]
        dataset_name = img_meta[1]
        ds_idx = int(img_meta[2])

        # Lazy load from HDF5
        with h5py.File(file_path, "r") as h5_file:
            io_buf = h5_file[dataset_name][ds_idx]
            img_array = decode_jpeg_bytes(io_buf)

            # Convert to PIL Image (RGB)
            img_pil = Image.fromarray(img_array).convert("RGB")

            # Apply transforms
            img_tensor = self.transform(img_pil)
            label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)

    def _load_csv(self, filename: str) -> Tuple[List[List[str]], List[int]]:
        """Load CSV index file or create if not exists.

        Args:
            filename: CSV filename (e.g., "images.csv")

        Returns:
            Tuple of (images_list, labels_list)
            - images_list: [[file, dataset, idx], ...]
            - labels_list: [label, ...]
        """
        csv_path = self.root / filename

        if not csv_path.exists():
            logger.info(f"CSV not found at {csv_path}, creating...")
            self._create_csv(filename)

        # Read from CSV
        logger.info(f"Loading CSV from {csv_path}")
        images, labels = [], []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                ds_file, ds_name, ds_idx, ds_label = row
                images.append([ds_file, ds_name, ds_idx])
                labels.append(int(ds_label))

        assert len(images) == len(labels), "CSV data mismatch"
        return images, labels

    def _create_csv(self, filename: str) -> None:
        """Create CSV index files from HDF5 files.

        Creates two CSV files:
        - unshuffled_{filename}: Original HDF5 order
        - {filename}: Shuffled for training

        Args:
            filename: Target CSV filename (e.g., "images.csv")
        """
        datasets = []
        ds_lens = []
        ds_labels = []
        ds_files = []

        # Scan all HDF5 files
        for file, label in zip(self.file_paths, self.labels_list):
            file_path = self.root / file
            with h5py.File(file_path, "r") as h5_file:
                for key in h5_file.keys():
                    datasets.append(key)
                    ds_lens.append(h5_file[key].shape[0])
                    ds_labels.append(label)
                    ds_files.append(file)

        logger.info(f"Found {len(datasets)} datasets across {len(self.file_paths)} files")

        # Write unshuffled CSV
        unshuffled_path = self.root / f"unshuffled_{filename}"
        with open(unshuffled_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "dataset", "ds_idx", "label"])
            for idx, ds in enumerate(datasets):
                for i in range(ds_lens[idx]):
                    writer.writerow([ds_files[idx], ds, i, ds_labels[idx]])
        logger.info(f"Created unshuffled CSV: {unshuffled_path}")

        # Create shuffled version
        df = pd.read_csv(unshuffled_path)
        df = df.sample(frac=1, random_state=42)  # Shuffle with fixed seed
        shuffled_path = self.root / filename
        df.to_csv(shuffled_path, index=False)
        logger.info(f"Created shuffled CSV: {shuffled_path}")


class SpatterGrayDataset(SpatterDataset):
    """Grayscale Spatter Dataset (USED IN TRAINING).

    Identical to SpatterDataset but outputs single-channel grayscale images.
    This is the dataset used for training the 98.38% accuracy model.

    Output shape: [1, H, W] instead of [3, H, W]

    Example:
        >>> dataset = SpatterGrayDataset(
        ...     file_paths=["left_low_0.h5", "left_high_2.h5"],
        ...     labels=[0, 1],
        ...     resize=32,
        ...     mode="train",
        ...     root="data/processed/hdf5"
        ... )
        >>> img, label = dataset[0]
        >>> print(img.shape)  # torch.Size([1, 32, 32])
    """

    def _default_transform(self) -> transforms.Compose:
        """Default transform pipeline for grayscale images."""
        return transforms.Compose([
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get grayscale item by index.

        Returns:
            Tuple of (image_tensor, label_tensor)
            - image_tensor: [1, H, W] grayscale image
            - label_tensor: Scalar label
        """
        img_meta, label = self.images[idx], self.labels[idx]
        file_path = self.root / img_meta[0]
        dataset_name = img_meta[1]
        ds_idx = int(img_meta[2])

        with h5py.File(file_path, "r") as h5_file:
            io_buf = h5_file[dataset_name][ds_idx]
            img_array = decode_jpeg_bytes(io_buf)

            # Convert to grayscale PIL Image
            img_pil = Image.fromarray(img_array).convert("L")

            img_tensor = self.transform(img_pil)
            label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
