"""Training script for spatter classification models.

Usage:
    # Train with default config
    python scripts/train.py

    # Override config
    python scripts/train.py training.batch_size=512 data.resize=64

    # Use specific experiment config
    python scripts/train.py +experiment=baseline

    # Enable MLflow tracking
    python scripts/train.py mlflow.enabled=true
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from spatter.data import SpatterGrayDataset
from spatter.models import create_lenet5, get_device

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg: DictConfig) -> tuple:
    """Create train, val, and test dataloaders.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating datasets...")

    # Create datasets
    train_dataset = SpatterGrayDataset(
        file_paths=cfg.data.files,
        labels=cfg.data.labels,
        resize=cfg.data.resize,
        mode="train",
        root=cfg.paths.data_root,
        csv_filename=cfg.data.csv_filename,
    )

    val_dataset = SpatterGrayDataset(
        file_paths=cfg.data.files,
        labels=cfg.data.labels,
        resize=cfg.data.resize,
        mode="val",
        root=cfg.paths.data_root,
        csv_filename=cfg.data.csv_filename,
    )

    test_dataset = SpatterGrayDataset(
        file_paths=cfg.data.files,
        labels=cfg.data.labels,
        resize=cfg.data.resize,
        mode="test",
        root=cfg.paths.data_root,
        csv_filename=cfg.data.csv_filename,
    )

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers if cfg.training.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers if cfg.training.num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
) -> dict:
    """Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        cfg: Hydra configuration

    Returns:
        Dictionary of metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.training.epochs}")

    for batch_idx, (images, labels) in enumerate(pbar):
        # Grayscale [B, 1, H, W] -> RGB [B, 3, H, W]
        if images.shape[1] == 1:
            images = torch.cat([images, images, images], dim=1)

        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass
        loss.backward()

        # Gradient clipping if enabled
        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip.max_norm
            )

        optimizer.step()

        # Metrics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        if (batch_idx + 1) % cfg.training.log_every == 0:
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

    metrics = {
        "loss": running_loss / len(dataloader),
        "accuracy": 100. * correct / total,
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> dict:
    """Evaluate model on validation or test set.

    Args:
        model: Neural network model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        split: Split name for logging ("val" or "test")

    Returns:
        Dictionary of metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f"Evaluating {split}"):
        # Grayscale [B, 1, H, W] -> RGB [B, 3, H, W]
        if images.shape[1] == 1:
            images = torch.cat([images, images, images], dim=1)

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())

        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    metrics = {
        f"{split}_loss": running_loss / len(dataloader),
        f"{split}_accuracy": 100. * correct / total,
    }

    return metrics


def create_run_directory(base_dir: str, experiment_name: str) -> Path:
    """Create timestamped run directory.

    Args:
        base_dir: Base checkpoint directory
        experiment_name: Name of experiment

    Returns:
        Path to run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")
    return run_dir


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    run_dir: Path,
    filename: str,
) -> None:
    """Save model checkpoint with metrics in filename.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Current metrics
        run_dir: Run directory for this training session
        filename: Checkpoint filename
    """
    checkpoint_path = run_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    logger.info(f"Checkpoint saved: {checkpoint_path}")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    setup_logging()
    logger.info("Starting training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.seed)

    # Get device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # Create model
    logger.info("Creating model...")
    model = create_lenet5(
        in_channels=cfg.model.architecture.in_channels,
        num_classes=cfg.model.architecture.num_classes,
        device=device,
        pretrained_path=cfg.model.pretrained,
    )

    # Create run directory with timestamp
    run_dir = create_run_directory(
        cfg.paths.checkpoint_dir,
        cfg.experiment.name
    )

    # Save config to run directory
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved: {config_path}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    if cfg.training.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer.name}")

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{cfg.training.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, split="val")
        logger.info(
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
        )

        # Save checkpoint with metrics in filename
        if epoch % cfg.training.checkpoint.save_every == 0:
            checkpoint_name = f"epoch{epoch:03d}_loss{val_metrics['val_loss']:.4f}.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {**train_metrics, **val_metrics},
                run_dir,
                checkpoint_name,
            )

        # Save best model and check early stopping
        if (
            cfg.training.checkpoint.save_best
            and val_metrics["val_accuracy"] > best_val_acc + cfg.training.early_stopping.min_delta
        ):
            best_val_acc = val_metrics["val_accuracy"]
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0

            # Save with descriptive filename
            best_model_name = f"epoch{epoch:03d}_loss{val_metrics['val_loss']:.4f}_acc{val_metrics['val_accuracy']:.2f}.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                {**train_metrics, **val_metrics},
                run_dir,
                best_model_name,
            )
            logger.info(f"New best model! Val Acc: {best_val_acc:.2f}%, Val Loss: {best_val_loss:.4f}")

            # Keep only the best 3 models (by accuracy in filename)
            checkpoint_files = sorted(
                run_dir.glob("epoch*_acc*.pth"),
                key=lambda p: float(p.stem.split("_acc")[-1]),
                reverse=True
            )
            if len(checkpoint_files) > 3:
                for old_checkpoint in checkpoint_files[3:]:
                    old_checkpoint.unlink()
                    logger.info(f"Deleted old checkpoint: {old_checkpoint.name}")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"No improvement for {epochs_without_improvement} epoch(s) "
                f"(patience: {cfg.training.early_stopping.patience})"
            )

        # Early stopping check
        if (
            cfg.training.early_stopping.enabled
            and epochs_without_improvement >= cfg.training.early_stopping.patience
        ):
            logger.info(
                f"\nEarly stopping triggered! No improvement for "
                f"{cfg.training.early_stopping.patience} epochs."
            )
            logger.info(f"Best Val Acc: {best_val_acc:.2f}%")
            break

    # Final test evaluation
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, split="test")
    logger.info(
        f"Test Loss: {test_metrics['test_loss']:.4f}, "
        f"Test Acc: {test_metrics['test_accuracy']:.2f}%"
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
