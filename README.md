# 🔥 Spatter Monitoring System

> **Production-ready MLOps pipeline for binary classification of metal 3D printing spatter patterns**
>
> Refactored from [my PhD research](https://hdl.handle.net/10356/155240) to a production-grade ML system

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Test Accuracy](https://img.shields.io/badge/test%20accuracy-98.38%25-brightgreen.svg)]()
[![Dataset Size](https://img.shields.io/badge/dataset-598K%20images-orange.svg)]()
[![Device Support](https://img.shields.io/badge/device-CPU%20%7C%20MPS%20%7C%20CUDA-blue.svg)]()

---

## 📊 Quick Stats

- **Accuracy**: 98.38% test accuracy
- **Dataset**: 598,933 images (2.1GB with 92% compression)
- **Architecture**: LeNet5 with device-agnostic training (CPU/MPS/CUDA)
- **Features**: Hydra configs, early stopping, auto-checkpointing

---

## 🛠️ Tech Stack

**Core ML**
- PyTorch 2.9+ - Deep learning framework
- torchvision - Image transforms and augmentation
- NumPy - Numerical computing

**Data Processing**
- HDF5 (h5py) - Efficient storage with lazy loading
- OpenCV - Video processing and JPEG codec
- Pandas - CSV indexing and data manipulation

**MLOps & Configuration**
- Hydra - Configuration management
- Python logging - Structured logging
- tqdm - Progress tracking

**Development**
- uv - Fast Python package manager
- pytest - Testing framework (planned)
- ruff - Modern linter and formatter

**Deployment Ready**
- Device agnostic (CPU/MPS/CUDA)
- FastAPI integration (planned)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install with training dependencies
uv sync
```

### 2. Train Model

```bash
# Train with default config
uv run python scripts/train.py

# Override parameters
uv run python scripts/train.py training.batch_size=512 training.epochs=30

# Use specific device
uv run python scripts/train.py device=mps    # Apple Silicon
uv run python scripts/train.py device=cuda   # NVIDIA GPU
uv run python scripts/train.py device=cpu    # CPU only
```

### 3. Monitor Training

Training automatically:
- ✅ Saves checkpoints with metrics in filenames
- ✅ Keeps only top 3 best models
- ✅ Stops early if no improvement (configurable patience)
- ✅ Creates timestamped run directories
- ✅ Saves config for reproducibility

**Example output:**
```
data/models/checkpoints/
└── baseline_20251113_143025/
    ├── config.yaml
    ├── epoch005_loss0.0423.pth
    ├── epoch010_loss0.0398.pth
    └── epoch018_loss0.0356_acc98.45.pth  ← Best model
```

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Getting started guide |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Project structure |
| **[MIGRATION_STATUS.md](MIGRATION_STATUS.md)** | Migration progress |
| **[docs/ANALYSIS.md](docs/ANALYSIS.md)** | Technical deep-dive |
| **[CLAUDE.md](CLAUDE.md)** | AI agent guidance |

---

## 🏗️ Project Structure

```
spatter_monitoring/
├── src/spatter/              # Production code
│   ├── data/                 # Dataset & transforms
│   ├── models/               # LeNet5 architecture
│   ├── training/             # Training infrastructure
│   └── inference/            # Prediction pipeline
├── config/                   # Hydra configurations
│   ├── config.yaml
│   ├── model/lenet5.yaml
│   └── training/default.yaml
├── scripts/                  # CLI entry points
│   └── train.py              # Training script
├── data/                     # Data storage
│   ├── processed/hdf5/       # 598K images (2.1GB)
│   └── models/checkpoints/   # Timestamped runs
├── legacy/                   # Archived research code
│   ├── 1 spatter extraction/
│   └── 2 cnn/
└── tests/                    # Test suite
```

---

## 🎯 Key Features

### MLOps Infrastructure
- **Hydra Configuration**: Override any parameter from CLI
- **Device Agnostic**: Auto-detects and uses CPU/MPS/CUDA
- **Smart Checkpointing**: Saves top 3 models only
- **Early Stopping**: Configurable patience and min_delta
- **Reproducibility**: Config saved with each run

### Model Performance
- **Test Accuracy**: 98.38%
- **Class Balance**: 51.25% low / 48.75% high
- **Compression**: 18.4x (JPEG) + gzip on HDF5

### Training
```bash
# Default: 20 epochs, batch size 1024, Adam optimizer
uv run python scripts/train.py

# Quick test: fewer epochs, smaller batch
uv run python scripts/train.py training.epochs=5 training.batch_size=256

# Experiment: different learning rate
uv run python scripts/train.py training.optimizer.lr=0.0001
```

---

## 🔧 Configuration

All parameters configurable via Hydra:

**Model**: `config/model/lenet5.yaml`
- Architecture: in_channels, num_classes

**Training**: `config/training/default.yaml`
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Epochs: 20
- Batch size: 1024
- Early stopping: patience=3, min_delta=0.001
- Checkpointing: save_every=5 epochs, keep top 3

**Data**: `config/data/dataset.yaml`
- Image size: 32x32
- Grayscale: true
- Splits: 60/20/20 (train/val/test)

---

## 📚 Next Steps

**Phase 1 (Current)**: MLOps Foundation ✅
- Production pipeline with Hydra configs
- Device-agnostic training
- Checkpoint management
- Early stopping

**Phase 2**: Model Optimization
- HPO framework (Optuna/Ray Tune)
- Architecture experiments (ResNet, EfficientNet)
- MLflow/WandB tracking

**Phase 3**: Semi-Supervised Learning
- Label refinement for fine-grained spatter classification
- Active learning loop

---

**Status**: ✅ Production-ready MLOps pipeline

**Last Updated**: 2025-11-13
