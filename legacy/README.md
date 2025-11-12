# Legacy Code Archive

This directory contains the original research code from the initial spatter monitoring project.

## Contents

- **1 spatter extraction/** - Original video processing pipeline
  - `spatter_fvideo_hdf5_bytes_v3.py` - Frame differencing and HDF5 storage
  - Video files and configuration

- **2 cnn/** - Original training pipeline
  - `spatter_dataset.py` - PyTorch Dataset implementation
  - `lenet5.py` - Winner model (98.38% accuracy)
  - `lenet5_withactivation.py` - Alternative model (93.52% accuracy)
  - `train_lenet5_ModelTraining.ipynb` - Training notebook
  - `data/` - HDF5 files and CSV indices (copied to `/data/processed/hdf5/`)

## Migration Status

This code has been **archived** and is kept for reference only. The production-ready refactored version is in:

- **Extraction**: `src/spatter/extraction/`
- **Dataset**: `src/spatter/data/`
- **Models**: `src/spatter/models/`
- **Training**: `src/spatter/training/`

## Usage

To run the legacy code (not recommended for production):

```bash
# Extraction (legacy)
uv run python "legacy/1 spatter extraction/spatter_fvideo_hdf5_bytes_v3.py" video.mp4 --capture

# Training (legacy)
uv run jupyter notebook "legacy/2 cnn/train_lenet5_ModelTraining.ipynb"
```

## Why Archived?

The legacy code was a research prototype with:
- Hardcoded parameters
- No configuration management
- No testing infrastructure
- Notebook-based training
- No experiment tracking
- No deployment capabilities

The new production-grade pipeline addresses all these gaps while maintaining the same proven algorithms.

---

**See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the new structure.**
