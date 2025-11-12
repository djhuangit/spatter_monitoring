# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Vision & Objectives

**Primary Goal**: Transform this research prototype into a **production-grade MLOps pipeline** to showcase professional ML engineering skills.

**Secondary Goal**: Evolve from supervised binary classification to **semi-supervised learning** for label refinement, addressing the coarse labeling problem (current: binary 0/1 for gas flow, future: fine-grained spatter characteristics).

### Evolution Roadmap

1. **Phase 1: MLOps Foundation** (Current Priority)
   - Production-grade pipeline infrastructure
   - Experiment tracking and model versioning
   - Automated training and evaluation
   - Deployment capabilities

2. **Phase 2: Model Optimization**
   - Hyperparameter optimization (HPO) framework
   - Model architecture experiments (beyond LeNet5)
   - Performance benchmarking and comparison

3. **Phase 3: Label Refinement (Semi-Supervised Learning)**
   - Current labels are too coarse: binary (0=low gas flow, 1=high gas flow)
   - Goal: Learn fine-grained spatter characteristics (size, intensity, shape, trajectory)
   - Approach: Semi-supervised learning to leverage unlabeled spatter variations
   - Note: "Semi-supervised" is correct (uses both labeled and unlabeled data)

---

## System Architecture

### Two-Stage Pipeline

**Stage 1: Spatter Extraction** (`1 spatter extraction/`)
- Input: 3D printing video (640×360 @ 25 FPS)
- Algorithm: Frame differencing + morphological operations
- Output: HDF5 files with JPEG-encoded spatter crops + CSV frame logs
- Key script: `spatter_fvideo_hdf5_bytes_v3.py`

**Stage 2: CNN Classification** (`2 cnn/`)
- Input: 598,933 images from 6 HDF5 files (3 positions × 2 gas flows)
- Current model: LeNet5 (98.38% test accuracy)
- Output: Binary predictions (low/high gas flow)
- Key files: `spatter_dataset.py`, `lenet5.py`, training notebook

### Critical Design Patterns

**Lazy Loading**: Images decoded on-the-fly from HDF5, not pre-loaded into memory
```python
# In spatter_dataset.py:58-87
def __getitem__(self, idx):
    with h5py.File(file) as f:
        img_bytes = f[dataset][ds_idx]
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    return transform(img), label
```

**CSV Indexing**: Shuffled CSV maps dataset indices to `(hdf5_file, dataset_name, index, label)`
- `images.csv`: Shuffled for training (used)
- `unshuffled_images.csv`: Preserves HDF5 structure (debugging)

**Double Compression**: JPEG encoding (18.4x) + HDF5 gzip compression
- Reduces 600K images from ~38GB to 2.1GB (92% savings)
- No accuracy loss (98.38% achieved with compressed images)

---

## Development Environment

### Package Management
**Use `uv` for all Python operations** (per user preference)

```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run Python scripts
uv run python script.py

# Run Jupyter
uv run jupyter notebook
```

### Required Dependencies
**Core** (in `pyproject.toml`):
- `h5py>=3.15.1` - HDF5 file handling
- `opencv-python>=4.11.0.86` - Computer vision (frame differencing, JPEG codec)
- `numpy>=2.3.4` - Array operations
- `pandas>=2.3.3` - CSV indexing, data manipulation
- `matplotlib>=3.10.7` - Visualization

**Training** (need to add):
- `torch` - Deep learning framework
- `torchvision` - Image transforms
- `pillow` - Image format conversions
- `tqdm` - Progress bars (modern replacement for imutils)

**MLOps** (to be added in Phase 1):
- `mlflow` or `wandb` - Experiment tracking (choose one)
- `hydra-core` - Configuration management
- `pytest` - Testing framework
- `pytest-cov` - Test coverage
- `ruff` - Modern linter and formatter
- `mypy` - Type checking

**Optional** (Phase 2+):
- `optuna` - Hyperparameter optimization
- `ray[tune]` - Distributed HPO
- `tensorboard` - Alternative visualization
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

**Deprecated/Legacy**:
- `visdom` - Use wandb or tensorboard instead for training visualization
- `imutils` - Use tqdm for progress tracking

---

## Key Commands

### Data Extraction
```bash
# Extract spatters from video with capture enabled
uv run python "1 spatter extraction/spatter_fvideo_hdf5_bytes_v3.py" \
    video.mp4 \
    --startframe 0 \
    --capture \
    --minarea 2000

# With additional options (cropping, debug mode)
uv run python "1 spatter extraction/spatter_fvideo_hdf5_bytes_v3.py" \
    video.mp4 \
    --startframe 6000 \
    --leftbound 200 \
    --rightbound 500 \
    --minarea 2500 \
    --capture \
    --debug  # Only processes first 10K images
```

### Model Training (Current - Notebook-based)
```bash
# Start Jupyter and run training notebook
uv run jupyter notebook "2 cnn/train_lenet5_ModelTraining.ipynb"
```

### Data Inspection
```python
# Quick HDF5 inspection
uv run python -c "
import h5py
with h5py.File('2 cnn/data/left_low_0.h5') as f:
    print('Datasets:', list(f.keys())[:5])
    print('Total images:', sum(f[ds].shape[0] for ds in f.keys()))
"

# Load sample through dataset
uv run python -c "
from spatter_dataset import Spatter_gray
ds = Spatter_gray(['left_low_0.h5'], [0], resize=32, mode='train', root='2 cnn/data')
img, label = ds[0]
print(f'Image shape: {img.shape}, Label: {label}')
"
```

---

## Code Architecture Notes

### Extraction Pipeline (`spatter_fvideo_hdf5_bytes_v3.py`)

**Critical Configuration** (hardcoded - Phase 1 will externalize):
- `min_area = 2000` (line 288): Minimum spatter size in pixels
- `max_area = 15000` (line 163): Maximum size filter
- `dataset_size = 10000` (line 105): HDF5 chunk size
- `frames_to_next_layer = 700` (line 78): Layer detection threshold

**Frame Differencing Algorithm** (lines 146-151):
```python
diff = cv2.absdiff(frame1, frame2)      # Motion detection
gray = cv2.cvtColor(diff, COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0) # Noise reduction
thresh = cv2.threshold(blur, 20, 255)    # Binary threshold
dilated = cv2.dilate(thresh, None, 3)    # Fill gaps
contours = cv2.findContours(dilated)     # Extract spatters
```

**Three Filtering Criteria** (lines 162-166):
1. Size: `2000 ≤ area ≤ 15000` pixels
2. Blue rejection: `B_channel > (R + G)` to remove recoating light
3. Both must pass or spatter is rejected

**HDF5 Chunking Pattern** (lines 191-199):
- Accumulate 10K images in memory as JPEG bytes
- Write to HDF5 as single dataset with gzip
- Naming: `{video_name}_{counter}` (e.g., `video_0`, `video_1`)
- Remainder written after loop ends (lines 256-267)

### Dataset Implementation (`spatter_dataset.py`)

**Three Dataset Classes**:
1. `Spatter` (lines 29-147): RGB images
2. `Spatter_gray` (lines 149-267): **Grayscale (USED IN TRAINING)**
3. `Spatter_gray_ratio` (lines 269-384): Adjustable class balance

**Transform Pipeline** (lines 74-82):
```python
Resize(40, 40)           # 1.25x target (buffer for rotation)
RandomRotation(±15°)     # Data augmentation
CenterCrop(32, 32)       # Final size
ToTensor()               # → [0, 1] range
```
Why 1.25x? Rotation creates black corners; extra 25% ensures full 32×32 after crop.

**Grayscale → RGB Conversion** (training code):
- Dataset outputs: `[batch, 1, 32, 32]`
- Training converts: `torch.cat([x,x,x], dim=1)` → `[batch, 3, 32, 32]`
- Reason: LeNet5 expects 3-channel input

### Model Architecture (`lenet5.py`)

**Winner: Model 1 (No Conv Activation)** - 98.38% test accuracy
```python
Conv2d(3→6, 5×5) → MaxPool(2×2) →
Conv2d(6→16, 5×5) → MaxPool(2×2) →
Flatten(400) →
Linear(400→120, ReLU) → Linear(120→84, ReLU) → Linear(84→1)
```

**Alternative: Model 2 (With Activations)** - 93.52% test accuracy
- Uses Tanh after Conv, Sigmoid after AvgPool
- Classical LeNet5 approach
- Underperforms modern variant by 4.86%

**Binary Classification Design**:
- Single output neuron (logit)
- BCEWithLogitsLoss (combines sigmoid + BCE)
- Evaluation: `sigmoid(logit) > 0.5` → class 1, else class 0

---

## Current Production Gaps (Phase 1 Tasks)

### Critical Missing Components

1. **Configuration Management**
   - All hyperparameters hardcoded
   - Need: `config.yaml` + Hydra for experiment management

2. **Experiment Tracking**
   - No version control for models/datasets
   - Training metrics only in Visdom (ephemeral)
   - Need: MLflow or Weights & Biases integration

3. **Inference Pipeline**
   - Training exists, but no way to use trained model on new videos
   - Need: `inference.py` that chains extraction → preprocessing → prediction

4. **Error Handling & Logging**
   - No try/except blocks, only print statements
   - Need: Proper exception handling + Python logging module

5. **Testing**
   - Zero tests
   - Need: pytest suite for dataset, transforms, model forward pass

6. **Model Export**
   - Checkpoint is PyTorch state_dict only
   - Need: ONNX export for deployment

---

## MLOps Pipeline Architecture (Phase 1 Target)

```
CONFIG MANAGEMENT (Hydra) → config/*.yaml
        ↓
DATA PIPELINE → extraction.py, preprocessing.py, validation.py
        ↓
TRAINING PIPELINE → train.py + MLflow/WandB tracking
        ↓
EVALUATION & EXPORT → evaluate.py, export.py (ONNX)
        ↓
INFERENCE API → inference.py, FastAPI, Docker
```

---

## Phase 2: Model Experimentation Strategy

### Architecture Candidates
1. ResNet variants (ResNet18, ResNet34)
2. EfficientNet (B0, B1)
3. Vision Transformer (ViT)
4. Custom CNN

### HPO Framework
- **Tool**: Optuna or Ray Tune
- **Search Space**: Learning rate, batch size, architecture depth, augmentation
- **Tracking**: All experiments logged to MLflow/WandB
- **Analysis**: Pareto frontier (accuracy vs. efficiency)

---

## Phase 3: Semi-Supervised Learning for Label Refinement

### Problem Statement
**Current**: Binary labels (0=low gas, 1=high gas) are coarse
- Does not capture spatter characteristics: size, brightness, shape
- Hypothesis: Sub-clusters exist within each class

### Approach
**Semi-supervised** (not self-supervised) because:
- We have 599K labeled images (coarse) + can extract millions more (unlabeled)
- Semi-supervised uses both labeled and unlabeled data

**Methods to Consider**:
1. Pseudo-labeling with confidence filtering
2. Consistency regularization (FixMatch/MixMatch)
3. Contrastive learning + fine-tuning (SimCLR/MoCo)
4. Active learning for iterative label refinement

### Expected Outcomes
- Discover natural clusters within binary classes
- Refine to multi-class labels (e.g., 5-10 spatter types)
- Improve interpretability and process understanding

---

## Documentation Architecture

### For Humans
- **README.md**: Quick start
- **ANALYSIS.md**: Complete technical deep-dive (500+ lines)
- **DOCUMENTATION_INDEX.md**: Navigation guide

### For AI Agents
- **AGENT_CONTEXT.json**: Machine-readable specs with line numbers
- **analysis_report.json**: Metrics and statistics

### Key Statistics (Verified)
- Total: 598,933 images
- Class balance: 51.25% / 48.75%
- Storage: 2.1GB (18.4x JPEG compression)
- Test accuracy: 98.38%

---

## Important Constraints & Gotchas

### File Paths with Spaces
Folder names: `"1 spatter extraction/"`, `"2 cnn/"` - always quote in shell

### HDF5 Handling
Always use: `with h5py.File(...) as f:` - prevents corruption

### Image Decoding
Must decode JPEG bytes: `cv2.imdecode(np.frombuffer(bytes, np.uint8), -1)`

### Transform Order
Critical: Resize 1.25x BEFORE rotation (prevents black corners)

### Training
Batch size 1024 works on CPU (~99 min for 20 epochs) - no GPU required

### Grayscale Convention
Dataset outputs 1-channel, training replicates to 3-channel in loop

---

## Next Steps (Immediate Phase 1 Tasks)

1. **Modularize extraction** → `extraction/pipeline.py` with config
2. **Create training script** → `training/train.py` with MLflow
3. **Write inference pipeline** → `inference/predict.py`
4. **Add testing** → pytest suite
5. **Setup experiment tracking** → MLflow or WandB

**Suggested start**: Training script modularization (highest impact)


# support cpu, mps and cuda training