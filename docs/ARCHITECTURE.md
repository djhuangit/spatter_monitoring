# Project Architecture

## Directory Structure

```
spatter_monitoring/
│
├── src/                              # All source code
│   ├── spatter/                      # Main package
│   │   ├── __init__.py
│   │   │
│   │   ├── extraction/               # Phase 1: Video → HDF5 pipeline
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py           # Main extraction orchestrator
│   │   │   ├── frame_processor.py    # Frame differencing algorithm
│   │   │   ├── filters.py            # Size/color filtering
│   │   │   ├── storage.py            # HDF5 writer with chunking
│   │   │   └── video_reader.py       # Video I/O wrapper
│   │   │
│   │   ├── data/                     # Dataset & data loading
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py            # PyTorch Dataset (lazy loading)
│   │   │   ├── transforms.py         # Image transforms
│   │   │   ├── datamodule.py         # Lightning DataModule (Phase 1)
│   │   │   └── validation.py         # Data quality checks
│   │   │
│   │   ├── models/                   # Neural network architectures
│   │   │   ├── __init__.py
│   │   │   ├── lenet5.py             # Current winner
│   │   │   ├── base.py               # Abstract base model
│   │   │   ├── resnet.py             # Phase 2: ResNet variants
│   │   │   ├── efficientnet.py       # Phase 2: EfficientNet
│   │   │   └── registry.py           # Model factory
│   │   │
│   │   ├── training/                 # Training infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py            # Main training loop
│   │   │   ├── callbacks.py          # Custom callbacks
│   │   │   ├── metrics.py            # Evaluation metrics
│   │   │   └── losses.py             # Loss functions
│   │   │
│   │   ├── inference/                # Production inference
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py          # End-to-end pipeline
│   │   │   ├── onnx_runtime.py       # ONNX inference
│   │   │   └── batch_predictor.py    # Batch processing
│   │   │
│   │   ├── hpo/                      # Phase 2: Hyperparameter optimization
│   │   │   ├── __init__.py
│   │   │   ├── optuna_search.py      # Optuna integration
│   │   │   ├── ray_search.py         # Ray Tune integration
│   │   │   └── search_spaces.py      # HPO configurations
│   │   │
│   │   ├── semisupervised/           # Phase 3: Label refinement
│   │   │   ├── __init__.py
│   │   │   ├── pseudolabel.py        # Pseudo-labeling
│   │   │   ├── consistency.py        # Consistency regularization
│   │   │   ├── contrastive.py        # Contrastive learning
│   │   │   └── active_learning.py    # Active learning loop
│   │   │
│   │   └── utils/                    # Shared utilities
│   │       ├── __init__.py
│   │       ├── config.py             # Configuration helpers
│   │       ├── logging.py            # Logging setup
│   │       ├── io.py                 # File I/O utilities
│   │       └── visualization.py      # Plotting helpers
│   │
│   └── api/                          # REST API (Phase 1)
│       ├── __init__.py
│       ├── main.py                   # FastAPI app
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── predict.py            # Prediction endpoint
│       │   └── health.py             # Health checks
│       └── schemas.py                # Pydantic models
│
├── config/                           # Hydra configurations
│   ├── config.yaml                   # Base config
│   ├── model/
│   │   ├── lenet5.yaml
│   │   ├── resnet18.yaml
│   │   └── efficientnet_b0.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── hpo.yaml
│   ├── data/
│   │   ├── extraction.yaml
│   │   └── dataset.yaml
│   └── experiment/                   # Experiment configs
│       ├── baseline.yaml
│       └── hpo_sweep.yaml
│
├── scripts/                          # CLI entry points
│   ├── extract_spatters.py           # Video → HDF5
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Model evaluation
│   ├── export_onnx.py                # Model export
│   ├── predict.py                    # Inference on new data
│   └── run_hpo.py                    # HPO sweep
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_extraction.py
│   │   ├── test_dataset.py
│   │   ├── test_models.py
│   │   └── test_transforms.py
│   ├── integration/
│   │   ├── test_training_pipeline.py
│   │   └── test_inference_pipeline.py
│   └── conftest.py                   # Pytest fixtures
│
├── notebooks/                        # Exploratory notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debugging.ipynb
│   └── 03_results_analysis.ipynb
│
├── data/                             # Data storage
│   ├── raw/                          # Original videos
│   │   └── videos/
│   ├── processed/                    # HDF5 files
│   │   └── hdf5/
│   ├── models/                       # Saved models
│   │   └── checkpoints/
│   └── predictions/                  # Inference outputs
│
├── legacy/                           # Original research code (archived)
│   ├── 1 spatter extraction/
│   └── 2 cnn/
│
├── docs/                             # Documentation
│   ├── ANALYSIS.md                   # Technical deep-dive
│   ├── ARCHITECTURE.md               # This file
│   ├── API.md                        # API documentation
│   └── DEPLOYMENT.md                 # Deployment guide
│
├── .github/                          # CI/CD
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── docker/                           # Docker configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── pyproject.toml                    # Project metadata
├── uv.lock                           # Lock file
├── .gitignore
├── README.md
├── CLAUDE.md                         # AI agent guidance
└── AGENT_CONTEXT.json                # Machine-readable specs
```

## Design Principles

### 1. Separation of Concerns
- **Extraction**: Isolated video processing logic
- **Data**: Dataset and loading only
- **Models**: Pure architecture definitions
- **Training**: Training infrastructure separate from models
- **Inference**: Production-ready prediction pipeline

### 2. Phase-Aligned Structure
- **Phase 1** (Current): `extraction/`, `data/`, `models/`, `training/`, `inference/`
- **Phase 2** (HPO): `hpo/` module, extended `models/`
- **Phase 3** (Semi-supervised): `semisupervised/` module

### 3. Configuration-Driven
- All parameters in `config/` (Hydra)
- No hardcoded values in source code
- Easy experiment tracking via config versioning

### 4. Testable
- Each module has corresponding tests
- Unit tests for components
- Integration tests for pipelines

### 5. Production-Ready
- API module for deployment
- ONNX export for serving
- Docker support
- Logging and monitoring

---

## Migration Strategy

### Step 1: Create Structure (No Code Changes)
```bash
# Create all directories
mkdir -p src/spatter/{extraction,data,models,training,inference,hpo,semisupervised,utils}
mkdir -p src/api/routes
mkdir -p config/{model,training,data,experiment}
mkdir -p scripts tests/{unit,integration} notebooks
mkdir -p data/{raw/videos,processed/hdf5,models/checkpoints,predictions}
mkdir -p legacy docs docker .github/workflows

# Create __init__.py files
touch src/spatter/__init__.py
touch src/spatter/{extraction,data,models,training,inference,hpo,semisupervised,utils}/__init__.py
touch src/api/__init__.py src/api/routes/__init__.py
touch tests/__init__.py
```

### Step 2: Archive Legacy Code
```bash
# Move old code to legacy/
mv "1 spatter extraction" legacy/
mv "2 cnn" legacy/
```

### Step 3: Migrate Core Components
1. **Extraction**: Refactor `spatter_fvideo_hdf5_bytes_v3.py` → `src/spatter/extraction/`
2. **Dataset**: Migrate `spatter_dataset.py` → `src/spatter/data/dataset.py`
3. **Models**: Move `lenet5.py` → `src/spatter/models/lenet5.py`
4. **Data**: Copy HDF5 files → `data/processed/hdf5/`

### Step 4: Create Entry Points
1. `scripts/extract_spatters.py` - CLI wrapper for extraction
2. `scripts/train.py` - Training script with MLflow
3. `scripts/predict.py` - Inference script

### Step 5: Add Configuration
1. Create `config/config.yaml` with base settings
2. Create model-specific configs
3. Add experiment tracking configs

---

## Package Installation

After migration, install as editable package:

```bash
# Update pyproject.toml to include src/
uv pip install -e .
```

Then import anywhere:
```python
from spatter.extraction import pipeline
from spatter.data import dataset
from spatter.models import lenet5
```

---

## Benefits of This Structure

### Developer Experience
- **Imports**: Clean `from spatter.models import LeNet5`
- **Testing**: Easy to test isolated components
- **IDE Support**: Proper autocomplete and navigation

### MLOps
- **Experiment Tracking**: All configs versioned in `config/`
- **Reproducibility**: Lock dependencies, config, and code
- **Deployment**: `src/api/` ready for Docker + K8s

### Scalability
- **Phase 2**: Add new models in `models/`, HPO in `hpo/`
- **Phase 3**: Semi-supervised in dedicated module
- **New Features**: Clear place for everything

### Collaboration
- **Onboarding**: Clear structure for new developers
- **Documentation**: Each module has purpose
- **Code Review**: Isolated changes easy to review

---

## Next Steps

1. **Create directory structure** (Step 1)
2. **Archive legacy code** (Step 2)
3. **Migrate extraction pipeline** (Step 3.1)
4. **Migrate dataset** (Step 3.2)
5. **Create training script** (Step 4.2)
6. **Add base config** (Step 5.1)

Would you like me to proceed with creating this structure?
