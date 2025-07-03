# CLIP-based Image Retrieval System

A complete pipeline for training and evaluating a CLIP-based image retrieval system using supervised contrastive learning.

## Features

- CLIP vision encoder fine-tuning with supervised contrastive learning
- Balanced batch sampling for effective training
- Automatic mixed precision training
- Comprehensive evaluation with top-k retrieval results
- Model checkpointing and result saving

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install torch torchvision transformers scikit-learn pillow tqdm numpy
```

**Or create a requirements.txt file:**
```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
scikit-learn>=1.3.0
pillow>=9.0.0
tqdm>=4.65.0
numpy>=1.24.0
```

Then install:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
project/
├── train/                    # Training data (class folders)
│   ├── class1/
│   ├── class2/
│   └── ...
├── test-2/
│   ├── gallery/             # Gallery images for retrieval
│   └── query/               # Query images
├── results/                 # Output retrieval results (JSON)
├── weights/                 # Model checkpoints
└── script.py
```

## Usage

### Basic Training

```python
python script.py
```

### Custom Configuration

```python
config = {
    'model_name': "openai/clip-vit-large-patch14-336",
    'num_epochs': 5,
    'learning_rate': 1e-7,
    'train_dir': "path/to/train",
    'gallery_dir': "path/to/gallery",
    'query_dir': "path/to/query",
    'top_k': 10
}

trainer = ImageRetrievalTrainer(config)
trainer.train()
```

## Key Components

### CLIPEncoder
- Wrapper around CLIP vision model
- L2-normalized embeddings for similarity computation

### SupConLoss
- Supervised contrastive loss implementation
- Temperature-scaled similarity computation

### BalancedBatchSampler
- Ensures equal samples per class in each batch
- Configurable classes and samples per batch

### Training Pipeline
- Mixed precision training with gradient scaling
- Automatic evaluation after each epoch
- Model checkpointing and result saving

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "openai/clip-vit-large-patch14-336" | CLIP model variant |
| `temperature` | 0.07 | Contrastive loss temperature |
| `num_epochs` | 1 | Number of training epochs |
| `learning_rate` | 1e-7 | Learning rate for optimizer |
| `n_classes_per_batch` | 4 | Classes per training batch |
| `n_samples_per_class` | 2 | Samples per class per batch |
| `top_k` | 10 | Number of top retrievals to return |

## Output Files

- `results/retrieval_results_epoch_XX.json`: Top-k retrieval results per epoch
- `weights/encoder_epoch_XX.pth`: Model checkpoints with optimizer state

## Performance Features

- CUDA acceleration with automatic device detection
- Mixed precision training for memory efficiency
- Persistent workers for faster data loading
- Batch processing for evaluation

## Training Data Format

Training images should be organized in class folders:
```
train/
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
```

Gallery and query images should be in flat directories without subfolders.
