# IDL SS26 Final Assignment — Operation Cyber-Histology
| NAME | MATRICULATION NUMBER |
|---|---|
| Chandana Lakshmisagara Udayashankar | 10013715 |
| Sahana Mysore Venkatesh | 10013467|

---

Reconstructed and repaired ML pipeline for multi-class medical image classification across four MedMNIST-derived datasets (cells, chest, lesions, orgs) using three CNN architectures (AlexNet, VGG16, ResNet18).

## Repository Structure

```
Code/
├── data.py                  # Dataset loading, train/val/test split
├── models.py                # AlexNet, VGG16, ResNet18 architectures
├── fit.py                   # Trainer class: training loop, validation, best-checkpoint tracking
├── train.py                 # Main training script — loops over all dataset/model combinations
├── evaluate.py              # Test-set evaluation — computes accuracy, precision, recall, F1
├── config.json              # All hyperparameters and dataset/model run configurations
├── AUDIT_LOG.md             # Full bug audit: symptoms, root causes, fixes, commit hashes
├── REPORT.md                # Benchmark results, analysis, architectural recommendations
├── checkpoints/             # Saved model weights per run (gitignored, generated locally)
├── logs/                    # Per-run training history (JSON, loss/accuracy per epoch)
└── evaluation_results.json  # Final test-set metrics for all 12 combinations
```


## Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- The MedMNIST-derived dataset files (cells.pt, chest.pt, lesions.pt, orgs.pt) — download from the backup link provided in the assignment PDF

## Installation

```bash
# Clone the repository
git clone https://github.com/chandanaluday/idl26-finalassignment-chandana_sahana.git
cd idl26-finalassignment-chandana_sahana/Code

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib
```

## Configuration

Edit config.json to set your local data path and hyperparameters:

```json
{
  "DATA_PATH": "C:/Users/Lenovo/Downloads/DEEP_LEARNING_FINAL_PORTFOLIO/idl26-finalassignment-chandana_sahana/data",
  "BATCH_SIZE": 64,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 10,
  "DROP_RATE": 0.5,
  "ACTIVATION": "ReLU",
  "RUNS": [
    { "DATA": "cells", "MODEL": "AlexNet", "CHANNELS": 3, "NUM_CLASSES": 8 }
  ]
}
```

DATA_PATH must point to the folder containing your downloaded .pt dataset files. The RUN list can contain any combination of the four datasets (cells, chest, lesions, orgs) and three models (AlexNet, VGG16, ResNet18).

## Running Training

```bash
python3 train.py
```

This trains every dataset/model combination listed in config.json's RUNS array. For each combination, it:
- Trains for the configured number of epochs, tracking the best validation-accuracy checkpoint
- Saves the best model weights to checkpoints/{dataset}_{model}_epoch{N}.pt
- Saves per-epoch training history to logs/{dataset}_{model}_epoch{N}_history.json

Training automatically uses a GPU if available: NVIDIA (CUDA), Apple Silicon (MPS), or falls back to CPU.

## Running Evaluation

Once training is complete and checkpoints exist:

```bash
python3 evaluate.py
```

This loads each saved checkpoint, evaluates it on the held-out test set, and computes accuracy, macro-precision, macro-recall, and macro-F1-score for every dataset/model combination. Results are printed to the console with pass/fail status against the assignment's minimum accuracy targets, and saved to evaluation_results.json.

## Documentation

- See AUDIT_LOG.md for the full list of bugs identified in the recovered codebase, their root causes, fixes, and corresponding commit hashes.
- See REPORT.md for the complete benchmark results table, per-dataset analysis, and architectural recommendations.
