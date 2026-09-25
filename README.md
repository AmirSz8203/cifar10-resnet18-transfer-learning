# CIFAR-10 Image Classification with ResNet-18

A PyTorch image-classification project using **transfer learning**, **fine-tuning**, and **Optuna hyperparameter optimization** to classify CIFAR-10 images into 10 classes.

> **Reported test accuracy: 93.10%**
> This result is taken from the training output included with this project and has not been independently reproduced.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Method](#method)
- [Hyperparameter Search](#hyperparameter-search)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Limitations and Notes](#limitations-and-notes)

## Project Overview

This project adapts an ImageNet-pretrained ResNet-18 model to CIFAR-10. The convolutional backbone is initially frozen, while the final residual block (`layer4`) and the classification head are fine-tuned.

The workflow is split into two notebooks:

1. **Hyperparameter tuning** — Optuna searches for suitable learning rates, weight decay, and dropout.
2. **Training and evaluation** — The selected hyperparameters are used to fine-tune the model, with validation-based learning-rate scheduling, early stopping, checkpoint saving, and test-set evaluation.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32×32 color images across 10 classes:

- 50,000 training images
- 10,000 test images

The training split is further divided into 45,000 training images and 5,000 validation images using a stratified 90/10 split.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Repository Structure

```text
cifar10-resnet18-transfer-learning/
├── notebooks/
│   ├── 01_hyperparameter_tuning.ipynb
│   └── 02_model_training_evaluation.ipynb
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── README.md
└── requirements.txt
```

## Method

| Component | Details |
|---|---|
| Model | ResNet-18 pretrained on ImageNet |
| Transfer learning | Backbone frozen initially; `layer4` and classification head unfrozen for training |
| Classifier | Dropout followed by a 10-class linear layer |
| Augmentation | Random horizontal flip, random crop, color jitter |
| Optimization | Adam, with separate learning rates for `layer4` and the classification head |
| Loss | Cross-entropy |
| Hyperparameter search | Optuna — 15 trials, up to 4 epochs per trial |
| Training controls | Mixed-precision training, `ReduceLROnPlateau`, early stopping |
| Evaluation | Test accuracy, per-class accuracy, classification report, confusion matrix |

## Hyperparameter Search

The best trial reported a validation loss of **0.19671**.

| Hyperparameter | Selected value |
|---|---|
| `layer4_lr` | 8.95768e-05 |
| `fc_lr` | 1.00073e-04 |
| `weight_decay` | 6.81943e-05 |
| `dropout_p` | 0.48994 |

These values were selected by the Optuna search in the first notebook.

## Results

### Overall performance

| Metric | Result |
|---|---|
| Test accuracy | **93.10%** |
| Best reported validation accuracy (epoch 10) | 94.30% |
| Best reported validation loss (hyperparameter search) | 0.19671 |

The selected checkpoint was based on validation loss, and the reported test accuracy was measured using that checkpoint.

### Classification report

```text
              precision    recall  f1-score   support

    airplane       0.90      0.98      0.94      1000
  automobile       0.95      0.97      0.96      1000
        bird       0.93      0.91      0.92      1000
         cat       0.88      0.84      0.86      1000
        deer       0.94      0.91      0.93      1000
         dog       0.88      0.88      0.88      1000
        frog       0.94      0.97      0.95      1000
       horse       0.93      0.96      0.94      1000
        ship       0.97      0.95      0.96      1000
       truck       0.98      0.94      0.96      1000

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000
```

### Per-class test accuracy

| Class | Accuracy |
|---|---|
| Automobile | 97.50% |
| Airplane | 97.70% |
| Frog | 96.90% |
| Horse | 95.70% |
| Ship | 95.10% |
| Truck | 94.20% |
| Bird | 91.30% |
| Deer | 91.10% |
| Dog | 87.80% |
| Cat | 83.70% |

The lowest per-class accuracy in the reported results was for **cat (83.70%)**, consistent with its lower precision/recall in the classification report above.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AmirSz8203/cifar10-resnet18-transfer-learning.git
cd cifar10-resnet18-transfer-learning
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The project was trained using PyTorch with CUDA 12.8 support. A plain
> `pip install -r requirements.txt` may install a CPU-only build of PyTorch. For
> GPU acceleration, install `torch` and `torchvision` first using the command
> for your system from the [official PyTorch site](https://pytorch.org/get-started/locally/),
> then run `pip install -r requirements.txt` for the remaining dependencies.

### 4. Prepare the dataset

The original notebooks use a Kaggle-specific dataset path. To run them outside Kaggle, download CIFAR-10 from the [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html), extract it, and update the dataset root in the notebooks.

The model uses pretrained ResNet-18 weights, which `torchvision` may download automatically on first run.

### 5. Run the notebooks

Run the notebooks in order:

1. `notebooks/01_hyperparameter_tuning.ipynb`
2. `notebooks/02_model_training_evaluation.ipynb`

A CUDA-capable GPU is recommended for faster training; CPU execution may be considerably slower.

## Dependencies

- PyTorch
- torchvision
- scikit-learn
- Optuna
- matplotlib
- seaborn
- NumPy
- Jupyter

See `requirements.txt` for the full dependency list.

## Limitations and Notes

- Reported metrics are based on notebook outputs and have not been independently reproduced.
- The notebooks originally use a Kaggle-specific dataset path; update it for local execution.
- Pretrained ResNet-18 weights must be downloaded on first use unless already cached.
- Results may vary with hardware, library versions, and random behavior.
- The model is evaluated on CIFAR-10 and is not intended to classify arbitrary real-world images without further adaptation.
