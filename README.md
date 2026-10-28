# ENEL707 Facial Feature Classification Challenge
Kaggle Competition Repository

Author: Dev Mehta
Kaggle username: devmehta111
Public score: 0.9791

https://www.kaggle.com/competitions/enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch/data

## Contents
- Task_1_work.ipynb  Dataset exploration
- Task_2_work.ipynb  Preprocessing, 64x64 cache, train/val split
- Task_3_work.ipynb  Writes pipeline.py and model_defs.py, speed test
- Task_4_work.ipynb  Trains ResNet18 and custom CNN, writes results.csv
- Task_5_work.ipynb  Ensemble, evaluation, submission.csv
- pipeline.py, model_defs.py  Shared modules (created by Task 3)
- submission.csv     Final Kaggle submission
- results.csv        Validation results per model

## Requirements
Python 3.x, CPU only:
pip install torch torchvision scikit-learn pandas numpy matplotlib pillow

## Data
Place the competition files in ENEL707-Kaggle-Comp/Datasets/
(train/, test/, train_labels.csv, sample_submission.csv). No external data used.

## How to run
Run the notebooks in order, Task 1 to Task 5, restarting the kernel between
notebooks. Each notebook is self-contained, and Task 3 must run before Task 4.
Approximate runtimes on an i5-1235U, 16 GB RAM:
caching ~23 min, ResNet18 ~53 min, custom CNN ~60 min.

## Reproducibility
Seed 42 everywhere (split, torch). Run cells in order. Exact results may differ
slightly across hardware and library versions.
