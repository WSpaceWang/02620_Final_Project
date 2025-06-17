# Manual and Scikit-learn SVM for Retinal Vessel Segmentation

This project implements and compares two Support Vector Machine (SVM) classifiers for retinal blood vessel segmentation based on RGB image data from the DRIVE dataset.

## Contents

- `SVM - Enhao He.py`: Manual implementation of SVM with RBF kernel and Platt scaling for probability calibration.
- `SVM_sklearn - Enhao He.py`: Implementation using `sklearn.svm.SVC` with class balancing and built-in probability estimation.

## Dataset

The code uses the DRIVE dataset, specifically:
- RGB images from `datasets/training/images`
- Ground truth labels from `datasets/training/1st_manual`
- Binary masks from `datasets/training/mask`

**Note**: A custom `load_dataset` function is used to load and preprocess the data. Ensure the `functions.py` script is available and contains this loader.

## Preprocessing

1. RGB pixels are extracted from regions of interest (non-black mask).
2. Ground truth labels are binarized: `1` for vessel, `0` for background.
3. Data is subsampled for speed and memory efficiency.

## Manual SVM Highlights

- RBF kernel manually implemented.
- Quadratic programming solved using `cvxopt`.
- Platt scaling added using `LogisticRegression` to estimate probabilities.
- Class imbalance is handled by adjusting the penalty `C` based on label frequency.

## Scikit-learn SVM Highlights

- Uses `SVC(kernel='rbf', class_weight='balanced')` for automatic handling of class imbalance.
- Enables probability output via `probability=True`.

## Evaluation

Both scripts print:
- Classification report (precision, recall, F1-score)
- AUC score

## Requirements

- Python 3.6+
- numpy
- scikit-learn
- tqdm
- cvxopt (only for the manual version)

Install requirements using:

```bash
pip install numpy scikit-learn tqdm cvxopt

Run the scripts by commands:

python "SVM - Enhao He.py"
python "SVM_sklearn - Enhao He.py"
