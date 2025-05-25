# Eigenfaces

This repository contains code and resources for a mini-project in the [Statistical signal and data processing](https://edu.epfl.ch/coursebook/en/statistical-signal-and-data-processing-through-applications-COM-500) through applications course.

## Features

- **Image Preprocessing:** Utilities for reshaping, normalizing, and preparing image datasets.
- **Feature Extraction:**  
  - PCA (Principal Component Analysis) for dimensionality reduction 
  - Phase congruency and other custom feature maps
- **Classification Pipelines:**  
  - Logistic Regression  
  - SVM (Support Vector Machine)  
  - Linear Regression (for regression tasks)  
  - Support for custom classifiers
- **Evaluation & Visualization:**  
  - Confusion matrix, F1 score, and accuracy metrics  
  - Visualization of PCA eigenvalues and feature maps  
  - Customizable plotting with adjustable font sizes

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [Pillow](https://python-pillow.org/)
- numpy

Install dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib facenet-pytorch pillow numpy
```

### Example Usage

**PCA Feature Extraction:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=100, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

**Classification:**
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
```

## Folder Structure

```
.
├── images.mat/                # Yale Face Database images
├── classes.mat/                # Yale image labels
├── generated_images.mat/                # Synthetic dataset images
├── generated_classes.mat/                # Synthetic dataset labels
├── Generate_data.ipynb/                # A script for generating synthetic dataset of random images
├── generate_results.ipynb/                # A script to generate classification results for the report
├── PCA.ipynb/                # Implementation of the PCA based classification
├── Fisherfaces.ipynb/                # Implementation of the Fisherfaces based classification
├── Gabor-Fisher.ipynb/                # Implementation of the Gabor filter based classification
├── README.md
```