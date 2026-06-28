# Breast Tumor Classification Using Deep Learning

This repository contains a PyTorch-based machine learning pipeline designed to classify histopathology/medical images of breast tumors into two categories: **Benign** and **Malignant**. The project implements, trains, and evaluates both a custom Convolutional Neural Network (CNN) architecture and a transfer learning approach using a pre-trained ResNet18 model.

## Dataset
The project utilizes a subset of the **Multi Cancer Dataset** originally available on Kaggle.
* **Task:** Binary classification (`breast_benign` vs. `breast_malignant`).
* **Pipeline Setup:** Automates dataset downloading via `kagglehub`, creates a local working subset directory, and splits images into specialized training and validation splits (`train/` and `val/`).

## Pipeline Features
* **Automated Data Fetching:** Directly downloads dataset files into a cache path using Kaggle's API tools.
* **Data Preprocessing & Augmentation:** Implements target transformations using `torchvision.transforms`, including:
  * Image resizing to $224 \times 224$ pixels
  * Random rotations up to 10 degrees
  * Horizontal flips and scale-based random resized crops to prevent model overfitting
  * Pixel tensor normalization
* **Custom CNN Architecture (`SimpleCNN`):** A custom neural net containing two 2D convolutional layers, ReLU activations, max pooling layers, a fully connected dense layer with dropout regularization (0.5), and a final linear layer outputting class logits.
* **Transfer Learning Implementation:** Incorporates a pre-trained `ResNet18` model, modifying its final fully-connected (`fc`) classification head to map to the target subset classes.

## Getting Started

### Prerequisites
Ensure you have the required packages installed:
```bash
pip install torch torchvision kagglehub transformers timm scikit-learn numpy matplotlib
