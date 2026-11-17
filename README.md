# Mechanical Parts Semantic Segmentation

## Overview

This project focuses on multi-class semantic segmentation of mechanical components using Deep Learning and Computer Vision.

The objective is to classify every pixel in an image of a cluttered mechanical parts tray into one of six component categories. The project was developed as part of a Kaggle competition.

## Problem Statement

Given RGB images of size 384 × 384, the model predicts pixel-level segmentation masks for six mechanical component classes.

### Target Classes

1. Hex Nut
2. Washer
3. Bolt
4. Ball Bearing
5. Spring
6. O-Ring

Background pixels are represented by class 0.

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Albumentations
- Pillow

## Model Architecture

A custom CNN-based encoder-decoder segmentation architecture inspired by U-Net was implemented.

Key components:

- Convolutional blocks with Batch Normalization and ReLU activation.
- Max Pooling for feature extraction and spatial downsampling.
- Transposed Convolutions for upsampling.
- Skip Connections to preserve spatial information.
- Pixel-wise classification into seven categories, including background.

## Methodology

1. Loaded and explored the training images and segmentation masks.
2. Prepared image and mask datasets.
3. Applied image normalization and preprocessing.
4. Implemented a custom CNN segmentation model.
5. Combined Dice Loss and Cross Entropy Loss for model optimization.
6. Trained the model and monitored validation loss and Mean Dice score.
7. Saved the best-performing model based on validation Dice.
8. Applied horizontal flip Test-Time Augmentation during inference.
9. Generated segmentation predictions for unseen test images.
10. Encoded predictions using Run-Length Encoding (RLE) to generate the Kaggle submission file.

## Loss Function

Combined Loss:

`Loss = 0.4 × CrossEntropyLoss + 0.6 × DiceLoss`

## Evaluation Metric

The competition uses the Dice Coefficient to measure the overlap between predicted segmentation masks and ground-truth masks.

## Dataset

The dataset was provided through the Kaggle competition and contains:

- 2,000 labelled training images.
- 500 unseen test images.
- Six mechanical component classes.

The dataset is not included in this repository. Access it through the original Kaggle competition.

## How to Run

1. Clone this repository.
2. Install the required dependencies.
3. Download the competition dataset from Kaggle.
4. Open the notebook in Kaggle or Jupyter Notebook.
5. Update dataset paths according to your environment.
6. Run the notebook cells to train the model and generate predictions.

## Project Highlights

- Implemented a custom CNN-based semantic segmentation model.
- Used a combination of Dice Loss and Cross Entropy Loss.
- Applied Test-Time Augmentation to improve prediction robustness.
- Generated pixel-level predictions for six mechanical component classes.
- Implemented Run-Length Encoding for Kaggle submission generation.

## Author

Ishya Bommireddy


