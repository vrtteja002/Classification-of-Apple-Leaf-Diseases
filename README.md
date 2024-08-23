# Apple Disease Classifier

## Overview
This project implements a deep learning model to classify various diseases in apple leaves. It uses a Convolutional Neural Network (CNN) and a pre-trained ResNet50 model to achieve high accuracy in disease classification.

## Dataset
The dataset used in this project contains images of apple leaves with four different classes:
1. Apple Scab
2. Black Rot
3. Cedar Apple Rust
4. Healthy

## Models
Two models are implemented and compared:

1. Custom CNN
2. Pre-trained ResNet50 (Transfer Learning)

## Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- numpy

## Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/apple-disease-classifier.git
   cd apple-disease-classifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your training images in `datasets/train/`
   - Place your testing images in `datasets/test/`
   - Ensure that images are organized in subdirectories according to their classes

## Usage
1. To train and evaluate the custom CNN model:
   ```
   python custom_cnn.py
   ```

2. To train and evaluate the pre-trained ResNet50 model:
   ```
   python pretrained_resnet50.py
   ```

## Results
Both models achieve high accuracy in classifying apple leaf diseases:

- Custom CNN: ~98% accuracy on the test set
- Pre-trained ResNet50: ~98.5% accuracy on the test set

Detailed classification reports and confusion matrices are provided in the output of each script.

## Future Work
- Implement data augmentation techniques to improve model generalization
- Explore other pre-trained models for comparison
- Deploy the model as a web application or mobile app for real-time disease detection
