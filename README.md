# MNIST Handwritten Digit Classification with Vision Transformers (ViT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ananya10-Coder/Image-Classification/blob/main/ImageClassification.ipynb)

This project demonstrates how to fine-tune a pre-trained Vision Transformer (ViT) model for image classification on the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). The project includes data preprocessing, model fine-tuning, evaluation, and prediction on a single image.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

The goal of this project is to classify handwritten digits from the MNIST dataset using a Vision Transformer (ViT) model. Vision Transformers are a state-of-the-art architecture for image classification tasks, leveraging the power of transformer models originally designed for natural language processing. This project fine-tunes a pre-trained ViT model on a small subset of the MNIST dataset and evaluates its performance.

Key steps in the project:
1. **Data Preprocessing**: Resize and normalize the MNIST images to match the input requirements of the ViT model.
2. **Model Fine-Tuning**: Fine-tune a pre-trained ViT model on the MNIST dataset.
3. **Evaluation**: Evaluate the model's accuracy on a test set.
4. **Prediction**: Make predictions on a single image.

## Requirements

To run this project, you need the following:

- Python 3.x
- `torch` (PyTorch)
- `torchvision`
- `transformers` (Hugging Face Transformers library)
- `datasets` (Hugging Face Datasets library)
- `evaluate` (Hugging Face Evaluate library)
- A CUDA-capable GPU (optional but recommended for faster training)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mnist-vit-classification.git
   cd mnist-vit-classification
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install torch torchvision transformers datasets evaluate
   ```

## Usage

1. **Run the script:**
   ```bash
   python mnist_vit_classification.py
   ```

2. **Steps performed by the script:**
   - Load a small subset of the MNIST dataset.
   - Preprocess the data (resize, normalize, and convert to tensor format).
   - Load a pre-trained Vision Transformer (ViT) model and fine-tune it on the MNIST dataset.
   - Evaluate the model's accuracy on a test set.
   - Make predictions on a single image from the test set.

3. **Output:**
   - Training loss for each epoch.
   - Accuracy on the test set.
   - Predicted and actual labels for a sample image.

## Results

After running the script, you will see output similar to the following:
```
Epoch 1/3, Loss: 1.2345
Epoch 2/3, Loss: 0.5678
Epoch 3/3, Loss: 0.3456
Evaluating on the test set...
Accuracy: 95.00%
Predicted Label: 7
Actual Label: 7
```

This indicates the model's performance on the test set and its ability to correctly classify a sample image.

## License

MIT License.
