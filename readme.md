# CNN From Scratch Using NumPy: Manual Forward and Backpropagation
### Overview

This project implements a complete Convolutional Neural Network (CNN) from scratch using only NumPy, without relying on deep learning frameworks such as TensorFlow or PyTorch.

The goal of this project was to gain a deep understanding of how convolutional neural networks operate internally by manually implementing forward propagation, backpropagation, weight updates, gradient computation, and image reconstruction tasks.

## Features

- 7-layers CNN architecture
- Manual convolution implementation
- Full convolution for backpropagation
- Correlation-based weight gradient computation
- He weight initialization
- Custom activation functions
- Binary classification output layer using Sigmoid
- Gradient clipping
- Checkpoint saving and loading
- Image reconstruction experiments


## Architecture
![Architecture](files/architecture.png)

## Feature Maps

![EPOCH 4970](files\test_result\7pixels\result_track\pptx/4970.png)
![EPOCH 5017](files\test_result\7pixels\result_track\pptx/5017.png)

## Implemented Components
### Forward Propagation
Implemented manually:

- Convolution
- Padding
- Bias addition
- Activation functions

### Backpropagation

Implemented manually:

- dL/dW
- dL/dA
- dL/dZ
- Full convolution
- Correlation-based kernel gradients

### Training Utilities
- Checkpointing
- Gradient clipping
- Custom learning rate control
- Weight initialization

## Training Resilts 


## Technical Challenges Solved

During development several deep learning challenges were encountered and investigated:

- Exploding gradients
- Vanishing gradients
- Activation saturation
- Numerical instability (NaN values)
- Gradient flow debugging
- Weight initialization effects
- Training stability


## Technologies Used
- Python
- NumPy
- OpenCV
- Matplotlib

## Educational Goal

This project was built primarily as a learning exercise to understand the mathematical and computational foundations of convolutional neural networks without relying on high-level deep learning libraries.