# Neural Network from Scratch - MNIST Digit Classifier

## Overview
This project executes a fully connected neural network from scratch using Python and NumPy.
The goal of this project was to understand the maths and internal processes behind neural networks by building each of the parts manually rather than using libraries like tensorflow.

## Dataset
The model uses the MNIST handwritten digit dataset.
Each example contains:
- One label representing 0-9 
- 784 pixel values representing a 28x28 image
Each image is flattened into a vector of 784 pixel values and normalised to the range 0–1. Normalisation improves training stability by keeping the input values on a consistent scale, making gradient descent more effective.

The network consists of:
- An input layer of 784 neurons
- A hidden layer of 64 neurons using ReLU activation
- An output layer of 10 neurons using Softmax activation

The parameters are:
- W1: (64, 784)
- b1: (64, 1)
- W2: (10, 64)
- b2: (10, 1)

## Training Process
The network is trained using gradient descent. 
Each training iteration follows these steps:
- Forward Propagation:
The input data passes through the network to generate the prediction
- Error calculation:
The prediction is compared to its actual label
- Back propagation:
The gradients of the weights and biases are calculated
- Parameter Updates:
The weights and biases are adjusted to reduce further errors
(Parameter = Parameter - learning_rate * gradient)

## Results
The trained model achieves approximately 90% training accuracy and 89–90% accuracy on the development set.

# Features
- Fully connected neural network implemented from scratch
- Forward propagation
- Backpropagation
- Gradient descent optimisation
- NumPy vectorised matrix operations
- Model saving and loading
- Prediction confidence scores
- Visual prediction display using Matplotlib

## Future Improvements
- Add loss function tracking
- Plot training loss
- Add confusion matrix
- Create an interactive web demo

## Credits

Inspired by the YouTube tutorial:

* *Neural Network from Scratch in Python* — Samson Zhang

The tutorial was used as a learning resource for understanding the fundamentals of neural networks. The implementation was extended with additional features, documentation, model saving/loading, confidence scoring, and visualisation.
