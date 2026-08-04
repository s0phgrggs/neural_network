# Neural Network from Scratch - MNIST Digit Classifier

## Overview
This project executes a fully connected neural network from scratch using Python and NumPy.
The goal of this project was to understand the maths and internal processes behind neural networks by building each of the parts manually rather than using libraries like tensorflow.

## Dataset
The model uses the MNIST handwritten digit dataset.
Each example contians:
- One label representing 0-9 
- 784 pixel values representing a 28x28 image
Each of thes eis then converted into a vextor of 784 pixels which are then normalised to hold a value of 0-1.
This improves the training stability as it makes the inputs easier for the optimation algorithm to work with.

The network contains an input layer of 784 neurons, a hidden layer of 64 layers, ReLU activation, an output layer of 10 neurons, softmax activation and prediction.

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
This model achieves approximately 90% accuracy on unseen development data.

# Features
- Neural network implemented from scratch
- NumPy matrix operations
- Custom forward propagation
- Custom backpropagation
- Gradient descent training
- Model saving/loading
- Prediction confidence scores
- Visual prediction display

## Future Improvements
- Add loss function tracking
- Plot training loss
- Add confusion matrix
- Create an interactive web demo