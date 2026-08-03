import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mnist_train.csv")
data = np.array(data)

np.random.shuffle(data)

labels = data[:, 0]
#normalise pixel values from 0-255 to 0-1
images = data[:, 1:] / 255.0

m, n = data.shape
print("Images:", m)
print("Pixels per image:", n)

dev_images = images[:1000].T
dev_labels = labels[:1000]

train_images = images[1000:].T
train_labels = labels[1000:]

print(train_images.shape)
print(dev_images.shape)

def init_params():

    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))

    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    #axis 0: does the operation down the rows separately for each column
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1

    return one_hot_Y.T

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y

    dW2 = (1 / X.shape[1]) * dZ2.dot(A1.T)
    db2 = (1 / X.shape[1]) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / X.shape[1]) * dZ1.dot(X.T)
    db1 = (1 / X.shape[1]) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, iterations, learning_rate):
    
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):

        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Backpropagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)

        # Update parameters
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Print progress every 10 iterations
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)

            print("Iteration:", i)
            print("Accuracy:", round(accuracy, 3))

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(train_images, train_labels, 500, 0.1)

def make_predictions(X, W1, b1, W2, b2):
    
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)

    predictions = get_predictions(A2)

    return predictions

def test_prediction(index, W1, b1, W2, b2):
    
    current_image = dev_images[:, index, None]

    prediction = make_predictions(
        current_image,
        W1,
        b1,
        W2,
        b2
    )

    print("Prediction:", prediction[0])
    print("Actual:", dev_labels[index])

    image = current_image.reshape((28, 28))

    plt.imshow(image, cmap="gray")
    plt.title("Prediction: " + str(prediction[0]) + ", Actual: " + str(dev_labels[index]))    
    plt.axis("off")
    plt.show()

def evaluate_dev_set(W1, b1, W2, b2):
    
    predictions = make_predictions(
        dev_images, W1, b1, W2, b2)
    accuracy = get_accuracy(predictions, dev_labels)    

    print("Dev accuracy:", round(accuracy, 3))

test_prediction(25, W1, b1, W2, b2)
evaluate_dev_set(W1, b1, W2, b2)