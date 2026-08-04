import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network import *

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

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")

W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

print("Model loaded!")

def test_prediction(index, W1, b1, W2, b2):
    
    current_image = dev_images[:, index, None]

    prediction, confidence = make_predictions(current_image, W1, b1, W2, b2)

    print("Prediction:", prediction[0])
    print("Actual:", dev_labels[index])
    print("Confidence:", round(confidence[0] * 100, 2), "%")

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