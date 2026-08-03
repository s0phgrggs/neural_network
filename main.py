import numpy as np
import pandas as pd

data = pd.read_csv("mnist_train.csv")
data = np.array(data)

np.random.shuffle(data)

labels = data[:, 0]
#changes to black or white rather than brightness
images = data[:, 1:] / 255.0

m, n = data.shape
print(f"Images: {m}")
print(f"Pixels per image: {n}")

dev_images = images[:1000]
dev_labels = labels[:1000]

train_images = images[1000:]
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

def softmax(Z):
    #axis 0: does the operation down the rows separately for each column
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

