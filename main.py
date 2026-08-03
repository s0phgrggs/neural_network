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