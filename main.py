import numpy as np
import pandas as pd

data = pd.read_csv("mnist_train.csv")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

labels = data[:, 0]
images = data[:, 1:]
images = images / 255.0

print("Number of images:", m)
print("Number of pixels:", n - 1)