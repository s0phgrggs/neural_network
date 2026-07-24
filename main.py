import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mnist import MNIST

mndata = MNIST("/Users/sophiegriggs/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1")

images, labels = mndata.load_training()

print(len(images))
print(labels[0])
print(images[0])