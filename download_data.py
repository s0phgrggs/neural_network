print("Starting download script")

import kagglehub

print("Kagglehub imported")

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)