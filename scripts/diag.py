import numpy as np
X = np.load("./dataset/features/train_embeddings.npy")
mean = X.mean(axis=0)
std  = X.std(axis=0) + 1e-8
X_n  = (X - mean) / std
print(X_n.mean(), X_n.std(), X_n.min(), X_n.max())

# Y cuantas dimensiones son todo ceros
print(f"dimensiones muertas (std~0): {(X.std(axis=0) < 1e-4).sum()} / {X.shape[1]}")