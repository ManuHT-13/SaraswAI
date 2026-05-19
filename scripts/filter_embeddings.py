"""
Elimina las dimensiones muertas de los embeddings (aquellas con std < STD_THRESHOLD)
y guarda los arrays filtrados junto con la mascara de dimensiones activas (array de booleanos que indica que dimensiones estan activas)

"""

import argparse
import numpy as np
import os

FEATURES_PATH    = "./dataset/features"
CHECKPOINTS_PATH = "./checkpoints"

# Consideraremos como muertas las dimensiones con desviacion por debajo de este umbral
STD_THRESHOLD = 0.001


def run():
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    # Cargamos los embeddings
    X_train = np.load(f"{FEATURES_PATH}/train_embeddings.npy")
    X_test  = np.load(f"{FEATURES_PATH}/test_embeddings.npy")
    print(f"  train: {X_train.shape}  test: {X_test.shape}")

    # Calculamos la std de cada dimension usando solo train
    std = X_train.std(axis=0)

    # Mascara booleana de dimensiones activas o muertas
    mask = std >= STD_THRESHOLD

    # Mostramos el numero de dimensiones activas y muertas
    n_activas = int(mask.sum())
    n_muertas = int((~mask).sum())
    print(f"\numbral std: {STD_THRESHOLD}")
    print(f"  dims activas: {n_activas} / {X_train.shape[1]}")
    print(f"  dims muertas: {n_muertas} / {X_train.shape[1]}")

    # Aplicamos la mascara a ambos splits
    X_train_f = X_train[:, mask]
    X_test_f  = X_test[:, mask]
    print(f"\n  train filtrado: {X_train.shape} → {X_train_f.shape}")
    print(f"  test  filtrado: {X_test.shape}  → {X_test_f.shape}")

    # Guardamos los arrays filtrados
    out_train = f"{FEATURES_PATH}/train_embeddings_filtered.npy"
    out_test  = f"{FEATURES_PATH}/test_embeddings_filtered.npy"
    np.save(out_train, X_train_f)
    np.save(out_test,  X_test_f)
    print(f"\n  guardado: {out_train}")
    print(f"  guardado: {out_test}")

    # Guardamos la mascara para aplicarla cuando hagamos predict etc
    mask_path = f"{CHECKPOINTS_PATH}/active_dims_mask.npy"
    np.save(mask_path, mask)
    print(f"  guardado: {mask_path}")

if __name__ == "__main__":
    run()