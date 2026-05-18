"""
Validacion cruzada de 5 folds sobre los embeddings extraidos con CNN14
"""

import sys
sys.path.insert(0, "./models")

import numpy
import cupy as np
from sklearn.model_selection import StratifiedKFold

from mlp_numpy import init_weights, training, predict

FEATURES_DIR = "./dataset/features"

LAYER_SIZES = [2048, 256, 10]
ALPHA       = 0.001
LAMBDA      = 0.0001
BATCH_SIZE  = 256
NUM_ITERS   = 100
N_FOLDS     = 5


def compute_class_weights(y_raw, n_classes):
    counts  = numpy.bincount(y_raw, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def normalize(X_tr, X_val):
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std


def main():
    # Cargamos y juntamos train y test
    X_train     = numpy.load(f"{FEATURES_DIR}/train_embeddings.npy")
    X_test      = numpy.load(f"{FEATURES_DIR}/test_embeddings.npy")
    y_train_raw = numpy.load(f"{FEATURES_DIR}/train_labels.npy")
    y_test_raw  = numpy.load(f"{FEATURES_DIR}/test_labels.npy")

    X_all = numpy.concatenate([X_train, X_test], axis=0)
    y_all = numpy.concatenate([y_train_raw, y_test_raw], axis=0)

    # Remapeamos labels a indices contiguos
    clases  = numpy.unique(y_all)
    mapa    = {int(c): i for i, c in enumerate(clases)}
    y_all   = numpy.array([mapa[int(c)] for c in y_all])
    n_classes = len(clases)

    print(f"total muestras: {X_all.shape[0]}  clases: {n_classes}")

    # Usamos StratifiedKFold para mantener la distribucion de clases en cada fold
    kf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all)):
        X_tr  = np.array(X_all[train_idx])
        X_val = np.array(X_all[val_idx])
        y_tr  = y_all[train_idx]
        y_val = y_all[val_idx]

        X_tr, X_val = normalize(X_tr, X_val)

        class_weights = compute_class_weights(y_tr, n_classes)
        class_weights = np.array(class_weights)

        y_tr_oh  = np.eye(n_classes)[y_tr]

        thetas_ini = init_weights(LAYER_SIZES)
        thetas, _, _ = training(
            X_tr, y_tr_oh, thetas_ini,
            alpha=ALPHA,
            num_iters=NUM_ITERS,
            lambda_=LAMBDA,
            batch_size=BATCH_SIZE,
            class_weights=class_weights,
        )

        y_pred = predict(thetas, X_val)
        y_val_cp = np.array(y_val)
        acc = float(np.mean(y_pred == y_val_cp) * 100)
        accs.append(acc)
        print(f"fold {fold+1}/{N_FOLDS}: {acc:.2f}%")

    print(f"\nmedia: {numpy.mean(accs):.2f}%  std: {numpy.std(accs):.2f}%")


if __name__ == "__main__":
    main()