"""
Validacion cruzada de 5 folds sobre los featurings extraidos con CNN14

* Uso opcional de GPU con Cupy, si se activa aqui tambien se debe de activar en mlp.py
* Uso opcional de embeddings filtrados (generados por filter_embeddings.py)
"""

# Parametro de usar la GPU con Cupy
USES_GPU     = False
# Parametro de usar los embeddings filtrados por el script filter_embeddings
USE_FILTERED = True

import numpy as np_cpu

if USES_GPU:
    # Nuestras funciones de la MLP deberan recibir arrays de Cupy
    # Mientras que otras funciones como las de sklearn esperan arrays de Numpy normales
    import cupy as xp
    xp.cuda.Device(0).use()
else:
    import numpy as xp

from sklearn.model_selection import StratifiedKFold

from models.mlp import init_weights, training, predict


RANDOM_SEED = 429

FEATURES_PATH = "./dataset/features"

# Arquitectura, la primera capa cogera la dimension de los embeddings directamente
LAYER_SIZES = [2048, 256, 128, 10]

# Hiperparametros de entreno igual que en train normal
ALPHA       = 0.001
LAMBDA      = 0.001
BATCH_SIZE  = 256
NUM_ITERS   = 120
N_FOLDS     = 5


def to_numpy(x):
    """
    Para resolver las diferencias entre Cupy y Numpy,
    siempre que se necesite un array estrictamente de Numpy se pasa
    por esta funcion
    """
    return x.get() if USES_GPU and hasattr(x, "get") else np_cpu.asarray(x)


def compute_class_weights(y_raw, n_classes):
    """
    Calculamos los pesos para cada clase igual que en train para
    compensar el desbalance entre clases
    """
    counts = np_cpu.bincount(y_raw, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def normalize(X_tr, X_val):
    """
    Normalizamos ambos splits, el de training
    y el de validacion
    """
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    return (X_tr - mean) / std, (X_val - mean) / std


def run():
    # Cargamos y juntamos train y test como arrays de Cupy,
    # pero las labels las mantenemos en Numpy para compatibilidad con sklearn

    # Sufijo opcional para usar los embeddings filtrados o no
    suffix = "_filtered" if USE_FILTERED else ""

    X_train = xp.load(f"{FEATURES_PATH}/train_embeddings{suffix}.npy")
    X_test  = xp.load(f"{FEATURES_PATH}/test_embeddings{suffix}.npy")
    y_train_raw = np_cpu.load(f"{FEATURES_PATH}/train_labels.npy")
    y_test_raw  = np_cpu.load(f"{FEATURES_PATH}/test_labels.npy")

    y_train_raw = to_numpy(y_train_raw)
    y_test_raw  = to_numpy(y_test_raw)

    X_all = xp.concatenate([X_train, X_test], axis=0)
    y_all = np_cpu.concatenate([y_train_raw, y_test_raw], axis=0)

    # Remapeamos labels a indices contiguos
    clases = np_cpu.unique(y_all)
    mapa = {int(c): i for i, c in enumerate(clases)}
    y_all = np_cpu.array([mapa[int(c)] for c in y_all])
    n_classes = len(clases)

    print(f"total muestras: {X_all.shape[0]}  clases: {n_classes}")

    # Usamos StratifiedKFold para mantener la distribucion de clases en cada fold
    # Los indices los manejamos siempre en Numpy ya que sklearn no soporta Cupy
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    accs = []

    # Para cada fold entrenamos el modelo
    for fold, (train_idx, val_idx) in enumerate(kf.split(to_numpy(X_all), y_all)):
        X_tr  = xp.array(X_all[train_idx])
        X_val = xp.array(X_all[val_idx])
        y_tr  = y_all[train_idx]
        y_val = y_all[val_idx]

        X_tr, X_val = normalize(X_tr, X_val)

        class_weights = compute_class_weights(y_tr, n_classes)
        class_weights = xp.array(class_weights)

        # One-hot encoding en Cupy usando los indices de Numpy
        y_tr_oh = xp.eye(n_classes)[y_tr]

        # Ajustamos la dimension de entrada segun los embeddings cargados
        layer_sizes = [X_tr.shape[1]] + LAYER_SIZES[1:]
        thetas_ini = init_weights(layer_sizes)

        thetas, _, _ = training(
            X_tr, y_tr_oh, thetas_ini,
            alpha=ALPHA,
            num_iters=NUM_ITERS,
            lambda_=LAMBDA,
            batch_size=BATCH_SIZE,
            class_weights=class_weights,
        )

        # Una vez entrenado calculamos la precision en este fold y la mostramos
        y_pred = predict(thetas, X_val)
        y_val_cp = xp.array(y_val)
        acc = float(xp.mean(y_pred == y_val_cp) * 100)
        accs.append(acc)
        print(f"fold {fold+1}/{N_FOLDS}: {acc:.2f}%")

    # Mostramos la media y desviacion de la precision entre todos los folds realizados
    print(f"\nmedia: {np_cpu.mean(accs):.2f}%  std: {np_cpu.std(accs):.2f}%")


if __name__ == "__main__":
    run()