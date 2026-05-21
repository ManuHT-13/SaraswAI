"""
Carga los embeddings extraidos con CNN14, entrena la MLP y guarda los pesos
resultantes del training

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
    # Mientras que otras funciones como las de matplotlib esperan arrays de Numpy normales
    import cupy as xp
    xp.cuda.Device(0).use()
else:
    import numpy as xp

import matplotlib.pyplot as plt
from models.mlp import init_weights, training, predict

FEATURES_PATH   = "./dataset/features"
WEIGHTS_PATH    = "./checkpoints/thetas_mlp.npz"
STATS_PATH      = "./checkpoints/embedding_stats.npy"
LABEL_MAP       = "./checkpoints/label_map.npy"

# Arquitectura, la primera capa cogera la dimension de los embeddings directamente
LAYER_SIZES = [2048, 256, 128, 10]

# Hiperparametros
ALPHA       = 0.001
LAMBDA      = 0.001
BATCH_SIZE  = 256
NUM_ITERS   = 120

INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}


def to_numpy(x):
    """
    Para resolver las diferencias entre Cupy y Numpy,
    siempre que se necesite un array estrictamente de Numpy se pasa
    por esta funcion
    """
    return x.get() if USES_GPU and hasattr(x, "get") else np_cpu.asarray(x)


def load_data():
    """
    Cargamos ambos splits como arrays de Cupy pero mapeando
    los index con Numpy manualmente para las labels.
    Si USE_FILTERED es True cargamos los embeddings sin las dims muertas
    generados por filter_embeddings.py
    """
    suffix = "_filtered" if USE_FILTERED else ""

    X_train = xp.load(f"{FEATURES_PATH}/train_embeddings{suffix}.npy")
    X_test = xp.load(f"{FEATURES_PATH}/test_embeddings{suffix}.npy")
    y_train_raw = np_cpu.load(f"{FEATURES_PATH}/train_labels.npy")
    y_test_raw = np_cpu.load(f"{FEATURES_PATH}/test_labels.npy")

   
    y_train_raw = to_numpy(y_train_raw)
    y_test_raw = to_numpy(y_test_raw)

    # Mapeamos las labels
    clases = np_cpu.unique(y_train_raw)
    mapa = {int(c): i for i, c in enumerate(clases)}
    y_train_raw = np_cpu.array([mapa[int(c)] for c in y_train_raw])
    y_test_raw = np_cpu.array([mapa[int(c)] for c in y_test_raw])

    return X_train, X_test, y_train_raw, y_test_raw, clases


def compute_class_weights(y_raw, n_classes):
    """
    Calculamos los pesos para cada clase, inversamente proporcionales
    a la cantidad de muestras que tenga dicha clase para compensar
    la distribuicion del dataset
    """
    counts = np_cpu.bincount(y_raw, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def normalize(X_train, X_test):
    """
    Normalizamos ambos splits usando Z-Score y guardamos
    la media y la desviacion estandar comoa arrays de Numpy
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test  - mean) / std

    mean_np = mean.get() if hasattr(mean, "get") else mean
    std_np = std.get()  if hasattr(std,  "get") else std
    np_cpu.save(STATS_PATH, np_cpu.array([mean_np, std_np]))

    return X_train, X_test


def plot_results(J_history, J_val_history, y_true, y_pred, n_classes, label_names):
    """
    Dibujamos y guardamos los resultados de la evaluacion
    despues del entreno (evolucion de la funcion de perdida normal y de validacion,
    matriz de confusion) e imprimimos la precision en cada clase junto con
    recall, F1-score y soporte
    """

    # Convertimos los historiales de la funcion de perdida en arrays de Numpy normales
    J_plot = np_cpu.array([float(j) for j in J_history])
    J_val_plot = np_cpu.array([float(j) for j in J_val_history])

    plt.figure(figsize=(8, 4))
    plt.plot(J_plot, label="train")
    plt.plot(J_val_plot, label="test")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./checkpoints/loss_curve.png", dpi=150)
    plt.show()

    # Realizamos la matriz de confusion
    conf = np_cpu.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(conf, cmap="Blues")
    plt.colorbar()
    ticks = range(n_classes)
    names = [label_names[i] for i in range(n_classes)]
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)

    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=8)

    plt.xlabel("prediccion")
    plt.ylabel("real")
    plt.tight_layout()
    plt.savefig("./checkpoints/confusion_matrix.png", dpi=150)
    plt.show()

    # Calculamos las metricas por clase 
    TP = np_cpu.diag(conf).astype(float)
    FP = (conf.sum(axis=0) - TP).astype(float)
    FN = (conf.sum(axis=1) - TP).astype(float)
    support = conf.sum(axis=1).astype(float)

    precision_per_class = np_cpu.where(TP + FP > 0, TP / (TP + FP), 0.0)
    recall_per_class    = np_cpu.where(TP + FN > 0, TP / (TP + FN), 0.0)
    f1_per_class        = np_cpu.where(precision_per_class + recall_per_class > 0, 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class), 0.0)

    # Metricas globales
    total_samples   = support.sum()
    accuracy_global = TP.sum() / total_samples * 100

    macro_precision = precision_per_class.mean()
    macro_recall    = recall_per_class.mean()
    macro_f1        = f1_per_class.mean()

    weighted_precision = (precision_per_class * support).sum() / total_samples
    weighted_recall    = (recall_per_class    * support).sum() / total_samples
    weighted_f1        = (f1_per_class        * support).sum() / total_samples

    # Mostramos los resultados
    header = f"\n{'clase':<14} {'precision':>10} {'recall':>8} {'f1':>8} {'soporte':>9}"
    print(header)
    print("-" * len(header))

    for i in range(n_classes):
        print(
            f"  {label_names[i]:<12} "
            f"{precision_per_class[i]:>10.3f} "
            f"{recall_per_class[i]:>8.3f} "
            f"{f1_per_class[i]:>8.3f} "
            f"{int(support[i]):>9}"
        )

    print("-" * len(header))
    print(
        f"  {'accuracy':<12} {'':>10} {'':>8} {accuracy_global / 100:>8.3f} "
        f"{int(total_samples):>9}"
    )
    print(
        f"  {'macro avg':<12} "
        f"{macro_precision:>10.3f} "
        f"{macro_recall:>8.3f} "
        f"{macro_f1:>8.3f} "
        f"{int(total_samples):>9}"
    )
    print(
        f"  {'weighted avg':<12} "
        f"{weighted_precision:>10.3f} "
        f"{weighted_recall:>8.3f} "
        f"{weighted_f1:>8.3f} "
        f"{int(total_samples):>9}"
    )


def run():
    # Cargamos los splits
    X_train, X_test, y_train_raw, y_test_raw, clases = load_data()
    print(f"train: {X_train.shape}  test: {X_test.shape}  clases: {len(clases)}")

    # Normalizamos los datos
    X_train, X_test = normalize(X_train, X_test)

    # Codeamos los vectores de label con one-hot encoding
    n_classes = len(clases)
    y_train = xp.eye(n_classes)[y_train_raw]
    y_test = xp.eye(n_classes)[y_test_raw]

    # Ajustamos la dimension de entrada segun los embeddings cargados e inicializamos los pesos
    layer_sizes = [X_train.shape[1]] + LAYER_SIZES[1:]
    thetas_ini = init_weights(layer_sizes)
    
    arquitectura = ""
    for i, size in enumerate(layer_sizes):
        arquitectura += str(size)
        if i < len(layer_sizes) - 1:
            arquitectura += " -> "
    print("arquitectura: " + arquitectura)

    # Inicializamos los pesos de cada clase para compensar la distribuicion de clases
    class_weights = compute_class_weights(y_train_raw, n_classes)
    class_weights = xp.array(class_weights)

    # Entrenamos el modelo pasando tambien (X_test, y_test) para ver la perdida de validacion
    thetas, J_history, J_val_history = training(
        X_train, y_train, thetas_ini,
        alpha=ALPHA,
        num_iters=NUM_ITERS,
        lambda_=LAMBDA,
        batch_size=BATCH_SIZE,
        class_weights=class_weights,
        X_val=X_test,
        y_val=y_test,
    )

    # Realizamos la evaluacion e imprimimos resultados
    y_pred_train = predict(thetas, X_train)
    y_pred_test = predict(thetas, X_test)

    y_train_cp = xp.array(y_train_raw)
    y_test_cp = xp.array(y_test_raw)

    acc_train = float(xp.mean(y_pred_train == y_train_cp) * 100)
    acc_test = float(xp.mean(y_pred_test  == y_test_cp)  * 100)
    print(f"accuracy train: {acc_train:.2f}%")
    print(f"accuracy test:  {acc_test:.2f}%")

    # Guardamos los pesos del modelo
    np_cpu.savez(WEIGHTS_PATH, *[to_numpy(t) for t in thetas])
    np_cpu.save(LABEL_MAP, clases)

    # Pasamos a array de Numpy los arrays de labels e imprimimos resultados de evaluacion
    y_true_np = to_numpy(y_test_raw)
    y_pred_np = to_numpy(y_pred_test)

    label_names = {i: INSTRUMENT_LABELS[k] for i, k in enumerate(clases)}
    plot_results(J_history, J_val_history, y_true_np, y_pred_np, n_classes, label_names)


if __name__ == "__main__":
    run()