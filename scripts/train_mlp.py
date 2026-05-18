"""
Carga los embeddings extraidos con CNN14, entrena la MLP y guarda los pesos
resultantes del training
"""

# Nuestras funciones de la MLP deberan recibir arrays de Cupy
import cupy as np
# Mientras que otras funciones como las de matplotlib esperan arrays de Numpy normales
import numpy

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./models")
from mlp import init_weights, training, predict

FEATURES_PATH = "./dataset/features"
WEIGHTS_PATH = "./checkpoints/thetas_mlp.npz"
STATS_PATH   = "./checkpoints/embedding_stats.npy"
LABEL_MAP    = "./checkpoints/label_map.npy"

# Arquitectura
LAYER_SIZES = [2048, 256, 10]

# Hiperparametros
ALPHA       = 0.001
LAMBDA      = 0.0001
BATCH_SIZE  = 256
NUM_ITERS   = 80

INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}


def load_data():
    """
    Cargamos ambos splits como arrays de Cupy pero mapeando
    los index con Numpy manualmente para las labels
    """
    X_train     = np.load(f"{FEATURES_PATH}/train_embeddings.npy")
    X_test      = np.load(f"{FEATURES_PATH}/test_embeddings.npy")
    y_train_raw = np.load(f"{FEATURES_PATH}/train_labels.npy")
    y_test_raw  = np.load(f"{FEATURES_PATH}/test_labels.npy")

    y_train_raw = y_train_raw.get() if hasattr(y_train_raw, "get") else y_train_raw
    y_test_raw  = y_test_raw.get()  if hasattr(y_test_raw,  "get") else y_test_raw

    # Mapeamos las labels
    clases      = numpy.unique(y_train_raw)
    mapa        = {int(c): i for i, c in enumerate(clases)}
    y_train_raw = numpy.array([mapa[int(c)] for c in y_train_raw])
    y_test_raw  = numpy.array([mapa[int(c)] for c in y_test_raw])

    return X_train, X_test, y_train_raw, y_test_raw, clases


def compute_class_weights(y_raw, n_classes):
    """
    Calculamos los pesos para cada clase, inversamente proporcionales
    a la cantidad de muestras que tenga dicha clase para compensar
    la distribuicion del dataset
    """
    counts = numpy.bincount(y_raw, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def normalize(X_train, X_test):
    """
    Normalizamos ambos splits usando Z-Score y guardamos
    la media y la desviacion estandar comoa arrays de Numpy
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    mean_np = mean.get() if hasattr(mean, "get") else mean
    std_np  = std.get()  if hasattr(std,  "get") else std
    numpy.save(STATS_PATH, numpy.array([mean_np, std_np]))

    return X_train, X_test


def plot_results(J_history, J_val_history, y_true, y_pred, n_classes, label_names):
    """
    Dibujamos y guardamos los resultados de la evaluacion
    despues del entreno (evolucion de la funcion de perdida normal y de validacion,
    matriz de confusion) e imprimimos la precision en cada clase
    """

    # Convertimos los historiales de la funcion de perdida en arrays de Numpy normales
    J_plot     = numpy.array([float(j) for j in J_history])
    J_val_plot = numpy.array([float(j) for j in J_val_history])

    plt.figure(figsize=(8, 4))
    plt.plot(J_plot,     label="train")
    plt.plot(J_val_plot, label="test")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./checkpoints/loss_curve.png", dpi=150)
    plt.show()

    # Realizamos la matriz de confusion
    conf = numpy.zeros((n_classes, n_classes), dtype=int)
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

    # Imprimimos la precision que hemos obtenido en cada clase
    print("\naccuracy por clase:")
    for i in range(n_classes):
        total   = conf[i].sum()
        correct = conf[i, i]
        acc     = correct / total * 100 if total > 0 else 0
        print(f"  {label_names[i]:<12} {correct:>4}/{total:<4} ({acc:.1f}%)")


def main():
    # Cargamos los splits
    X_train, X_test, y_train_raw, y_test_raw, clases = load_data()
    print(f"train: {X_train.shape}  test: {X_test.shape}  clases: {len(clases)}")

    # Normalizamos los datos
    X_train, X_test = normalize(X_train, X_test)

    # Codeamos los vectores de label con one-hot encoding
    n_classes = len(clases)
    y_train   = np.eye(n_classes)[y_train_raw]
    y_test    = np.eye(n_classes)[y_test_raw]

    # Inicializamos todos los pesos 
    thetas_ini = init_weights(LAYER_SIZES)
    print(f"arquitectura: {' -> '.join(map(str, LAYER_SIZES))}")

    # Inicializamos los pesos de cada clase para compensar la distribuicion de clases
    class_weights = compute_class_weights(y_train_raw, n_classes)
    class_weights = np.array(class_weights)

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
    y_pred_test  = predict(thetas, X_test)

    y_train_cp = np.array(y_train_raw)
    y_test_cp  = np.array(y_test_raw)

    acc_train = float(np.mean(y_pred_train == y_train_cp) * 100)
    acc_test  = float(np.mean(y_pred_test  == y_test_cp)  * 100)
    print(f"accuracy train: {acc_train:.2f}%")
    print(f"accuracy test:  {acc_test:.2f}%")

    # Guardamos los pesos del modelo
    numpy.savez(WEIGHTS_PATH, *[t.get() for t in thetas])
    numpy.save(LABEL_MAP, clases)

    # Pasamos a array de Numpy los arrays de labels e imprimimos resultados de evaluacion
    y_true_np = numpy.array(y_test_cp.get() if hasattr(y_test_cp, "get") else y_test_cp)
    y_pred_np = numpy.array(y_pred_test.get() if hasattr(y_pred_test, "get") else y_pred_test)

    label_names = {i: INSTRUMENT_LABELS[k] for i, k in enumerate(clases)}
    plot_results(J_history, J_val_history, y_true_np, y_pred_np, n_classes, label_names)


if __name__ == "__main__":
    main()