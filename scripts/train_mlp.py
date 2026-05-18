"""
Carga los embeddings extraidos con CNN14, entrena la MLP y guarda los pesos
"""

import cupy as np
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./models")
from mlp_numpy import init_weights, training, predict

FEATURES_DIR = "./dataset/features"
WEIGHTS_PATH = "./checkpoints/thetas_mlp.npz"
STATS_PATH   = "./checkpoints/embedding_stats.npy"
LABEL_MAP    = "./checkpoints/label_map.npy"

LAYER_SIZES = [2048, 256, 10]
ALPHA       = 0.001
LAMBDA      = 0.0001
BATCH_SIZE  = 256
NUM_ITERS   = 80

INSTRUMENT_LABELS = {
    0: "bass", 1: "brass", 2: "flute", 3: "guitar",
    4: "keyboard", 5: "mallet", 6: "organ", 7: "reed",
    8: "string", 9: "synth_lead", 10: "vocal",
}


def load_data():
    X_train     = np.load(f"{FEATURES_DIR}/train_embeddings.npy")
    X_test      = np.load(f"{FEATURES_DIR}/test_embeddings.npy")
    y_train_raw = np.load(f"{FEATURES_DIR}/train_labels.npy")
    y_test_raw  = np.load(f"{FEATURES_DIR}/test_labels.npy")

    # cupy no puede indexar con arrays de numpy directamente
    y_train_raw = y_train_raw.get() if hasattr(y_train_raw, "get") else y_train_raw
    y_test_raw  = y_test_raw.get()  if hasattr(y_test_raw,  "get") else y_test_raw

    # Remapeamos labels a indices contiguos 0-10
    clases      = numpy.unique(y_train_raw)
    mapa        = {int(c): i for i, c in enumerate(clases)}
    y_train_raw = numpy.array([mapa[int(c)] for c in y_train_raw])
    y_test_raw  = numpy.array([mapa[int(c)] for c in y_test_raw])

    return X_train, X_test, y_train_raw, y_test_raw, clases


def compute_class_weights(y_raw, n_classes):
    # Cuantos menos ejemplos tiene una clase, mas peso le damos
    counts = numpy.bincount(y_raw, minlength=n_classes).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    mean_np = mean.get() if hasattr(mean, "get") else mean
    std_np  = std.get()  if hasattr(std,  "get") else std
    numpy.save(STATS_PATH, numpy.array([mean_np, std_np]))

    return X_train, X_test


def plot_results(J_history, J_val_history, y_true, y_pred, n_classes, label_names):
    # Curva de loss
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

    # Matriz de confusion
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

    # Accuracy por clase
    print("\naccuracy por clase:")
    for i in range(n_classes):
        total   = conf[i].sum()
        correct = conf[i, i]
        acc     = correct / total * 100 if total > 0 else 0
        print(f"  {label_names[i]:<12} {correct:>4}/{total:<4} ({acc:.1f}%)")


def main():
    X_train, X_test, y_train_raw, y_test_raw, clases = load_data()
    print(f"train: {X_train.shape}  test: {X_test.shape}  clases: {len(clases)}")

    X_train, X_test = normalize(X_train, X_test)

    n_classes = len(clases)
    y_train   = np.eye(n_classes)[y_train_raw]
    y_test    = np.eye(n_classes)[y_test_raw]

    thetas_ini = init_weights(LAYER_SIZES)
    print(f"arquitectura: {' -> '.join(map(str, LAYER_SIZES))}")

    class_weights = compute_class_weights(y_train_raw, n_classes)
    class_weights = np.array(class_weights)  # pasamos a cupy

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

    y_pred_train = predict(thetas, X_train)
    y_pred_test  = predict(thetas, X_test)

    y_train_cp = np.array(y_train_raw)
    y_test_cp  = np.array(y_test_raw)

    acc_train = float(np.mean(y_pred_train == y_train_cp) * 100)
    acc_test  = float(np.mean(y_pred_test  == y_test_cp)  * 100)
    print(f"accuracy train: {acc_train:.2f}%")
    print(f"accuracy test:  {acc_test:.2f}%")

    numpy.savez(WEIGHTS_PATH, *[t.get() for t in thetas])
    numpy.save(LABEL_MAP, clases)

    y_true_np = numpy.array(y_test_cp.get() if hasattr(y_test_cp, "get") else y_test_cp)
    y_pred_np = numpy.array(y_pred_test.get() if hasattr(y_pred_test, "get") else y_pred_test)

    label_names = {i: INSTRUMENT_LABELS[k] for i, k in enumerate(clases)}
    plot_results(J_history, J_val_history, y_true_np, y_pred_np, n_classes, label_names)


if __name__ == "__main__":
    main()