"""
Clasificador de regresion logistica sobre los embeddings de CNN14 para comparar con la MLP
"""

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

FEATURES_PATH = "./dataset/features"

INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}


def load_data():
    """
    Cargar los datos y mapear las clases
    """
    X_train = numpy.load(f"{FEATURES_PATH}/train_embeddings.npy")
    X_test = numpy.load(f"{FEATURES_PATH}/test_embeddings.npy")
    y_train_raw = numpy.load(f"{FEATURES_PATH}/train_labels.npy")
    y_test_raw = numpy.load(f"{FEATURES_PATH}/test_labels.npy")

    classes = numpy.unique(y_train_raw)
    mapa = {int(c): i for i, c in enumerate(classes)}
    y_train_raw = numpy.array([mapa[int(c)] for c in y_train_raw])
    y_test_raw = numpy.array([mapa[int(c)] for c in y_test_raw])

    return X_train, X_test, y_train_raw, y_test_raw, classes


def normalize(X_train, X_test):
    """
    Normalizar ambos splits
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def plot_confusion(conf, n_classes, label_names):
    """
    Dibujamos la matriz de confusion
    """
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
    plt.savefig("./checkpoints/confusion_matrix_rl.png", dpi=150)
    plt.show()


def run():
    # Cargamos y normalizamos los datos
    X_train, X_test, y_train_raw, y_test_raw, clases = load_data()
    X_train, X_test = normalize(X_train, X_test)

    n_classes = len(clases)
    label_names = {i: INSTRUMENT_LABELS[int(clases[i])] for i in range(n_classes)}

    # Entrenamos la regresion logistica 
    print("entrenando RL...")
    rl = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )
    rl.fit(X_train, y_train_raw)

    # Evaluamos con el split de test e imprimimos los resultados
    y_pred_train = rl.predict(X_train)
    y_pred_test = rl.predict(X_test)

    acc_train = accuracy_score(y_train_raw, y_pred_train) * 100
    acc_test = accuracy_score(y_test_raw,  y_pred_test)  * 100
    print(f"accuracy train: {acc_train:.2f}%")
    print(f"accuracy test:  {acc_test:.2f}%")

    # Mostramos la matriz de confusion 
    print("\naccuracy por clase:")
    conf = confusion_matrix(y_test_raw, y_pred_test)
    for i in range(n_classes):
        total = conf[i].sum()
        correct = conf[i, i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {label_names[i]:<12} {correct:>4}/{total:<4} ({acc:.1f}%)")

    plot_confusion(conf, n_classes, label_names)


if __name__ == "__main__":
    run()