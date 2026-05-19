"""
Analisis simple de los embeddings generados por la capa convolucional de CNN14

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FEATURES_PATH = "./dataset/features"
OUT_PATH = "./checkpoints/analysis"

INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}

os.makedirs(OUT_PATH, exist_ok=True)


def load():
    """
    Cargamos los datos
    """
    X_train = np.load(f"{FEATURES_PATH}/train_embeddings.npy")
    X_test  = np.load(f"{FEATURES_PATH}/test_embeddings.npy")
    y_train = np.load(f"{FEATURES_PATH}/train_labels.npy")
    y_test  = np.load(f"{FEATURES_PATH}/test_labels.npy")

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    return X, y, X_train


def normalize(X, X_train_ref):
    """
    Normalizamos los datos para que el analisis sea mas afin
    a lo que despues se van a encontrar nuestros modelos
    """
    mean = X_train_ref.mean(axis=0)
    std = X_train_ref.std(axis=0) + 1e-8
    return (X - mean) / std


def basic_stats(X, y):
    """
    Analizamos estadisticas basicas como la media, desviacion,
    minimo y maximo de cada dimension
    """
    print("shape:", X.shape)
    print("media:", X.mean())
    print("std:", X.std())
    print("min:", X.min())
    print("max:", X.max())


def plot_distribution(y):
    """
    Analizamos la distribucion de clases
    """
    classes, counts = np.unique(y, return_counts=True)
    names = [INSTRUMENT_LABELS[int(c)] for c in classes]

    plt.figure(figsize=(10, 4))
    plt.bar(names, counts)
    plt.xticks(rotation=45)
    plt.title("Distribucion de clases")
    plt.tight_layout()

    path = f"{OUT_PATH}/class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.show()

    print("guardado:", path)


def variance_analysis(X):
    """
    Analizamos la varianza de las dimensiones para determinar
    cuales no tienen importancia separando clases y simplemente estan
    introduciendo ruido (esto va a ser util para luego eliminar
    dimensiones en el script de filter embeddings)
    """
    var = X.var(axis=0)
    sorted_var = np.sort(var)[::-1]

    plt.figure(figsize=(10, 4))
    plt.plot(sorted_var)
    plt.title("Varianza de embeddings")
    plt.xlabel("dimension")
    plt.ylabel("varianza")
    plt.tight_layout()

    path = f"{OUT_PATH}/variance.png"
    plt.savefig(path, dpi=150)
    plt.show()

    print("dimensiones muertas:", np.sum(var < 1e-3))


def plot_pca(X, y):
    """
    Reducimos el espacio de los embeddings a solo sus dimensiones con
    mas variaza para visualizarlos en una representacion de dos dimensiones
    y analizar la separacion entre familias
    """
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for c in np.unique(y):
        mask = y == c
        name = INSTRUMENT_LABELS.get(int(c), str(c))
        plt.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.4, label=name)

    plt.title("PCA 2 Dimensiones")
    plt.legend()
    plt.tight_layout()

    path = f"{OUT_PATH}/pca_2d.png"
    plt.savefig(path, dpi=150)
    plt.show()

    print("guardado en:", path)



def run():
    X, y, X_train = load()

    # Normalizamos los datos
    X = normalize(X, X_train)

    # Calculamos los distintos analisis y los mostramos
    print("\n========== ESTADISTICAS =========")
    basic_stats(X, y)

    plot_distribution(y)

    print("\n========== VARIANZA ==========")
    variance_analysis(X)

    print("\n========== PCA ==========")
    plot_pca(X, y)


if __name__ == "__main__":
    run()