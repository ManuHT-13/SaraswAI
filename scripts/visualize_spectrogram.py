"""
Visualiza un espectrograma aleatorio de una clase del dataset
Clases: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, vocal
"""

import sys
import random
import csv
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

PROCESSED_DIR = "./dataset/processed"

SR            = 16000
HOP_LENGTH    = 160
F_MIN         = 50.0
F_MAX         = 8000.0


def run(clase: str):
    # Buscamos en el CSV de train muestras de esa clase
    csv_path = f"{PROCESSED_DIR}/train_labels.csv"
    candidatos = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["family_name"].lower().replace(" ", "_") == clase.lower().replace(" ", "_"):
                candidatos.append(row["npy_path"])

    if not candidatos:
        print(f"[error] Clase '{clase}' no encontrada. Clases disponibles: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, vocal")
        sys.exit(1)

    # Elegimos uno aleatorio y lo cargamos
    npy_path = random.choice(candidatos)
    log_mel  = np.load(npy_path)

    print(f"Mostrando: {npy_path}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=SR, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Clase: {clase}")
    plt.tight_layout()
    plt.savefig(f"./checkpoints/mel_{clase}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.visualize_mel <clase>")
        sys.exit(1)
    run(sys.argv[1])