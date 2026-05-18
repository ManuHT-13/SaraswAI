"""
Convertimos los audios del dataset en espectrogramas de Mel con los mismos parametros que usa CNN14
y los guardamos como vectores de numpy junto a un csv con los datos de cada muestra
"""

import argparse
import sys
import csv
import time
from pathlib import Path
from collections import Counter
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Parametros para el espectrograma de Mel
SR         = 16000      # Sample rate (frecuencia de muestreo)
N_FFT      = 512        # Tamanyo de la ventana para la transformada de fourier rapida
HOP_LENGTH = 160        # Muestras realizadas en cada ventana
N_MELS     = 64         # Numero de bandas de frecuencia
F_MIN      = 50.0       # Frecuencia minima del espetrograma
F_MAX      = 8000.0     # Frecuencia maxima del espectrograma

# Labels de las distintas clases
INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}

# Nombre que tiene cada split en el dataset
SPLITS = {"train": "train_ds", "test":  "test_ds"}


# Funcion de convertir el audio en espectrograma de mel
def audio_to_spectogram(audio):
    """
    Convierte un audio crudo en un espectrograma de Mel con los parametros que usa CNN14
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.astype(np.float32)


def process_split(data_dir, out_dir, split, max_samples=None):
    """
    Preprocesamos uno de los splits (train o test) del dataset
    Muestra la distribucion de cada una de las clases de instrumentos del split
    """

    snapshot_dir  = data_dir / SPLITS[split]
    out_split_dir = out_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{split}_labels.csv"

    ds   = tf.data.Dataset.load(str(snapshot_dir))
    rows = []

    # Para cada muestra del dataset
    for i, sample in enumerate(ds):
        # Si llegamos a las muestras maximas terminamos
        if max_samples and i >= max_samples:
            break
        
        # Extraemos los datos
        sample_id     = sample["id"].numpy().decode("utf-8")
        audio         = sample["audio"].numpy()
        family_id     = int(sample["instrument"]["family"].numpy())
        source_id     = int(sample["instrument"]["source"].numpy())
        instrument_id = int(sample["instrument"]["label"].numpy())
        pitch         = int(sample["pitch"].numpy())
        velocity      = int(sample["velocity"].numpy())
        family_name   = INSTRUMENT_LABELS.get(family_id, "unknown")

        out_path = out_split_dir / f"{sample_id}.npy"

        # Guardamos el vector de valores del espectrograma
        np.save(out_path, audio_to_spectogram(audio))

        # Guardamos la fila con todas las columnas
        rows.append([sample_id, family_id, family_name, source_id, instrument_id, pitch, velocity, str(out_path)])

    # Escribimos el csv con todos los datos
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "family_id", "family_name", "source_id", "instrument_id", "pitch", "velocity", "npy_path"])
        writer.writerows(rows)

    print(f"{split}: {len(rows)} muestras procesadas")

    # Calculamos la proporcion de cada familia 
    fam_counts = Counter(r[2] for r in rows)
    total = len(rows)
    for name, cnt in fam_counts.items():
        print(f"  {name}: {cnt} ({cnt/total*100:.1f}%)")

    # Mostramos las distribucion de clases
    classes = list(fam_counts.keys())
    counts  = list(fam_counts.values())

    plt.figure(figsize=(10, 4))
    plt.bar(classes, counts)
    plt.title(f"Distribución de clases - {split}")
    plt.xlabel("Clase")
    plt.ylabel("Número de muestras")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(out_dir / f"{split}_class_distribution.png", dpi=150)
    plt.show()


def run():
    parser = argparse.ArgumentParser()

    # Argumento de muestras maximas para hacer pruebas
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path("./dataset")
    out_dir  = Path("./dataset/processed")

    if not data_dir.exists():
        print(f"[error] No se encuentra: {data_dir}")
        sys.exit()

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        print(f"Procesando {split}...\n")
        process_split(data_dir, out_dir, split, args.max)

    print("Preprocesado completado")


if __name__ == "__main__":
    run()