"""
Visualiza un espectrograma aleatorio de una clase del dataset o de un archivo que le pases (depende de lo que pongas en el parametro)
Clases: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, vocal

"""

import sys
import random
import csv
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

PROCESSED_DIR = "./dataset/processed"

SR            = 16000
HOP_LENGTH    = 160
N_FFT         = 512
N_MELS        = 64
F_MIN         = 50.0
F_MAX         = 8000.0
DURATION      = 4.0

CLASES_VALIDAS = {"bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal"}


def mel_de_audio(audio_path: str) -> tuple[np.ndarray, str]:
    """
    Carga un archivo de audio y calcula su espectrograma de Mel
    """
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Recortamos o rellenamos a 4s igual que en el preprocesado
    target_len = int(SR * DURATION)
    if len(audio) >= target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32), Path(audio_path).stem


def run(arg: str):
    clase_norm = arg.lower().replace(" ", "_")

    if clase_norm in CLASES_VALIDAS:
        # Buscamos en el CSV de train muestras de esa clase
        csv_path = f"{PROCESSED_DIR}/train_labels.csv"
        candidatos = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["family_name"].lower().replace(" ", "_") == clase_norm:
                    candidatos.append(row["npy_path"])

        if not candidatos:
            print(f"[error] Clase '{arg}' no encontrada en el CSV.")
            sys.exit(1)

        npy_path = random.choice(candidatos)
        log_mel  = np.load(npy_path)
        titulo   = f"Clase: {arg}"
        nombre   = clase_norm
        print(f"Mostrando: {npy_path}")

    elif Path(arg).exists():
        # Es una ruta de audio valida
        log_mel, nombre = mel_de_audio(arg)
        titulo = f"Audio: {Path(arg).name}"
        print(f"Mostrando mel de: {arg}")

    else:
        print(f"[error] '{arg}' no es una clase valida ni un archivo existente.")
        print(f"  Clases disponibles: {', '.join(sorted(CLASES_VALIDAS))}")
        sys.exit(1)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel, sr=SR, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(f"./checkpoints/mel_{nombre}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.visualize_mel <clase o mi_audio.mp3/wav>")
        sys.exit()
    run(sys.argv[1])