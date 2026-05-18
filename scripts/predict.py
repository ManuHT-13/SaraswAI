"""
Predice la familia de instrumento de un archivo de audio
"""

# Para que no de error importar pytorch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, "./models")

import argparse
from pathlib import Path

import numpy as np
import torch
import librosa

from embedder import Cnn14Embedder

# Parametros para el espectrograma de Mel
SR         = 16000      # Sample rate (frecuencia de muestreo)
N_FFT      = 512        # Tamanyo de la ventana para la transformada de fourier rapida
HOP_LENGTH = 160        # Muestras realizadas en cada ventana
N_MELS     = 64         # Numero de bandas de frecuencia
F_MIN      = 50.0       # Frecuencia minima del espetrograma
F_MAX      = 8000.0     # Frecuencia maxima del espectrograma
DURATION   = 4.0        # Duracion de los audios de nsynth

INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}

CNN14_CHECKPOINT   = "./checkpoints/Cnn14_16k_mAP=0.438.pth"
THETAS_PATH  = "./checkpoints/thetas_mlp.npz"
LABEL_MAP    = "./checkpoints/label_map.npy"
MEL_STATS    = "./dataset/features/mel_stats.npy"
EMB_STATS    = "./checkpoints/embedding_stats.npy"


def relu(z):
    """
    Funcion de activacion reLu igual que la de la MLP
    """
    return np.maximum(0, z)

def softmax(z):
    """
    Funcion de activacion final softmax igual que la de la MLP
    """
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def mlp_forward(thetas, x):
    """
    Propagacion hacia adelante de la MLP con las thetas ya calculadas
    """
    a = x
    for i, theta in enumerate(thetas):
        a_bias = np.concatenate([[1.0], a])
        z = theta @ a_bias
        a = relu(z) if i < len(thetas) - 1 else softmax(z)
    return a


def audio_to_mel(segment):
    """
    Convertimos una ventana o segmento de audio en espectrograma de Mel
    con los mismos parametros que hemos usado siempre 
    """
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def predict(audio_path):
    """
    Procesamos el audio en cuestion pasandolo a espectrograma de Mel
    y a vector de features a traves de la capa convolucional de CNN14
    para finalmente clasificarlo con la MLP y dar un top 3 predicciones mas probables
    """
    # Si tenemos gpu usamos gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Cnn14Embedder(CNN14_CHECKPOINT, device).to(device)
    embedder.eval()

    # Cargamos los pesos de la mlp
    data = np.load(THETAS_PATH)
    thetas = [data[k] for k in sorted(data.files)]

    # Cargamos las etiquetas con sus ids 
    label_map = np.load(LABEL_MAP).flatten()
    idx_to_name = {i: INSTRUMENT_LABELS[int(c)] for i, c in enumerate(label_map)}

    # Cargamos los parametros de regularizacion 
    mel_mean, mel_std = np.load(MEL_STATS)
    emb_stats = np.load(EMB_STATS)
    emb_mean, emb_std = emb_stats[0], emb_stats[1]

    # Cargamos el audio
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Dividimos el audio en ventanas para poder procesar audios de mas de 4s
    target_len = int(SR * DURATION)
    hop_len = int(SR * 2.0)
    starts = list(range(0, len(audio) - target_len + 1, hop_len)) or [0]

    # Calculamos las probabilidades para cada ventana
    all_probs = []
    for start in starts:
        segment = audio[start : start + target_len]
        # Si el segmento es mas corto que target_len lo rellenamos con ceros
        if len(segment) < target_len:
            segment = np.pad(segment, (0, target_len - len(segment)))

        # Convertimos a mel y lo normalizamos
        log_mel = audio_to_mel(segment)
        log_mel = ((log_mel - mel_mean) / (mel_std + 1e-6)).astype(np.float32)

        # Extraemos el vector de features
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emb = embedder(tensor.to(device)).cpu().numpy().squeeze()

        # Normalizamos el vector
        emb_n = (emb - emb_mean) / emb_std

        # Lo metemos en la MLP para obtener las probabilidades
        all_probs.append(mlp_forward(thetas, emb_n))

    # Calculamos la media de entre todas las ventanas y damos un top 3
    mean_probs = np.mean(all_probs, axis=0)
    top3 = np.argsort(mean_probs)[::-1][:3]

    # Imprimimos resultados
    print(f"\naudio: {audio_path}")
    for i, idx in enumerate(top3):
        marker = " <--" if i == 0 else ""
        print(f"  {idx_to_name[idx]:<14} {mean_probs[idx]*100:.1f}%{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"[error] No se encuentra el archivo: {args.audio}")

    predict(args.audio)


if __name__ == "__main__":
    main()