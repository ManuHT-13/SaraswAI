"""
Predice la familia de instrumento de un archivo de audio

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import librosa

from models.embedder import Cnn14Embedder

# Parametros para el espectrograma de Mel
SR         = 16000      # Sample rate (frecuencia de muestreo)
N_FFT      = 512        # Tamanyo de la ventana para la transformada de fourier rapida
HOP_LENGTH = 160        # Muestras realizadas en cada ventana
N_MELS     = 64         # Numero de bandas de frecuencia
F_MIN      = 50.0       # Frecuencia minima del espetrograma
F_MAX      = 8000.0     # Frecuencia maxima del espectrograma
DURATION   = 4.0        # Duracion de los audios de nsynth

# Mapeo oficial de nombres (Mantenlo igual para mostrar el resultado al usuario)
INSTRUMENT_LABELS = {0: "bass", 1: "brass", 2: "flute", 3: "guitar", 4: "keyboard", 5: "mallet", 6: "organ", 7: "reed", 8: "string", 9: "synth_lead", 10: "vocal"}

CNN14_CHECKPOINT = "./checkpoints/Cnn14_16k_mAP=0.438.pth"
THETAS_PATH      = "./checkpoints/thetas_mlp.npz"
LABEL_MAP        = "./checkpoints/label_map.npy"
EMB_STATS        = "./checkpoints/embedding_stats.npy"
ACTIVE_DIMS_MASK = "./checkpoints/active_dims_mask.npy"
MEL_STATS        = './dataset/features/mel_stats.npy'



def relu(z):
    """
    Como funcion de activacion he decidido usar ReLu en lugar de Sigmoid por
    la alta dimensionalidad de los vectores de featurings y su gran magnitud
    """
    return np.maximum(0, z)

def softmax(z):
    """
    Funcion de activacion de la capa de salida normalizando las
    salidas como probabilidades que suman 1
    """
    # Restamos el maximo para estabilidad numerica porque si no da problemas
    z_stable = z - np.max(z)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)

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

    # Cargamos el mapa de etiquetas
    label_map = np.load(LABEL_MAP).flatten()
    idx_to_name = {i: INSTRUMENT_LABELS[int(c)] for i, c in enumerate(label_map)}

    # Cargamos los stats de normalizacion de los espectrogramas de mel
    mel_stats = np.load(MEL_STATS)
    mel_mean, mel_std = float(mel_stats[0]), float(mel_stats[1])
    
    emb_stats = np.load(EMB_STATS).astype(np.float32)
    emb_mean, emb_std = emb_stats[0], emb_stats[1]

    mask = np.load(ACTIVE_DIMS_MASK) if Path(ACTIVE_DIMS_MASK).exists() else None

    # Cargamos el audio objetivo
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
        log_mel = (log_mel - mel_mean) / (mel_std + 1e-6)
        
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            emb = embedder(tensor.to(device)).cpu().numpy().squeeze()

        if mask is not None:
            emb = emb[mask]

        # Normalizamos el vector de características devuelto
        emb_n = (emb - emb_mean) / (emb_std + 1e-8)

        # Inferimos las clases que predice el mlp
        all_probs.append(mlp_forward(thetas, emb_n))

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
        print(f"[error] No se encuentra el archivo: {args.audio}")
        sys.exit()

    predict(args.audio)


if __name__ == "__main__":
    main()