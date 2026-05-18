"""
Predice la familia de instrumento de un archivo de audio
Uso: python scripts/predict.py ruta/al/audio.wav
"""

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

SR         = 16000
N_FFT      = 512
HOP_LENGTH = 160
N_MELS     = 64
F_MIN      = 50.0
F_MAX      = 8000.0
DURATION   = 4.0

INSTRUMENT_LABELS = {
    0: "bass", 1: "brass", 2: "flute", 3: "guitar",
    4: "keyboard", 5: "mallet", 6: "organ", 7: "reed",
    8: "string", 9: "synth_lead", 10: "vocal",
}

CHECKPOINT   = "./checkpoints/Cnn14_16k_mAP=0.438.pth"
THETAS_PATH  = "./checkpoints/thetas_mlp.npz"
LABEL_MAP    = "./checkpoints/label_map.npy"
MEL_STATS    = "./dataset/features/mel_stats.npy"
EMB_STATS    = "./checkpoints/embedding_stats.npy"


def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def mlp_forward(thetas, x):
    a = x
    for i, theta in enumerate(thetas):
        a_bias = np.concatenate([[1.0], a])
        z = theta @ a_bias
        a = relu(z) if i < len(thetas) - 1 else softmax(z)
    return a


def audio_to_mel(segment):
    mel = librosa.feature.melspectrogram(
        y=segment, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def predict(audio_path):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Cnn14Embedder(CHECKPOINT, device).to(device)
    embedder.eval()

    data   = np.load(THETAS_PATH)
    thetas = [data[k] for k in sorted(data.files)]

    label_map   = np.load(LABEL_MAP).flatten()
    idx_to_name = {i: INSTRUMENT_LABELS[int(c)] for i, c in enumerate(label_map)}

    mel_mean, mel_std = np.load(MEL_STATS)
    emb_stats         = np.load(EMB_STATS)
    emb_mean, emb_std = emb_stats[0], emb_stats[1]

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)

    target_len = int(SR * DURATION)
    hop_len    = int(SR * 2.0)
    starts     = list(range(0, len(audio) - target_len + 1, hop_len)) or [0]

    all_probs = []
    for start in starts:
        segment = audio[start : start + target_len]
        if len(segment) < target_len:
            segment = np.pad(segment, (0, target_len - len(segment)))

        log_mel = audio_to_mel(segment)
        log_mel = ((log_mel - mel_mean) / (mel_std + 1e-6)).astype(np.float32)

        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emb = embedder(tensor.to(device)).cpu().numpy().squeeze()

        emb_n = (emb - emb_mean) / emb_std
        all_probs.append(mlp_forward(thetas, emb_n))

    mean_probs = np.mean(all_probs, axis=0)
    top3       = np.argsort(mean_probs)[::-1][:3]

    print(f"\naudio: {audio_path}")
    for i, idx in enumerate(top3):
        marker = " <--" if i == 0 else ""
        print(f"  {idx_to_name[idx]:<14} {mean_probs[idx]*100:.1f}%{marker}")

    return idx_to_name[top3[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"no encuentro el archivo: {args.audio}")

    predict(args.audio)


if __name__ == "__main__":
    main()