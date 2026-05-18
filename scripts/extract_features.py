"""
Pasa los espectrogramas de Mel por la capa convolucional de CNN14 para extraer vectores de features
de 2048 dimensiones y los guarda listos para entrenar la MLP
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.embedder import Cnn14Embedder


# Tenemos que crear un wrapper que extienda Dataset para cargar los datos en pytorch de forma natural
class MelDataset(Dataset):
    """
    Clase para cargar los datos de nuestro dataset a pytorch
    IMPORTANTE: La normalizacion de z-score la he hecho directamente en el metodo de getitem
    """
    def __init__(self, csv_path, mel_mean, mel_std):
        self.df   = pd.read_csv(csv_path)
        self.mean = mel_mean
        self.std  = mel_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        mel    = np.load(row["npy_path"])
        # Normalizamos el vector que contiene el espectrograma ya que Cnn14 espera valores normalizados
        mel    = (mel - self.mean) / (self.std + 1e-6)
        tensor = torch.from_numpy(mel).unsqueeze(0)
        label  = int(row["family_id"])
        return tensor, label


def dataset_mean_std(csv_path):
    """
    Devolvemos la media y la desviacion estandar de los espectrogramas
    de Mel para poder normalizar los datos antes de extraer las features
    """
    df = pd.read_csv(csv_path)
    all_mean, all_std = [], []

    for path in tqdm(df["npy_path"], desc="calculando stats"):
        mel = np.load(path)
        all_mean.append(mel.mean())
        all_std.append(mel.std())

    mean = float(np.mean(all_mean))
    std  = float(np.mean(all_std))
    print(f"media={mean:.4f}  desviacion estandar={std:.4f}")

    return mean, std


# Desactivamos el calculo de gradiente porque solo queremos inferir y extraer los features, no entrenar
@torch.no_grad()
def extract_features_split(model, csv_path, out_emb, out_labels, mel_mean, mel_std, batch_size, device):
    """
    Extraemos las features de uno de los splits (train o test)
    """
    dataset = MelDataset(csv_path, mel_mean, mel_std)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings, labels_out = [], []
    for mels, labels in tqdm(loader, desc=csv_path.stem):
        emb = model(mels.to(device))
        embeddings.append(emb.cpu().numpy())
        labels_out.append(labels.numpy())

    # Concatenamos todos los batches en un solo array y lo guardamos junto a las labels
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    labels_out = np.concatenate(labels_out, axis=0).astype(np.int64)
    np.save(out_emb, embeddings)
    np.save(out_labels, labels_out)

    print(f"{out_emb.name}: {embeddings.shape}")


def run():
    parser = argparse.ArgumentParser()
    # Argumento para el tamanyo del batch
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path("./dataset/processed")
    out_dir  = Path("./dataset/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intentamos usar gpu para que sea mas rapido
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargamos el modelo y lo ponemos en modo evaluacion porque solo vamos a inferir
    model = Cnn14Embedder("./checkpoints/Cnn14_16k_mAP=0.438.pth", device).to(device)
    model.eval()

    train_csv = data_dir / "train_labels.csv"
    test_csv  = data_dir / "test_labels.csv"

    mel_mean, mel_std = dataset_mean_std(train_csv)

    for split, csv_path in [("train", train_csv), ("test", test_csv)]:
        if not csv_path.exists():
            print(f"[error] No se encuentra: {csv_path}")
            continue
        extract_features_split(model, csv_path, out_dir / f"{split}_embeddings.npy", out_dir / f"{split}_labels.npy", mel_mean, mel_std, args.batch_size, device)

    np.save(out_dir / "mel_stats.npy", np.array([mel_mean, mel_std]))
    print("Features extraidas")


if __name__ == "__main__":
    run()