"""
Exploramos las claves del checkpoint de CNN14 para mapear los pesos correctamente
cuando extraigamos la features
"""

# Para que no de problemas a la hora de importar pytorch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path

ckpt_path  = Path("./checkpoints/Cnn14_16k_mAP=0.438.pth")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

state_dict = checkpoint["model"]

# Mostramos las claves y shapes de los pesos del checkpoint
print(f"Total de claves: {len(state_dict)}\n")
for k, v in state_dict.items():
    print(f"  {k:60s}  {list(v.shape)}")