"""
Exploramos el dataset en formato tensorflow recursivamente para conocer bien los datos de cada muestra
"""

import tensorflow as tf
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "train_ds"

# Cargamos el dataset guardado en disco porque anteriormente habia sido guardado con experimental.save
data = tf.data.Dataset.load(str(DATASET_PATH))


# Funcion recursiva para leer el dataset de tensor flow
def info_sample(d, indent=0):
    prefijo = "  " * indent

    if isinstance(d, dict):
        for k, v in d.items():

            if isinstance(v, dict):
                print(f"{prefijo}{k}: (dict)")
                info_sample(v, indent + 1)

            else:
                print(f"{prefijo}{k}: shape={v.shape}  type={v.dtype}  valor={v.numpy()}")

    else:
        print(f"{prefijo}shape={d.shape}  type={d.dtype}")


# Imprimimos la estructura de un sample
for sample in data.take(1):
    info_sample(sample)