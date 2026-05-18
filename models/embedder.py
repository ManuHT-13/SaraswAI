"""
Embedder para extraer las features de los espectrogramas de Mel usando la capa convolucional de CNN14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from panns.models import Cnn14_16k

# Parametros para el embedder, tenemos que ponerle los mismos parametros con los que hemos creado los espectrogramas
SR         = 16000      # Sample rate (frecuencia de muestreo)
N_FFT      = 512        # Tamanyo de la ventana para la transformada de fourier rapida
HOP_LENGTH = 160        # Muestras realizadas en cada ventana
N_MELS     = 64         # Numero de bandas de frecuencia
F_MIN      = 50.0       # Frecuencia minima del espetrograma
F_MAX      = 8000.0     # Frecuencia maxima del espectrograma


class Cnn14Embedder(nn.Module):
    """
    Clase de embedding usando los bloques convolucionales del modelo Cnn14
    """
    # Cargamos CNN14 y nos quedamos solo con el embedding, quitnadonos el extractor de espectrograma
    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.cnn = Cnn14_16k(sample_rate=SR, window_size=N_FFT, hop_size=HOP_LENGTH, mel_bins=N_MELS, fmin=F_MIN, fmax=F_MAX, classes_num=527)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.cnn.load_state_dict(checkpoint["model"])
        self.cnn.eval()

    def forward(self, x):
        """
        Replicamos el forward que realiza CNN14, pasando la entrada por 6 bloques convolucionales
        y aplicando una funcion de activacion reLu
        """
        cnn = self.cnn
        x = x.transpose(2, 3)
        x = x.transpose(1, 3)
        x = cnn.bn0(x)
        x = x.transpose(1, 3)

        # Tenemos que especificar que estamos realizando una transformacion puramente, no entrenando los pesos
        # de las capas convolucionales
        x = cnn.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = cnn.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = cnn.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = cnn.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = cnn.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = cnn.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        return F.relu_(cnn.fc1(x))
