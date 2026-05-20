# SaraswAI – Instrument Classification con Transfer Learning

Este proyecto clasifica familias de instrumentos musicales a partir de audio utilizando:

- Espectrogramas Mel
- CNN14 (PANNs) como extractor de features
- MLP implementada en NumPy/CuPy como clasificador final

---

# Dataset

Se utiliza el dataset:

https://www.kaggle.com/datasets/dmytrotiapukhin/nsynth-small

Debe colocarse en la raíz del proyecto con la siguiente estructura:


dataset/

    test_ds/
        9554115882362711283/
        dataset_spec.pb
        snapshot.metadata

    train_ds/
        1859930310722468987/
        dataset_spec.pb
        snapshot.metadata

---

# Embedder

Para extraer los embeddings de los espectrogramas de Mel he usado las capas convolucionales del PANN CNN14.
Para el funcionamiento del modelo es necesario poner el checkpoint del Cnn14_16khz en checkpoints/

Puedes descargarlo desde aquí:

https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1

---

# Dependencias

El proyecto hace uso de las siguientes librerías:
numpy -> Manejo de arrays y operaciones matemáticas, además de guardado de datos
matplotlib -> Representación visual de los resultados
scikit-learn -> Modelos de KNN y Regresión Logística además de otras herramientas como la confección de los folds en cross_validation
tensorflow -> Carga del conjunto de datos 
torch -> Embedder 
torchaudio -> Utilidades del embedder
tqdm -> Barras de carga en extracción de embeddings
librosa -> Procesado de audio para la creación de los espectrogramas
cupy-cuda12x -> Uso de GPU para entrenar el MLP (opcional)

---

# Instalación

Instalar dependencias:

pip install -r requirements.txt

CuPy es opcional y solo necesario si se quiere usar GPU para el entrenamiento del MLP.

---

# Uso

Como punto de entrada principal está python -m main.main en el que puedes ejecutar todas las acciones necesarias sobre los modelos.
Todos los scripts deben ejecutarse como módulos desde la raíz del proyecto.
Se permite usar los vectores de embeddings con todas las dimensiones (2048) desactivando USE_FILTERED en:

- cross_validation.py
- train_mlp.py

---

# Uso de GPU (CuPy)

Se puede activar el uso de GPU modificando:

USES_GPU = True

en los archivos:

- mlp.py
- train_mlp.py
- cross_validation.py

Advertencias:

- CuPy y NumPy no son completamente compatibles en comportamiento aleatorio así que los resultados pueden variar
- Si se usa GPU, se recomienda ejecutar únicamente el script de entrenamiento del MLP de forma aislada

---

# Scripts extra
- python -m scripts.visualize_spectrogram <instrument_class o archivo_de_audio> -> Visualizar el espectrograma normalizado de un elemento aleatorio de una clase concreta o de un archivo de audio
- python -m scripts.predict <archivo_de_audio> -> Predicción del modelo de MLP sobre un archivo de audio

---
