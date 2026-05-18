Link del dataset usado: https://www.kaggle.com/datasets/dmytrotiapukhin/nsynth-small
Poner en la carpeta dataset, que quede así:
    dataset
        test_ds
            9554115882362711283
            dataset_spec.pb
            snapshot.metadata
        train_ds
            1859930310722468987
            dataset_spec.pb
            snapshot.metadata

Invocar desde la raíz el punto de entrada como: python -m main.main

Puedes descargar las dependencias con pip install requirements.txt
Cupy no es obligatorio instalarlo si no vas a usar GPU para entrenar el MLP.

Existe la opción de usar GPU para entrenar el MLP, poniendo USES_GPU a True en train_mlp.py y en mlp.py, lo que provoca que se use Cupy en lugar de Numpy.
Cuidado porque Cupy se rompe si se usa train_mlp después de haber hecho otra accion que también pueda usar GPU.
Lo mejor, si se quiere usar GPU, es entrenar el MLP invocando el script individual con python -m scripts.train_ml
Cupy y Numpy no tienen exactamente la misma implementación de uniform así que se pueden esperar resultados distintos usando GPU o CPU.

Siempre invocar los scripts como módulos desde la raíz.

Se puede probar a usar el script de predicción: python -m scripts.predict mi_audio.wav/mp3
