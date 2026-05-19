"""
IMPORTANTE !!! Ejecutar como un modulo: python -main.main
Punto de entrada del proyecto.
Apagar USES_GPU (mlp.py, train_mlp.py, cross_validation.py) si se ejecuta train_mlp junto a otros comandos, hay conflictos de uso de GPU entre Cupy y otras libs

Punto de entrada del proyecto
Flujo:
Vemos la estructura del dataset -> Preprocesamos los audios a espectrogramas y mostramos caracteristicas del dataset -> Analizamos el checkpoint de CNN14 ->
-> Extraemos features con la capa convolucional de CNN14 -> Limpiamos los embeddings de las dimensiones muertas -> 
-> Entramos el MLP con los embeddings conseguidos -> Evaluamos el modelo -> Entrenamos un modelo KNN y otro de Regresion Logistica
para comparar rendimientos con el MLP -> Realizamos validacion cruzada del MLP
"""


def print_help():

    print("\n" + "=" * 60)
    print("        Proyecto AA Realizado por Manuel Hernando Torres")
    print("=" * 60)

    print("""
Comandos:

  dataset_structure   -> Analiza estructura del dataset
  preprocess          -> Convierte audio a espectrogramas y muestra la distribucion del dataset
  explore_checkpoint  -> Analiza el checkpoint de CNN14
  extract_features    -> Extrae embeddings con CNN14
  analyze_embeddings  -> Analiza las propiedades de los embeddings extraidos
  filter_embeddings   -> Filtra las dimensiones con varianza nula de los embeddings
  train_mlp           -> Entrena MLP
  train_knn           -> Entrena KNN
  train_rl            -> Entrena Regresion Logistica
  cross_validation    -> Validacion cruzada
  all                 -> Ejecuta pipeline completo (Hay que apagar USES_GPU antes)
  help                -> Muestra este mensaje
  exit                -> Salir

Orden:

   1. dataset_structure
   2. preprocess
   3. explore_checkpoint
   4. extract_features
   5. analyze_embeddings
   6. filter_embeddings
   7. train_mlp *
   8. train_knn
   9. train_rl
  10. cross_validation *
          
* -> Apagar USES_GPU (mlp.py, train_mlp.py, cross_validation.py) si se ejecuta junto a otros comandos, hay conflictos de uso de GPU entre Cupy y otras libs
          
Scripts extra:
    python -m scripts.predict <mi_audio.wav/mp3>  ->  Prediccion del modelo a un audio nuevo
    python -m scripts.visualize_spectrogram <familia_instrumentos o mi_audio.wav/mp3>  ->  Te ensenya como se ve un espectrograma de un audio de x instrumento o de un audio 
""")


def run_step(step: str):

    # Imports dentro de cada opcion para evitar conflictos de Cupy con otras libs que tambien usan GPU

    if step == "dataset_structure":
        from scripts.dataset_structure import run as explore_dataset_run
        explore_dataset_run()

    elif step == "preprocess":
        from scripts.preprocess import run as preprocess_run
        preprocess_run()

    elif step == "explore_checkpoint":
        from scripts.explore_checkpoint import run as explore_checkpoint_run
        explore_checkpoint_run()

    elif step == "extract_features":
        from scripts.extract_features import run as extract_features_run
        extract_features_run()

    elif step == "analyze_embeddings":
        from scripts.analyze_embeddings import run as analyze_embeddings_run
        analyze_embeddings_run()

    elif step == "filter_embeddings":
        from scripts.filter_embeddings import run as filter_embeddings_run
        filter_embeddings_run()

    elif step == "train_mlp":
        from scripts.train_mlp import run as train_mlp_run
        train_mlp_run()

    elif step == "train_knn":
        from scripts.train_knn import run as train_knn_run
        train_knn_run()

    elif step == "train_rl":
        from scripts.train_rl import run as train_rl_run
        train_rl_run()

    elif step == "cross_validation":
        from scripts.cross_validation import run as cross_validation_run
        cross_validation_run()

    elif step == "all":
        from scripts.cross_validation import run as cross_validation_run
        from scripts.train_rl import run as train_rl_run
        from scripts.train_knn import run as train_knn_run
        from scripts.train_mlp import run as train_mlp_run
        from scripts.extract_features import run as extract_features_run
        from scripts.filter_embeddings import run as filter_embeddings_run
        from scripts.explore_checkpoint import run as explore_checkpoint_run
        from scripts.preprocess import run as preprocess_run
        from scripts.dataset_structure import run as explore_dataset_run
        from scripts.analyze_embeddings import run as analyze_embeddings_run

        print("\n\n========== Viendo la estructura del dataset ==========\n\n")
        explore_dataset_run()
        print("\n\n========== Convirtiendo a espectrogramas los audios ==========\n\n")
        preprocess_run()
        print("\n\n========== Explorando el checkpoint de CNN14 ==========\n\n")
        explore_checkpoint_run()
        print("\n\n========== Creando vectores de embeddings ==========\n\n")
        extract_features_run()
        print("\n\n========== Analizando los vectores de embeddings ==========\n\n")
        analyze_embeddings_run()
        print("\n\n========== Filtrando los vectores de embeddings ==========\n\n")
        filter_embeddings_run()
        print("\n\n========== Entrenando el MLP ==========\n\n")
        train_mlp_run()
        print("\n\n========== Entrenando el modelo de KNN ==========\n\n")
        train_knn_run()
        print("\n\n========== Entrenando la Regresion Logistica ==========\n\n")
        train_rl_run()
        print("\n\n========== Iniciando validacion cruzada ==========\n\n")
        cross_validation_run()

    else:
        print("Comando desconocido. Escribe 'help'.")


def main():

    print_help()

    while True:

        cmd = input("\n(AA) > ").strip()

        if cmd == "exit":
            print("Adios...")
            break

        elif cmd == "help":
            print_help()

        elif cmd == "":
            continue

        else:
            run_step(cmd)


if __name__ == "__main__":
    main()