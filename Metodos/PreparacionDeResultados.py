import pickle
import os

def guardar_errores_pickle():
    direccion = "/EntornodeExperimentacion"
    carpeta = "Resultados"
    carpeta_error = "Errores 2"

    # Obtener la lista de archivos en la carpeta de resultados
    lista_archivos = os.listdir(os.path.join(direccion, carpeta))
    lista_nombre = [archivo[:-4] for archivo in lista_archivos]  # Eliminar la extensión del nombre

    for i, archivo in enumerate(lista_archivos):
        # Leer archivo directamente si ya no es necesario usar .RDS (ejemplo de datos simulados)
        temporal = {"error": f"Datos simulados {i}"}  # Datos simulados, reemplázalo con tus datos reales
        error = temporal["error"]

        # Guardar errores en archivo pickle
        with open(os.path.join(direccion, carpeta_error, f"Error_{lista_nombre[i]}.pkl"), 'wb') as pickle_file:
            pickle.dump(error, pickle_file)

        print(f"Se guardó el archivo {lista_nombre[i]}")

def guardar_resultados_pickle():
    direccion = "/EntornodeExperimentacion"
    carpeta = "Resultados"
    carpeta_resultados = "Resultados Prediccion contra Real"

    # Obtener la lista de archivos en la carpeta de resultados
    lista_archivos = os.listdir(os.path.join(direccion, carpeta))
    lista_nombre = [archivo[:-4] for archivo in lista_archivos]  # Eliminar la extensión del nombre

    for i, archivo in enumerate(lista_archivos):
        # Leer archivo directamente si ya no es necesario usar .RDS (ejemplo de datos simulados)
        temporal = {"predreal": f"Predicción simulada {i}"}  # Datos simulados, reemplázalo con tus datos reales
        predreal = temporal["predreal"]

        # Guardar predicciones y reales en archivo pickle
        with open(os.path.join(direccion, carpeta_resultados, f"Prediccion_{lista_nombre[i]}.pkl"), 'wb') as pickle_file:
            pickle.dump(predreal, pickle_file)

        print(f"Se guardó el archivo {lista_nombre[i]}")

# Ejecutar las funciones
guardar_errores_pickle()
guardar_resultados_pickle()