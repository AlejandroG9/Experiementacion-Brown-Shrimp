import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import random

# Función para cargar y procesar los datos
def cargar_datos(archivo, normalizado, indices_eliminar):
    # Leer archivo CSV
    dataset = pd.read_csv(archivo)

    # Eliminar las columnas indicadas en 'indices_eliminar'
    dataset = dataset.drop(columns=indices_eliminar)

    # Reescalar una variable específica (Profundidad)
    if 'Profundiad' in dataset.columns:
        dataset['Profundiad'] = MinMaxScaler().fit_transform(dataset[['Profundiad']])

    # Normalizar todos los datos si se indica
    if normalizado:
        for col in dataset.columns:
            if np.issubdtype(dataset[col].dtype, np.number):
                dataset[col] = MinMaxScaler().fit_transform(dataset[[col]])

    return dataset

# Función para generar índices de entrenamiento
def generar_indices(entrenamiento, dataset):
    # Calcular el tamaño de los datos de prueba (porcentaje)
    ntest = int(entrenamiento * len(dataset))
    # Generar una muestra aleatoria de índices para entrenamiento
    train = random.sample(range(len(dataset)), ntest)
    return train

# Función para generar múltiples índices para experimentación
def generar_indices_experimentacion(direccion, dataset, replicas, porcentajes_entrenamiento):
    lista_indices = {}

    # Generar índices para diferentes réplicas y porcentajes
    for i in range(1, replicas + 1):
        for j in porcentajes_entrenamiento:
            train_indices = generar_indices(entrenamiento=j, dataset=dataset)
            lista_indices[f"C{i}S{j}"] = train_indices

    # Guardar la lista de índices en un archivo JSON
    with open(f'{direccion}/Indices.json', 'w') as json_file:
        json.dump(lista_indices, json_file)

    return lista_indices

# Ejemplo de uso:
# archivo = 'ruta/datos.csv'
# normalizado = True
# indices_eliminar = ['columna_a_eliminar1', 'columna_a_eliminar2']
# dataset = cargar_datos(archivo, normalizado, indices_eliminar)
# replicas = 30
# porcentajes_entrenamiento = [0.7, 0.8]  # Ejemplo de porcentajes
# lista_indices = generar_indices_experimentacion('/ruta/direccion', dataset, replicas, porcentajes_entrenamiento)