import os
from joblib import dump, load
from torch.utils.data import Dataset
from Metodos.ProcesadorDeDatos import *
from Metodos.BoostedTreesV2GridAuto import ExperimentacionBoostedTrees
from Metodos.RandomForestV2GridAuto import ExperimentacionRandomForest
from Metodos.ProcesadorDeDatos import cargar_datos, generar_indices_experimentacion

# Paths
dir_path = "/Users/alejandrogonzalezturrubiates/Proyectos/Experimentacion_Doctorado"

# Training percentages
pe = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ntree = 500
n_trees = np.arange(100, ntree + 1, 100)  # Number of trees
replicas = 60

# Prepare datasets

archivo_v5 = "Estadisticas V5.csv"
columnas_eliminar_v5 = ['Unnamed: 0', 'Fecha.Completa', 'CafeLineaSC', 'CafeLineaCC', 'CafePacotillaSC',
                        'CafePacotillaCC', 'Mes', 'Anno',]
dataset_capt_v5 = cargar_datos(archivo=os.path.join(dir_path, archivo_v5), indices_eliminar=columnas_eliminar_v5, normalizado=False)
dataset_norm_v5 = cargar_datos(archivo=os.path.join(dir_path, archivo_v5), indices_eliminar=columnas_eliminar_v5, normalizado=True)



experimentacon = [
    {'Nombre': "Capt_V5", 'DataSet': dataset_capt_v5, 'Respuesta': 'CapturaTotal', 'ClaseDelModelo': 'Regresion'},
    {'Nombre': "Capt_V5", 'DataSet': dataset_norm_v5, 'Respuesta': 'CapturaTotal', 'ClaseDelModelo': 'Regresion'}
]

ResultadosRFST = {}
ResultadosBRTS = {}

for i in experimentacon:
    print(f'Inicia Experimentacion {i["Nombre"]}')
    # Random Forest
    resultados_random_forest = ExperimentacionRandomForest(DataSet=i['DataSet'], Respuesta=i['Respuesta'], Semilla=123, pe=pe, #ntree=ntree,
                                                           Replicas=replicas, ClaseDelModelo=i['ClaseDelModelo'], Nombre=i['Nombre'])
    dump(resultados_random_forest, os.path.join(dir_path, f'Resultados/resultados_random_forest_{i['Nombre']}.joblib'))
    experimentacon[experimentacon.index(i)]['RFST']=resultados_random_forest
    # Boosted Trees
    resultados_boosted_tree = ExperimentacionBoostedTrees(DataSet=i['DataSet'], Respuesta=i['Respuesta'], Semilla=123, pe=pe, #ntree=ntree,
                                                          Replicas=replicas, ClaseDelModelo=i['ClaseDelModelo'], Nombre=i['Nombre'])
    dump(resultados_boosted_tree, os.path.join(dir_path, f'Resultados/resultados_boosted_tree_{i['Nombre']}.joblib'))
    experimentacon[experimentacon.index(i)]['BRTS']=resultados_boosted_tree

