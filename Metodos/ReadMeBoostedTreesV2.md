# Módulo BoostedTreesV2 - Documentación

## Descripción General

Este módulo implementa algoritmos de árboles potenciados (Boosted Trees) utilizando XGBoost para tareas de clasificación y regresión. Proporciona una interfaz completa para entrenar modelos, evaluar su rendimiento y analizar la importancia de las variables en diferentes conjuntos de datos.

## Funciones Principales

### 1. Función `BoostedTrees`

Esta función principal implementa el algoritmo de árboles potenciados con las siguientes características:

- **Preprocesamiento automático de datos**: Convierte variables categóricas a numéricas mediante codificación.
- **Soporte para clasificación y regresión**: Determina automáticamente el tipo de problema basado en el parámetro `ClaseDelModelo`.
- **Entrenamiento optimizado**: Utiliza la API de bajo nivel de XGBoost para un control preciso del proceso de entrenamiento.
- **Análisis de importancia de variables**: Calcula y reporta la importancia de cada variable predictora.

#### Parámetros:
- `DataSet`: DataFrame con los datos de entrada
- `mtry`: Número de variables a considerar en cada división
- `ntree`: Número de árboles a construir
- `Respuesta`: Nombre de la columna objetivo
- `Semilla`: Valor para reproducibilidad de resultados
- `Test`: Proporción de datos para prueba
- `ClaseDelModelo`: "Clasificacion" o "Regresion"

#### Devuelve:
Un diccionario con los siguientes elementos según el tipo de modelo:
- Modelo entrenado
- Predicciones
- Valores reales
- Métricas de rendimiento (precisión/MAE/RMSE)
- Matriz de confusión (para clasificación)
- Importancia de variables

### 2. Función `ExperimentacionBoostedTrees`

Esta función coordina experimentos completos con múltiples repeticiones y porcentajes de entrenamiento:

- **Experimentación sistemática**: Ejecuta múltiples réplicas con diferentes porcentajes de datos de entrenamiento.
- **Registro de resultados**: Almacena métricas de error, predicciones y valores reales por cada experimento.
- **Análisis de variables**: Calcula y guarda la importancia promedio de cada variable predictora.

#### Parámetros:
- `DataSet`: DataFrame con los datos completos
- `Respuesta`: Nombre de la columna objetivo
- `Semilla`: Valor para reproducibilidad
- `pe`: Lista de porcentajes de entrenamiento a probar
- `ntree`: Número de árboles
- `Replicas`: Número de repeticiones para cada configuración
- `ClaseDelModelo`: "Clasificacion" o "Regresion"
- `Nombre`: Identificador para los archivos de salida

#### Devuelve:
Un diccionario con:
- `BRTSerror`: DataFrame con métricas de error para cada experimento
- `BRTpredic`: DataFrame con predicciones y valores reales
- `BRTSvariables`: DataFrame con importancia de variables por experimento
- `importancia_promedio`: DataFrame con importancia promedio de variables

## Características Técnicas

- **Manejo de datos categóricos**: Conversión automática mediante `LabelEncoder`
- **Configuración personalizable**: Los parámetros como tasa de aprendizaje (eta=0.05) se pueden ajustar dentro de la función
- **Guardar resultados**: Exporta métricas de error a CSV durante la ejecución en la carpeta "Errores/"
- **Análisis de variables**: Exporta importancia de variables a la carpeta "Variables/"

## Flujo de trabajo

1. **Preparación de datos**:
   - Las variables categóricas se convierten automáticamente a numéricas
   - La variable respuesta se codifica si es necesario
   - Los datos se dividen en conjuntos de entrenamiento y prueba

2. **Entrenamiento del modelo**:
   - Se configura el objetivo según el tipo de problema (clasificación/regresión)
   - Se entrena el modelo XGBoost con los parámetros especificados
   - Se extraen las importancias de las variables

3. **Evaluación**:
   - Para clasificación: se calcula precisión y matriz de confusión
   - Para regresión: se calculan MAE y RMSE
   - Se guardan los resultados para cada experimento

4. **Análisis de variables**:
   - Se promedia la importancia de las variables a través de todas las réplicas
   - Se exportan los resultados a archivos CSV para análisis posterior

## Ejemplo de Uso
