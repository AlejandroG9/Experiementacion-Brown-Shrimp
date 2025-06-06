# Módulo RandomForestV2 - Documentación

## Descripción General

Este módulo implementa algoritmos de Random Forest (Bosques Aleatorios) para tareas de clasificación y regresión utilizando scikit-learn. Proporciona una interfaz completa para entrenar modelos, evaluar su rendimiento y analizar la importancia de las variables en diferentes conjuntos de datos.

## Funciones Principales

### 1. Función `RandomForest`

Esta función principal implementa el algoritmo de Random Forest con las siguientes características:

- **Preprocesamiento automático de datos**: Convierte variables categóricas a numéricas mediante codificación.
- **Soporte dual**: Maneja tanto problemas de clasificación como de regresión.
- **Evaluación integrada**: Calcula métricas de rendimiento relevantes para cada tipo de modelo.
- **Análisis de variables**: Identifica y cuantifica la importancia de cada variable predictora.

#### Parámetros:
- `DataSet`: DataFrame con los datos de entrada
- `mtry`: Número de variables a considerar en cada división (max_features)
- `ntree`: Número de árboles a construir (n_estimators)
- `Respuesta`: Nombre de la columna objetivo
- `Semilla`: Valor para reproducibilidad de resultados
- `Test`: Proporción de datos para prueba
- `ClaseDelModelo`: "Clasificacion" o "Regresion"

#### Devuelve:
Un diccionario con los siguientes elementos según el tipo de modelo:
- `model`: Modelo entrenado
- `predictions`: Predicciones generadas
- `real_values`: Valores reales para comparación
- `accuracy`/`mae`/`rmse`: Métricas de rendimiento según el tipo de modelo
- `conf_matrix`: Matriz de confusión (solo para clasificación)
- `importancia_variables`: DataFrame con la importancia de cada variable

### 2. Función `ExperimentacionRandomForest`

Esta función coordina experimentos completos con múltiples repeticiones y porcentajes de entrenamiento:

- **Evaluación robusta**: Ejecuta múltiples réplicas con diferentes divisiones de datos.
- **Registro detallado**: Almacena métricas, predicciones y valores reales para cada experimento.
- **Análisis estadístico**: Calcula la importancia promedio de variables a través de todas las réplicas.

#### Parámetros:
- `DataSet`: DataFrame con los datos completos
- `Respuesta`: Nombre de la columna objetivo
- `ntree`: Número de árboles para cada modelo
- `Semilla`: Valor para reproducibilidad
- `pe`: Lista de porcentajes de entrenamiento a probar
- `Replicas`: Número de repeticiones para cada configuración
- `ClaseDelModelo`: "Clasificacion" o "Regresion"
- `Nombre`: Identificador para los archivos de salida

#### Devuelve:
Un diccionario con:
- `RFSerror`: DataFrame con métricas de error para cada experimento
- `RFSpredic`: DataFrame con predicciones y valores reales
- `RFSvariables`: DataFrame con importancia de variables por experimento
- `importancia_promedio`: DataFrame con importancia promedio de variables

## Características Técnicas

- **Codificación automática**: Utiliza `LabelEncoder` para variables categóricas, tanto predictoras como objetivo
- **División estratificada**: Usa `train_test_split` para dividir los datos preservando la distribución
- **Métricas especializadas**: 
  - Para clasificación: precisión (accuracy) y matriz de confusión
  - Para regresión: error absoluto medio (MAE) y raíz del error cuadrático medio (RMSE)
- **Exportación de resultados**: Guarda automáticamente métricas de error en la carpeta "Errores/"
- **Análisis de importancia**: Exporta la importancia de variables a la carpeta "Variables/"

## Flujo de trabajo

1. **Preparación de datos**:
   - Las variables categóricas se convierten automáticamente a numéricas
   - Los datos se dividen en conjuntos de entrenamiento y prueba según el porcentaje especificado

2. **Entrenamiento del modelo**:
   - Se selecciona la clase de modelo adecuada (clasificador o regresor)
   - Se entrena el modelo con los hiperparámetros especificados
   - Se extraen las importancias de las variables

3. **Evaluación**:
   - Se generan predicciones sobre el conjunto de prueba
   - Se calculan métricas relevantes según el tipo de modelo
   - Se organizan los resultados para análisis posterior

4. **Experimentación**:
   - Se realizan múltiples réplicas con diferentes porcentajes de entrenamiento
   - Se agregan los resultados en DataFrames consolidados
   - Se calcula la importancia promedio de las variables

## Ejemplo de Uso

```python
# Ejemplo de uso para clasificación
resultados_clasificacion = ExperimentacionRandomForest(
    DataSet=dataset,
    Respuesta="clase",
    ntree=500,
    Semilla=123,
    pe=[0.7, 0.8],
    Replicas=30,
    ClaseDelModelo="Clasificacion",
    Nombre="Experimento1"
)

# Ejemplo de uso para regresión
resultados_regresion = ExperimentacionRandomForest(
    DataSet=dataset,
    Respuesta="CapturaTotal",
    ntree=500,
    Semilla=123,
    pe=[0.7, 0.8],
    Replicas=30,
    ClaseDelModelo="Regresion",
    Nombre="Experimento2"
)
