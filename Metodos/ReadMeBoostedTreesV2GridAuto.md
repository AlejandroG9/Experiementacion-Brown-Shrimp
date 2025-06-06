# Módulo BoostedTreesV2GridAuto - Documentación

## Descripción General

Este módulo implementa algoritmos de árboles potenciados (Boosted Trees) utilizando XGBoost con optimización automática de hiperparámetros mediante Grid Search. Proporciona funcionalidades para entrenar modelos de clasificación y regresión, evaluando sistemáticamente diferentes combinaciones de hiperparámetros para encontrar la configuración óptima.

## Funciones Principales

### 1. Función `BoostedTrees`

Esta función implementa Boosted Trees (XGBoost) con búsqueda automática de hiperparámetros:

- **Preprocesamiento de datos**: Conversión automática de variables categóricas.
- **Optimización de hiperparámetros**: Utiliza GridSearchCV para encontrar la mejor configuración.
- **Soporte dual**: Manejo automático de problemas de clasificación o regresión.
- **Evaluación integrada**: Cálculo de métricas relevantes según el tipo de problema.

#### Parámetros:
- `DataSet`: DataFrame con los datos de entrada
- `Respuesta`: Nombre de la columna objetivo
- `ntree`: Número de árboles a considerar
- `eta`: Tasa de aprendizaje inicial
- `max_depth`: Profundidad máxima inicial del árbol
- `Semilla`: Valor para reproducibilidad de resultados
- `Test`: Proporción de datos para prueba
- `ClaseDelModelo`: "Clasificacion" o "Regresion"

#### Devuelve:
Un diccionario con los siguientes elementos según el tipo de modelo:
- `model`: El mejor modelo encontrado por Grid Search
- `predictions`: Predicciones generadas
- `real_values`: Valores reales para comparación
- `accuracy`/`mae`/`rmse`: Métricas de rendimiento según el tipo de modelo
- `conf_matrix`: Matriz de confusión (solo para clasificación)

### 2. Función `ExperimentacionBoostedTrees`

Esta función coordina experimentos completos con múltiples repeticiones y porcentajes de entrenamiento:

- **Experimentación sistemática**: Ejecuta múltiples réplicas con diferentes divisiones de datos.
- **Optimización por experimento**: Cada experimento utiliza Grid Search para optimizar hiperparámetros.
- **Almacenamiento eficiente**: Guarda modelos, predicciones y métricas de error.

#### Parámetros:
- `DataSet`: DataFrame con los datos completos
- `Respuesta`: Nombre de la columna objetivo
- `Semilla`: Valor para reproducibilidad
- `pe`: Lista de porcentajes de entrenamiento a probar
- `Replicas`: Número de repeticiones para cada configuración
- `ClaseDelModelo`: "Clasificacion" o "Regresion"
- `Nombre`: Identificador para los archivos de salida

#### Devuelve:
Un diccionario con:
- `BRTSerror`: DataFrame con métricas de error para cada experimento
- `BRTS`: Diccionario con los mejores modelos para cada combinación de réplica y porcentaje
- `BRTpredic`: DataFrame con predicciones y valores reales

## Proceso de Grid Search

El módulo utiliza `GridSearchCV` de scikit-learn para realizar búsqueda automática de hiperparámetros:

1. **Definición del espacio de búsqueda**:
   ```python
   param_grid = {
       'eta': [eta],
       'max_depth': [max_depth, 5, 10],
       'n_estimators': [ntree],
       'objective': [objective],
       'eval_metric': ['mlogloss' if ClaseDelModelo == "Clasificacion" else 'rmse']
   }
   ```

2. **Configuración de la búsqueda**:
   - Utiliza validación cruzada con 5 pliegues (cv=5)
   - Métrica de evaluación adaptada al tipo de problema:
     - Para clasificación: accuracy
     - Para regresión: neg_mean_squared_error

3. **Selección automática**: El mejor modelo se selecciona en base a la puntuación de validación cruzada.

## Características Principales

- **Optimización automática**: No es necesario ajustar manualmente los hiperparámetros.
- **Exploración sistemática**: Evalúa diferentes valores de profundidad máxima automáticamente.
- **Validación cruzada**: Utiliza CV=5 para una evaluación más robusta de cada configuración.
- **Adaptabilidad**: Configura automáticamente objetivos y métricas según el tipo de problema.
- **Soporte para problemas multiclase**: Detecta y configura automáticamente problemas de clasificación binaria o multiclase.

## Diferencias con versiones anteriores

A diferencia de implementaciones anteriores:
- Incorpora Grid Search para optimización automática de hiperparámetros
- Explora sistemáticamente diferentes valores de max_depth (profundidad del árbol)
- Utiliza validación cruzada para una evaluación más robusta
- Detecta automáticamente el tipo de problema de clasificación (binario vs multiclase)

## Ejemplo de Uso
