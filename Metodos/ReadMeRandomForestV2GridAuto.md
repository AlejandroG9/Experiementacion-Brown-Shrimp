# Módulo RandomForestV2GridAuto - Documentación

## Descripción General

Este módulo implementa algoritmos de Random Forest (Bosques Aleatorios) con optimización automática de hiperparámetros mediante Grid Search. Proporciona funcionalidades para entrenar modelos de clasificación y regresión, evaluando sistemáticamente diferentes combinaciones de hiperparámetros para encontrar la configuración óptima.

## Funciones Principales

### 1. Función `RandomForest`

Esta función implementa Random Forest con búsqueda automática de hiperparámetros:

- **Preprocesamiento de datos**: Conversión automática de variables categóricas.
- **Optimización de hiperparámetros**: Utiliza GridSearchCV para encontrar la mejor configuración.
- **Soporte dual**: Manejo automático de problemas de clasificación o regresión.
- **Evaluación integrada**: Cálculo de métricas relevantes según el tipo de problema.

#### Parámetros:
- `DataSet`: DataFrame con los datos de entrada
- `mtry`: Número inicial de variables a considerar en cada división (max_features)
- `ntree`: Número inicial de árboles a construir (n_estimators)
- `Respuesta`: Nombre de la columna objetivo
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

### 2. Función `ExperimentacionRandomForest`

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
- `RFSerror`: DataFrame con métricas de error para cada experimento
- `RFS`: Diccionario con los mejores modelos para cada combinación de réplica y porcentaje
- `RFSpredic`: DataFrame con predicciones y valores reales

## Proceso de Grid Search

El módulo utiliza `GridSearchCV` de scikit-learn para realizar búsqueda automática de hiperparámetros:

1. **Definición del espacio de búsqueda**:
   ```python
   param_grid = {
       'max_features': [mtry],
       'n_estimators': [ntree],
       'min_samples_split': [2, 5, 10],
       'max_depth': [None, 10, 20, 30]
   }
   ```

2. **Configuración de la búsqueda**:
   - Validación cruzada con 5 pliegues (cv=5)
   - Métrica de evaluación adaptada al tipo de problema:
     - Para clasificación: accuracy
     - Para regresión: neg_mean_squared_error

3. **Selección automática**: El mejor modelo se selecciona en base a la puntuación de validación cruzada.

## Características Clave

- **Optimización automática**: Ajusta hiperparámetros clave sin intervención manual:
  - `min_samples_split`: Número mínimo de muestras requeridas para dividir un nodo
  - `max_depth`: Profundidad máxima de los árboles (None para crecimiento sin límite)

- **Preprocesamiento integrado**:
  - Codificación automática de variables categóricas mediante LabelEncoder
  - Transformación automática de la variable objetivo si es categórica

- **Validación cruzada**: Utiliza CV=5 para una evaluación más robusta de cada configuración

- **Métricas específicas**:
  - Para clasificación: precisión (accuracy) y matriz de confusión
  - Para regresión: error absoluto medio (MAE) y raíz del error cuadrático medio (RMSE)

## Diferencias con la versión estándar

A diferencia de la implementación básica de Random Forest:

1. **Búsqueda automática** de hiperparámetros mediante Grid Search
2. **Exploración de múltiples valores** para `min_samples_split` y `max_depth`
3. **Selección basada en validación cruzada** para mayor robustez
4. **Optimización específica** según el tipo de problema (clasificación/regresión)

## Flujo de trabajo

1. **Preparación de datos**:
   - División del conjunto de datos en características (X) y variable objetivo (y)
   - Codificación de variables categóricas
   - División en conjuntos de entrenamiento y prueba

2. **Optimización de hiperparámetros**:
   - Definición del espacio de búsqueda
   - Configuración de Grid Search con validación cruzada
   - Entrenamiento y selección del mejor modelo

3. **Evaluación del modelo**:
   - Generación de predicciones en el conjunto de prueba
   - Cálculo de métricas de rendimiento específicas
   - Retorno de resultados completos

4. **Experimentación sistemática**:
   - Ejecución de múltiples réplicas con diferentes semillas
   - Prueba con distintos porcentajes de entrenamiento/prueba
   - Almacenamiento de resultados para análisis posterior

## Ejemplo de Uso
