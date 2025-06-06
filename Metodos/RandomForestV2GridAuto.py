import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Función para aplicar Random Forest con Grid Search
def RandomForest(DataSet, mtry, Respuesta, Semilla, Test, ClaseDelModelo):
    X = DataSet.drop(columns=[Respuesta])
    y = DataSet[Respuesta]

    # Convertir variables categóricas en numéricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Convertir variable respuesta si es categórica
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test, random_state=Semilla)

    # Definir el tipo de modelo
    ModelClass = RandomForestClassifier if ClaseDelModelo == "Clasificacion" else RandomForestRegressor

    # Configuración de hiperparámetros para Grid Search
    param_grid = {
        'max_features': [mtry],
        'n_estimators': [300,400,500,600],
        'min_samples_split': [2, 5, 10],  # Ejemplo de hiperparámetro
        'max_depth': [None, 10, 20, 30]   # Ejemplo de hiperparámetro
    }

    # Realizar búsqueda en grid
    grid_search = GridSearchCV(ModelClass(), param_grid, cv=5, scoring='accuracy' if ClaseDelModelo == "Clasificacion" else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_

    # Realizar predicciones
    y_pred = best_model.predict(X_test)

    if ClaseDelModelo == "Clasificacion":  # Clasificación
        return {
            'model': best_model,
            'predictions': y_pred,
            'real_values': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }
    else:  # Regresión
        return {
            'model': best_model,
            'predictions': y_pred,
            'real_values': y_test,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred),#, squared=False)
        }

# Función para la experimentación de Random Forest
def ExperimentacionRandomForest(DataSet, Respuesta, Semilla, pe, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Random Forest")

    RFSerror, RFSpredic = pd.DataFrame(), pd.DataFrame()
    RFS = {}

    for i in range(1, Replicas + 1):
        for j in pe:
            print(f"Inicia Replica: {i}, P. Entrnamiento: {j}")

            resultados = RandomForest(DataSet=DataSet, mtry=4, Respuesta=Respuesta, Semilla=Semilla, Test=(1-j), ClaseDelModelo=ClaseDelModelo)

            RFS[f"C{i}S{j}"] = resultados['model']

            # Guardar resultados
            error_data = {
                'Accuracy': resultados.get('accuracy'),
                'MAE': resultados.get('mae'),
                'RMSE': resultados.get('rmse'),
                'NUMERO': i,
                'PercentTest': j
            }

            predic_data = {
                'Prediccion': resultados['predictions'],
                'Real': resultados['real_values'],
                'NUMERO': i,
                'PercentTest': j
            }
            RFSerror.to_csv(f'{Nombre}_{ClaseDelModelo}_RFSerror_log_Grid_Auto.csv', index=False)
            print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')

            RFSerror = pd.concat([RFSerror, pd.DataFrame([error_data])], ignore_index=True)
            RFSpredic = pd.concat([RFSpredic, pd.DataFrame(predic_data)], ignore_index=True)

    return {
        'RFSerror': RFSerror,
        'RFS': RFS,
        'RFSpredic': RFSpredic
    }