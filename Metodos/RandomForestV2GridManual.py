import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Función simplificada para aplicar Random Forest
def RandomForest(DataSet, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, Respuesta, Semilla, Test, ClaseDelModelo):
    X = DataSet.drop(columns=[Respuesta])
    y = DataSet[Respuesta]

    # Convertir variables categóricas en numéricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Convertir variable respuesta si es categórica
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)






    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test, random_state=Semilla)

    # Seleccionar el tipo de modelo (clasificación o regresión)
    ModelClass = RandomForestClassifier if ClaseDelModelo == "Clasificacion" else RandomForestRegressor
    print(ModelClass)
    model = ModelClass(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    ).fit(X_train, y_train)

    # Realizar predicciones y calcular métricas
    y_pred = model.predict(X_test)
    #print("Y PRED")
    #print(y_pred)
    y_pred_series = pd.Series(y_pred)
    y_test_series = pd.Series(y_test)
    if ClaseDelModelo == "Clasificacion":  # Clasificación
        return {
            'model': model,
            'predictions': y_pred_series,
            'real_values': y_test_series.values,
            'accuracy': accuracy_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }
    else:  # Regresión
        return {
            'model': model,
            'predictions': y_pred,
            'real_values': y_test.values,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False)
        }

# Función para la experimentación de Random Forest
def ExperimentacionRandomForest(DataSet, Respuesta, Semilla, pe, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Random Forest")

    RFSerror, RFSpredic = pd.DataFrame(), pd.DataFrame()
    RFS = {}

    # Definir la cuadrícula de parámetros
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]  # Cambié 'auto' a None
    }

    for i in range(1, Replicas + 1):
        for j in pe:
            # Iterar sobre todas las combinaciones de parámetros
            for n_estimators in param_grid['n_estimators']:
                for max_depth in param_grid['max_depth']:
                    for min_samples_split in param_grid['min_samples_split']:
                        for min_samples_leaf in param_grid['min_samples_leaf']:
                            for max_features in param_grid['max_features']:
                                print(f"Inicia Replica: {i}, P. Entrnamiento: {j}, N. Arboles: {n_estimators}, Max. Produndidad: {max_depth}, Min. de M. Nodo: {min_samples_split}, Min. de M. Hoja: {min_samples_leaf}, Num. Caracteristicas: {max_features}")
                                resultados = RandomForest(
                                    DataSet=DataSet,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    Respuesta=Respuesta,
                                    Semilla=Semilla,
                                    Test=(1 - j),
                                    ClaseDelModelo=ClaseDelModelo
                                )

                                RFS[f"C{i}S{j}NE{n_estimators}MD{max_depth}mss{min_samples_split}MSL{min_samples_leaf}MF{max_features}"] = resultados['model']

                                # Guardar resultados según tipo de modelo
                                error_data = {
                                    'Accuracy': resultados.get('accuracy'),
                                    'MAE': resultados.get('mae'),
                                    'RMSE': resultados.get('rmse'),
                                    'NUMERO': i,
                                    'PercentTest': j,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features
                                }

                                predic_data = {
                                    'Prediccion': resultados['predictions'],
                                    'Real': resultados['real_values'],
                                    'NUMERO': i,
                                    'PercentTest': j,
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features
                                }
                                RFSerror.to_csv(f'{Nombre} RFSerror_log_Grid_Manual.csv', index=False)
                                print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')

                                RFSerror = pd.concat([RFSerror, pd.DataFrame([error_data])], ignore_index=True)
                                RFSpredic = pd.concat([RFSpredic, pd.DataFrame(predic_data)], ignore_index=True)

    return {
        'RFSerror': RFSerror,
        'RFS': RFS,
        'RFSpredic': RFSpredic
    }