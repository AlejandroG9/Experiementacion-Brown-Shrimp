import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Función simplificada para aplicar Boosted Trees
def BoostedTrees(DataSet, ntree, eta, max_depth, Respuesta, Semilla, Test, ClaseDelModelo):
    X = DataSet.drop(columns=[Respuesta])
    y = DataSet[Respuesta]

    # Convertir variables categóricas en numéricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = pd.factorize(X[column])[0]

    # Convertir variable respuesta si es categórica
    if ClaseDelModelo == "Clasificacion":
        y = pd.factorize(y)[0]
        objective = 'multi:softmax' if len(np.unique(y)) > 2 else 'binary:logistic'
        num_class = len(np.unique(y)) if len(np.unique(y)) > 2 else None
    else:
        objective = 'reg:squarederror'
        num_class = None  # No se necesita para regresión

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test, random_state=Semilla)

    # Seleccionar el tipo de modelo (clasificación o regresión)
    ModelClass = XGBClassifier if ClaseDelModelo == "Clasificacion" else XGBRegressor

    # Entrenar el modelo
    model = ModelClass(
        n_estimators=ntree,
        learning_rate=eta,
        max_depth=max_depth,
        objective=objective
    ).fit(X_train, y_train)

    # Realizar predicciones y calcular métricas
    y_pred = model.predict(X_test)

    if ClaseDelModelo == "Clasificacion":  # Clasificación
        return {
            'model': model,
            'predictions': pd.Series(y_pred),
            'real_values': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred)
        }
    else:  # Regresión
        return {
            'model': model,
            'predictions': pd.Series(y_pred),
            'real_values': y_test,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

# Función para la experimentación de Boosted Trees con búsqueda manual en grid
def ExperimentacionBoostedTrees(DataSet, Respuesta, Semilla, pe, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Boosted Trees")

    BRTSerror, BRTpredic = pd.DataFrame(), pd.DataFrame()
    BRTS = {}

    # Definir la cuadrícula de parámetros manualmente
    param_grid = {
        'n_estimators': [100, 200, 300],
        'eta': [0.01, 0.05, 0.1],
        'max_depth': [None, 3, 5, 7, 10]
    }



    for i in range(1, Replicas + 1):
        for j in pe:
            # Iterar sobre todas las combinaciones de parámetros
            for n_estimators in param_grid['n_estimators']:
                for eta in param_grid['eta']:
                    for max_depth in param_grid['max_depth']:
                        print(f"Inicia Replica:{i}, P. Entrenamiento:{j}, N. Árboles:{n_estimators}, Tasa de aprendizaje:{eta}, Profundidad máxima:{max_depth}")

                        resultados = BoostedTrees(
                            DataSet=DataSet,
                            ntree=n_estimators,
                            eta=eta,
                            max_depth=max_depth,
                            Respuesta=Respuesta,
                            Semilla=Semilla,
                            Test=(1 - j),
                            ClaseDelModelo=ClaseDelModelo
                        )

                        BRTS[f"C{i}S{j}NE{n_estimators}ETA{eta}MD{max_depth}"] = resultados['model']

                        # Guardar resultados según tipo de modelo
                        error_data = {
                            'Accuracy': resultados.get('accuracy'),
                            'MAE': resultados.get('mae'),
                            'RMSE': resultados.get('rmse'),
                            'NUMERO': i,
                            'PercentTest': j,
                            'n_estimators': n_estimators,
                            'eta': eta,
                            'max_depth': max_depth
                        }

                        predic_data = {
                            'Prediccion': resultados['predictions'],
                            'Real': resultados['real_values'],
                            'NUMERO': i,
                            'PercentTest': j,
                            'n_estimators': n_estimators,
                            'eta': eta,
                            'max_depth': max_depth
                        }
                        # Guardar resultados en archivo CSV en cada iteración
                        BRTSerror.to_csv(f'{Nombre} BRTerror_log_Grid_Manual.csv', index=False)
                        print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')


                    BRTSerror = pd.concat([BRTSerror, pd.DataFrame([error_data])], ignore_index=True)
                    BRTpredic = pd.concat([BRTpredic, pd.DataFrame(predic_data)], ignore_index=True)

    return {
        'BRTSerror': BRTSerror,
        'BRTS': BRTS,
        'BRTpredic': BRTpredic
    }