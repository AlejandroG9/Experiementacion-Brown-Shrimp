import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor

# Función para aplicar Boosted Trees con Grid Search
def BoostedTrees(DataSet, Respuesta, ntree, eta, max_depth, Semilla, Test, ClaseDelModelo):
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

    # Definir el tipo de modelo
    ModelClass = XGBClassifier if ClaseDelModelo == "Clasificacion" else XGBRegressor

    # Configuración de hiperparámetros para Grid Search
    param_grid = {
        'eta': [eta],
        'max_depth': [max_depth, 5, 10],
        'n_estimators': [300,400,500,600],
        'objective': [objective],
        'eval_metric': ['mlogloss' if ClaseDelModelo == "Clasificacion" else 'rmse']
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
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

# Función para la experimentación de Boosted Trees
def ExperimentacionBoostedTrees(DataSet, Respuesta, Semilla, pe, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Boosted Trees")

    BRTSerror, BRTpredic = pd.DataFrame(), pd.DataFrame()
    BRTS = {}

    for i in range(1, Replicas + 1):
        for j in pe:
            print(f"Inicia Replica:{i}, P. Entrenamiento:{j}")

            resultados = BoostedTrees(DataSet=DataSet, ntree=500, eta=0.05, max_depth=5, Respuesta=Respuesta, Semilla=Semilla, Test=(1-j), ClaseDelModelo=ClaseDelModelo)

            BRTS[f"C{i}S{j}"] = resultados['model']

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
            # Guardar resultados en archivo CSV en cada iteración
            BRTSerror.to_csv(f'{Nombre} BRTerror_log_Grid_Auto.csv', index=False)
            print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')

            BRTSerror = pd.concat([BRTSerror, pd.DataFrame([error_data])], ignore_index=True)
            BRTpredic = pd.concat([BRTpredic, pd.DataFrame(predic_data)], ignore_index=True)

    return {
        'BRTSerror': BRTSerror,
        'BRTS': BRTS,
        'BRTpredic': BRTpredic
    }