import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor, DMatrix, train as xgb_train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Función adaptada para aplicar Boosted Trees
def BoostedTrees(DataSet, mtry, ntree, Respuesta, Semilla, Test, ClaseDelModelo):
    # Identificar y convertir variables categóricas en numéricas
    for column in DataSet.select_dtypes(include='object').columns:
        DataSet[column] = LabelEncoder().fit_transform(DataSet[column])

    # Verificar si la variable de respuesta es categórica
    y = DataSet[Respuesta]
    if ClaseDelModelo == "Clasificacion":
        y = LabelEncoder().fit_transform(y)
        objective = 'multi:softmax'
    else:
        objective = 'reg:squarederror'

    # Crear la matriz de características
    X = pd.get_dummies(DataSet.drop(columns=[Respuesta]), drop_first=True)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test, random_state=Semilla)

    # Configurar parámetros
    params = {
        'objective': objective,
        'eta': 0.05,
        #'max_depth': 5,
        'eval_metric': 'mlogloss' if ClaseDelModelo == 'Clasificacion' else 'rmse',
        'num_class': len(np.unique(y)) if ClaseDelModelo == 'Clasificacion' else None
    }

    dtrain = DMatrix(data=X_train, label=y_train)
    dtest = DMatrix(data=X_test, label=y_test)

    # Entrenar el modelo
    bst_model = xgb_train(params, dtrain, num_boost_round=ntree)

    # Extraer la importancia de las variables
    importancias = bst_model.get_score(importance_type='gain')
    importancia_variables = pd.DataFrame({
        'Variable': importancias.keys(),
        'Importancia': importancias.values()
    }).sort_values(by='Importancia', ascending=False)

    # Generar predicciones
    BRT_pred = bst_model.predict(dtest)

    if ClaseDelModelo == 'Clasificacion':
        accuracy = accuracy_score(y_test, BRT_pred)
        conf_matrix = confusion_matrix(y_test, BRT_pred)
        return {
            'model': bst_model,
            'predictions': BRT_pred,
            'real_values': y_test,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'importancia_variables': importancia_variables
        }
    else:
        BRT_MAE = mean_absolute_error(y_test, BRT_pred)
        BRT_RMSE = np.sqrt(mean_squared_error(y_test, BRT_pred))
        return {
            'model': bst_model,
            'predictions': BRT_pred,
            'real_values': y_test,
            'mae': BRT_MAE,
            'rmse': BRT_RMSE,
            'importancia_variables': importancia_variables
        }

# Función para la experimentación de Boosted Trees
def ExperimentacionBoostedTrees(DataSet, Respuesta, Semilla, pe, ntree, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Boosted Trees")

    BRTSerror, BRTpredic = pd.DataFrame(), pd.DataFrame()
    BRTS = {}
    BRTSvariables = pd.DataFrame()

    for i in range(1, Replicas + 1):
        for j in pe:
            resultados = BoostedTrees(DataSet=DataSet, mtry=4, ntree=ntree, Respuesta=Respuesta, Semilla=Semilla, Test=(1-j), ClaseDelModelo=ClaseDelModelo)

            #BRTS[f"C{i}S{j}"] = resultados['model']
            importancias_replica = resultados['importancia_variables']
            importancias_replica['Réplica'] = i
            importancias_replica['PercentTest'] = j
            BRTSvariables = pd.concat([BRTSvariables, importancias_replica], ignore_index=True)

            # Guardar resultados según tipo de modelo
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
            BRTSerror.to_csv(f'Errores/{Nombre}_{ClaseDelModelo}_BRTerror_log.csv', index=False)
            print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')

            BRTSerror = pd.concat([BRTSerror, pd.DataFrame([error_data])], ignore_index=True)
            BRTpredic = pd.concat([BRTpredic, pd.DataFrame(predic_data)], ignore_index=True)

    # Calcular la importancia promedio de las variables
    importancia_promedio = BRTSvariables.groupby('Variable').agg({
        'Importancia': 'mean'
    }).sort_values(by='Importancia', ascending=False).reset_index()

    # Guardar la importancia promedio en un archivo CSV
    importancia_promedio.to_csv(f'Variables/{Nombre}_{ClaseDelModelo}_BRTSImportanciaPromedio.csv', index=False)

    return {
        'BRTSerror': BRTSerror,
        #'BRTS': BRTS,
        'BRTpredic': BRTpredic,
        'BRTSvariables': BRTSvariables,
        'importancia_promedio': importancia_promedio,
    }