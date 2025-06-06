import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor, DMatrix, train as xgb_train

def boosted_regression_trees(data_set, response, train_indices, ntree=500, eta=0.05, max_depth=5):
    # Identificar variables categóricas y convertirlas en factores (si es necesario)
    cat_vars = data_set.select_dtypes(include=['object']).columns
    if len(cat_vars) > 0:
        data_set[cat_vars] = data_set[cat_vars].apply(lambda x: pd.factorize(x)[0])

    # Verificar si la variable de respuesta es categórica
    if data_set[response].dtype == 'object' or len(data_set[response].unique()) > 2:
        data_set[response] = pd.factorize(data_set[response])[0]
        objective = 'multi:softmax'
        num_class = len(np.unique(data_set[response]))
    else:
        objective = 'reg:squarederror'
        num_class = None

    # Crear la matriz de características
    X = pd.get_dummies(data_set.drop(columns=[response]), drop_first=True)
    y = data_set[response].values

    # Crear la máscara booleana que identifica los índices de prueba
    mask = np.isin(range(len(y)), train_indices, invert=True)

    # Crear la matriz de entrenamiento y prueba
    dtrain = DMatrix(data=X.iloc[train_indices, :], label=y[train_indices])
    dtest = DMatrix(data=X.iloc[mask, :], label=y[mask])

    params = {
        'objective': objective,
        'eta': eta,
        'max_depth': max_depth,
        'eval_metric': 'mlogloss' if objective == 'multi:softmax' else 'rmse'
    }

    if num_class:
        params['num_class'] = num_class

    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    bst_model = xgb_train(params, dtrain, num_boost_round=ntree, evals=watchlist, verbose_eval=False)

    # Generar predicciones
    BRT_pred = bst_model.predict(dtest)
    actual = y[mask]

    if objective == 'multi:softmax':
        accuracy = accuracy_score(actual, BRT_pred)
        conf_matrix = confusion_matrix(actual, BRT_pred)
        return {
            'BRT': bst_model,
            'BRT_pred': BRT_pred,
            'BRT_real': actual,
            'Accuracy': accuracy,
            'ConfusionMatrix': conf_matrix
        }
    else:
        BRT_MAE = mean_absolute_error(actual, BRT_pred)
        BRT_RMSE = np.sqrt(mean_squared_error(actual, BRT_pred))
        return {
            'BRT': bst_model,
            'BRT_pred': BRT_pred,
            'BRT_real': actual,
            'BRT_MAE': BRT_MAE,
            'BRT_RMSE': BRT_RMSE
        }
def ExperimentacionBoostedTrees(DataSet, Respuesta, listaIndices, pe, Direccion, Replicas):
    print("Inicia Experimentación de Boosted Trees")

    BRTSerror = pd.DataFrame(columns=['Accuracy', 'MAE', 'RMSE', 'NUMERO', 'PercentTest'])
    BRTpredic = pd.DataFrame(columns=['Prediccion', 'Real', 'NUMERO', 'PercentTest'])
    BRTS = listaIndices

    for i in range(Replicas):
        print(f"Inicia Experimentación Réplica: {i+1}")
        for j in pe:
            train_indices = listaIndices[f"C{i+1}S{j}"]
            resultados = boosted_regression_trees(DataSet, Respuesta, train_indices, ntree=500, eta=0.05, max_depth=5)
            print(f"Termina Experimentación Réplica: {i+1} Porcentaje de Experimentación: {j}")

            BRTS[f"C{i+1}S{j}"] = resultados['BRT']

            if 'Accuracy' in resultados:
                error = pd.DataFrame({'Accuracy': [resultados['Accuracy']], 'MAE': [None], 'RMSE': [None], 'NUMERO': [i+1], 'PercentTest': [j]})
            else:
                error = pd.DataFrame({'Accuracy': [None], 'MAE': [resultados['BRT_MAE']], 'RMSE': [resultados['BRT_RMSE']], 'NUMERO': [i+1], 'PercentTest': [j]})

            prediccion = pd.DataFrame({'Prediccion': resultados['BRT_pred'], 'Real': resultados['BRT_real'], 'NUMERO': i+1, 'PercentTest': j})
            BRTSerror = pd.concat([BRTSerror, error], ignore_index=True)
            BRTpredic = pd.concat([BRTpredic, prediccion], ignore_index=True)

    # Guardar resultados si es necesario
    #BRTSerror.to_csv(f"{Direccion}/Boosted_Regression_Trees_Error_Res.csv", index=False)

    return {
        'BRTSerror': BRTSerror,
        'BRTS': BRTS,
        'BRTpredic': BRTpredic
    }