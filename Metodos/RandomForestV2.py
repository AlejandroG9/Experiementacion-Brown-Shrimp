import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Función simplificada para aplicar Random Forest
def RandomForest(DataSet, mtry, ntree, Respuesta,  Semilla, Test,ClaseDelModelo):
    X = DataSet.drop(columns=[Respuesta])
    y = DataSet[Respuesta]
    #print("Variables 2 1")
    #print("Variable de Respuesta")

    # Convertir variables categóricas en numéricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Si la variable respuesta es categórica, convertirla
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test, random_state=Semilla)

    #print("X TRAIN")
    #print(X_train.head)
    #print("X TEST")
    #print(X_test.head)
    #print("y TRAIN")
    #print(y_train)
    #print("y TEST")
    #print(y_test)

    # Seleccionar el tipo de modelo (clasificación o regresión)
    # Convertir a pandas.Series
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)

    ModelClass = RandomForestClassifier if ClaseDelModelo=="Clasificacion" else RandomForestRegressor
    #print(ModelClass)
    model = ModelClass(max_features=mtry, n_estimators=ntree).fit(X_train, y_train)


    # Extraer la importancia de las variables
    importancias = model.feature_importances_
    variables = X_train.columns
    importancia_variables = pd.DataFrame({
        'Variable': variables,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    # Realizar predicciones y calcular métricas
    y_pred = model.predict(X_test)
    #print("Y PRED")
    #print(y_pred)
    if ClaseDelModelo=="Clasificacion":  # Clasificación
        return {
            'model': model,
            'predictions': y_pred,
            'real_values': y_test_series.values,
            'accuracy': accuracy_score(y_test, y_pred),
            'conf_matrix': confusion_matrix(y_test, y_pred),
            'importancia_variables': importancia_variables
        }
    else:  # Regresión
        return {
            'model': model,
            'predictions': y_pred,
            'real_values': y_test.values,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred),#, squared=False),
            'importancia_variables': importancia_variables
        }

# Función para la experimentación de Random Forest
def ExperimentacionRandomForest(DataSet, Respuesta, ntree, Semilla, pe, Replicas, ClaseDelModelo, Nombre):
    print("Iniciando experimentación de Random Forest")

    RFSerror, RFSpredic = pd.DataFrame(), pd.DataFrame()
    RFS = {}
    RFSvariables = pd.DataFrame()  # Para almacenar las importancias globales

    for i in range(1, Replicas + 1):
        for j in pe:

            resultados = RandomForest(DataSet=DataSet, mtry=4, ntree=ntree, Respuesta=Respuesta, Semilla=Semilla, Test = (1-j),ClaseDelModelo=ClaseDelModelo)

            #RFS[f"C{i}S{j}"] = resultados['model']
            importancias_replica = resultados['importancia_variables']
            importancias_replica['Réplica'] = i
            importancias_replica['PercentTest'] = j
            RFSvariables = pd.concat([RFSvariables, importancias_replica], ignore_index=True)

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
            RFSerror.to_csv(f'Errores/{Nombre}_{ClaseDelModelo}_RFSerror_log.csv', index=False)
            print(f'Resultados: Accuracy: {resultados.get('accuracy')}, MAE: {resultados.get('mae')}, RMSE: {resultados.get('rmse')}')

            RFSerror = pd.concat([RFSerror, pd.DataFrame([error_data])], ignore_index=True)
            RFSpredic = pd.concat([RFSpredic, pd.DataFrame(predic_data)], ignore_index=True)

    # Calcular la importancia promedio de las variables
    importancia_promedio = RFSvariables.groupby('Variable').agg({
        'Importancia': 'mean'
    }).sort_values(by='Importancia', ascending=False).reset_index()

    # Guardar la importancia promedio en un archivo CSV
    importancia_promedio.to_csv(f'Variables/{Nombre}_{ClaseDelModelo}_RFSTImportanciaPromedio.csv', index=False)

    # Devolvemos los resultados en lugar de escribir en archivos
    return {
        'RFSerror': RFSerror,
        #'RFS': RFS,
        'RFSpredic': RFSpredic,
        'RFSvariables': RFSvariables,
        'importancia_promedio': importancia_promedio
    }