import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Función simplificada para aplicar Random Forest
def RandomForest(DataSet, mtry, ntree, Respuesta, train_indices):
    X = DataSet.drop(columns=[Respuesta])
    y = DataSet[Respuesta]

    # Convertir variables categóricas en numéricas
    for column in X.select_dtypes(include='object').columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    # Si la variable respuesta es categórica, convertirla
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Dividir los datos en entrenamiento y prueba usando los índices proporcionados
    #X_train, X_test = X.iloc[train_indices], X.drop(train_indices)
    #y_train, y_test = y[train_indices], y.drop(train_indices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("X TRAIN")
    print(X_train.head)
    print("X TEST")
    print(X_test.head)
    print("y TRAIN")
    print(y_train.head)
    print("y TEST")
    print(y_test.head)

    # Seleccionar el tipo de modelo (clasificación o regresión)
    ModelClass = RandomForestClassifier if y_train.nunique() > 1 else RandomForestRegressor
    model = ModelClass(max_features=mtry, n_estimators=ntree).fit(X_train, y_train)

    # Realizar predicciones y calcular métricas
    y_pred = model.predict(X_test)
    if y_train.nunique() > 1:  # Clasificación
        return {
            'model': model,
            'predictions': y_pred,
            'real_values': y_test.values,
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
def ExperimentacionRandomForest(DataSet, Respuesta, listaIndices, pe, Replicas):
    print("Iniciando experimentación de Random Forest")

    RFSerror, RFSpredic = pd.DataFrame(), pd.DataFrame()
    RFS = {}

    for i in range(1, Replicas + 1):
        for j in pe:
            train = listaIndices[f"C{i}S{j}"]
            resultados = RandomForest(DataSet=DataSet, mtry=4, ntree=100, Respuesta=Respuesta, train_indices=train)

            RFS[f"C{i}S{j}"] = resultados['model']

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

            RFSerror = pd.concat([RFSerror, pd.DataFrame([error_data])], ignore_index=True)
            RFSpredic = pd.concat([RFSpredic, pd.DataFrame(predic_data)], ignore_index=True)

    # Devolvemos los resultados en lugar de escribir en archivos
    return {
        'RFSerror': RFSerror,
        'RFS': RFS,
        'RFSpredic': RFSpredic
    }