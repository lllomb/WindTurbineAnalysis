#####################################################################################
# Utilities for the Eolic Forecasting Project
#####################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import seaborn as sns
import math
from types import SimpleNamespace
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def supervised_dataset(df, window, forecast_horizon, offset, features, target):
    """
    Prepara un conjunto de datos supervisados para un problema de series temporales o forecasting,
    dividiendo el DataFrame en características (X) y una variable objetivo (y).
    
    Parámetros:
    - df: DataFrame de pandas que contiene los datos históricos.
    - window: Número de periodos anteriores que se utilizarán como entrada para predecir el siguiente periodo.
    - forecast_horizon: Número de periodos hacia adelante que se desea predecir.
    - offset: Parámetro opcional que determina si solo se debe tomar el último valor del horizonte de pronóstico (offset == 1) o todos los valores dentro de ese horizonte.
    - features: Lista de nombres de columnas a utilizar como características.
    - target: Nombre de la columna a utilizar como variable objetivo.
    
    Retorna:
    - X: Secuencias de entrada como array numpy.
    - y: Secuencias de salida como array numpy.
    """
    X_sequences = []
    y_sequences = []

    for i in range(len(df) - window - forecast_horizon + 1):
        # Selección de características para la secuencia de entrada:
        X_seq = df.iloc[i:i+window][features].values
        X_sequences.append(X_seq)
        
        # Selección de la variable objetivo para la secuencia de salida:
        y_seq = df.iloc[i+window:i+window+forecast_horizon][target].values
        if offset == 1:
            y_sequences.append(y_seq[-1])
        else:
            y_sequences.append(y_seq)

    # Convert lists to arrays
    X = np.array(X_sequences)
    y = np.array(y_sequences)

    return X, y


def train_val_test_split(X, y, first_split, second_split):
    """
    Función para dividir los datos en conjuntos de entrenamiento, validación y prueba basándose en los porcentajes proporcionados.
    
    Parámetros:
    - X: Conjunto de datos de características.
    - y: Conjunto de datos objetivo.
    - first_split: Porcentaje inicial para dividir entre entrenamiento y validación.
    - second_split: Porcentaje adicional para dividir entre validación y prueba.
    
    Retorna:
    - X_train, y_train: Datos de entrenamiento.
    - X_val, y_val: Datos de validación.
    - X_test, y_test: Datos de prueba.
    """
    n = X.shape[0]
    
    # Calculando los índices de división
    split_index_first = int(n * first_split)
    split_index_second = int(n * second_split)
    
    # Dividiendo los datos según los porcentajes proporcionados
    X_train = X[:split_index_first]
    y_train = y[:split_index_first]
    
    X_val = X[split_index_first:split_index_second]
    y_val = y[split_index_first:split_index_second]
    
    X_test = X[split_index_second:]
    y_test = y[split_index_second:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

    
def plot_forecast(y_test=None, pred_raw=None, index=None, history=None, aero=None, parque_eolico=None):
    
    """
    This function generates two plots:
    1. Loss curve over epochs for both training and validation sets.
    2. Comparison between forecasted values and actual values.

    Parameters:
    - y_test (numpy.ndarray): The actual values for the test set.
    - pred_raw (numpy.ndarray): The raw predictions made by the model - no scaled.
    - index (list or numpy.ndarray): The indices corresponding to the x-axis of the plots.
    - history (keras.callbacks.History object): The training history object returned by Keras during model fitting.

    Returns:
    None. The function displays the plots using matplotlib.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    
    # Plotting loss and validation loss over epochs
    (pd.Series(history.history['loss']).plot(color="black", title='Loss by Epoch', ax=ax[0], label='loss'))
    (pd.Series(history.history['val_loss']).plot(color=(30/255, 165/255, 221/255), ax=ax[0], label='val_loss'))
    ax[0].legend()
    
    # Plotting forecasted vs actual values
    pd.Series(y_test.reshape(-1), index=index).plot(style='k--', ax=ax[1], title=f'Prediction vs Actual Potencia Activa - {aero}', label='Actual')
    pd.Series(pred_raw.reshape(-1), index=index).plot(color=(30/255, 165/255, 221/255), linestyle='--', label='Prediction', ax=ax[1])
    
    fig.tight_layout()
    ax[1].legend()
    plt.show()


def test_val_metrics(y_test, y_pred, y_val, y_val_pred):
    """
    Calcula las métricas MSE, RMSE, MAE y R^2 para los conjuntos de prueba y validación,
    y luego imprime los resultados.

    Parameters:
    - y_test: Array de verdad para el conjunto de prueba.
    - y_pred: Predicciones para el conjunto de prueba.
    - y_val: Array de verdad para el conjunto de validación.
    - y_val_pred: Predicciones para el conjunto de validación.
    
    """
    # Aplana los datos si son multidimensionales
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    y_val_flat = y_val.flatten()
    y_val_pred_flat = y_val_pred.flatten()

    mse_test = mean_squared_error(y_test_flat, y_pred_flat)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test_flat, y_pred_flat)
    r2_test = r2_score(y_test_flat, y_pred_flat)
    
    mse_val = mean_squared_error(y_val_flat, y_val_pred_flat)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_val_flat, y_val_pred_flat)
    r2_val = r2_score(y_val_flat, y_val_pred_flat)
    
    metrics_results = {
        "Test": {"MSE": mse_test, "RMSE": rmse_test, "MAE": mae_test, "R^2": r2_test},
        "Validation": {"MSE": mse_val, "RMSE": rmse_val, "MAE": mae_val, "R^2": r2_val}
    }
    
    # Imprime los resultados
    for key, value in metrics_results.items():
        print(f"{key}:")
        for metric_name, metric_value in value.items():
            print(f"  {metric_name}: {metric_value:.2f}")
