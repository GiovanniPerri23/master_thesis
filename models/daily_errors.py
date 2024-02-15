import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

def calculate_mae_errors(y_test_inv, transforecast):
    errors = []
    for row_idx in range(y_test_inv.shape[0]):
        y_true_row = y_test_inv[row_idx, :]
        y_pred_row = transforecast[row_idx, :]
        error_row = metrics.mean_absolute_error(y_true_row, y_pred_row)
        errors.append(error_row)
    return errors

def calculate_mape_errors(y_test_inv, transforecast):
    errors = []
    for row_idx in range(y_test_inv.shape[0]):
        y_true_row = y_test_inv[row_idx, :]
        y_pred_row = transforecast[row_idx, :]
        error_row = metrics.mean_absolute_percentage_error(y_true_row, y_pred_row)
        errors.append(error_row)
    return errors

def calculate_rmse_errors(y_test_inv, transforecast):
    errors = []
    for row_idx in range(y_test_inv.shape[0]):
        y_true_row = y_test_inv[row_idx, :]
        y_pred_row = transforecast[row_idx, :]
        error_row = np.sqrt(metrics.mean_squared_error(y_true_row, y_pred_row))
        errors.append(error_row)
    return errors

def visualize_errors_with_timestamps(errors, timestamps, mean_midnight_error):
    """
    Visualizza gli errori calcolati per ogni riga con i relativi timestamp utilizzando un grafico a barre.

    Args:
        errors (array-like): Array degli errori calcolati per ogni riga.
        timestamps (array-like): Array dei timestamp corrispondenti.
        mean_midnight_error (float): Valore medio degli errori calcolati.

    Returns:
        None
    """
    # Plot degli errori con i timestamp utilizzando un grafico a barre
    plt.figure(figsize=(12, 6))
    plt.bar(timestamps, errors, width=0.8, color='skyblue', label='Errors')
    plt.axhline(y=mean_midnight_error, color='red', linestyle='--', label='Mean Error')
    plt.xlabel('Timestamp')
    plt.ylabel('Error per day [€]')
    plt.title('Errors Over Time')
    plt.xticks(rotation=45)  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Linee orizzontali tratteggiate
    plt.legend()
    plt.tight_layout()
    plt.show()

def filter_errors_at_midnight(errors, timestamps):
    '''Function used to filter only forecasts starting at 00.00'''
    midnight_errors = []
    midnight_timestamps = []
    count = 0
    for error, timestamp in zip(errors, timestamps):
        hour = timestamp.hour

        if hour == 0:
            midnight_errors.append(error)
            midnight_timestamps.append(timestamp)
            count += 1
    print(f'Errore relativo a {count} giorni, a partire da mezzanotte')        
    return midnight_errors, midnight_timestamps

def errors_on_rows(y_test_inv, forecast, dataset_input):
    errors = calculate_rmse_errors(y_test_inv, forecast)
    # Ottieni il timestamp per le righe di test
    test_timestamps = dataset_input.iloc[-y_test_inv.shape[0]:].index

    # Converti l'indice in oggetti datetime
    test_timestamps = pd.to_datetime(test_timestamps)
    midnight_errors, midnight_timestamps = filter_errors_at_midnight(errors, test_timestamps)

    mean_midnight_error = np.mean(midnight_errors)

    visualize_errors_with_timestamps(midnight_errors, midnight_timestamps, mean_midnight_error)


    print("Media degli errori relativi alla mezzanotte:", mean_midnight_error)