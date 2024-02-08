import numpy as np
from sklearn import metrics


def single_timeseries_evaluation_metrics_func(y_true, y_pred):
    """
    Calculate evaluation metrics for a single time series.

    Args:
        y_true (array-like): True values of the time series.
        y_pred (array-like): Predicted values of the time series.

    Returns:
        None
    """

    def mean_absolute_percentage_error(y_true, y_pred): 
        """Calculate the mean absolute percentage error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate evaluation metrics
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    nmae = NMAE_error(y_true, y_pred)

    dec = 3  # Decimal places for rounding

    # Print evaluation metric results
    print('Evaluation metric results:-')
    print(f'MAE is : {round(mae, dec)}')
    print(f'RMSE is : {round(rmse, dec)}')
    print(f'MAPE is : {round(mape, dec)}')
    print(f'NMAE is : {round(nmae, dec)}')

def NMAE_error(y_true, y_pred):
    """
    Calculate the normalized mean absolute error.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Normalized mean absolute error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return metrics.mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))

def timeseries_evaluation_metrics_func(y_true, y_pred):
    """
    Calculate evaluation metrics for multiple time series.

    Args:
        y_true (array-like): True values of the time series.
        y_pred (array-like): Predicted values of the time series.

    Returns:
        None
    """

    def mean_absolute_percentage_error(y_true, y_pred):
        """
        Calculate the mean absolute percentage error for each time series column.
        """
        mask = y_true != 0  # Create a mask to avoid division by zero
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100

    def NMAE_error(y_true, y_pred):
        """Calculate the normalized mean absolute error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return metrics.mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))
   
    mae_values = []
    rmse_values = []
    mape_values = []
    nmae_values = []
    dec = 3  # Decimal places for rounding
    
    # Calculate evaluation metrics for each time series column
    for col_idx in range(y_true.shape[1]):
        col_true = y_true[:, col_idx]
        col_pred = y_pred[:, col_idx]
        
        mae_values.append(metrics.mean_absolute_error(col_true, col_pred))
        rmse_values.append(np.sqrt(metrics.mean_squared_error(col_true, col_pred)))
        mape_values.append(mean_absolute_percentage_error(col_true, col_pred))
        nmae_values.append(NMAE_error(col_true, col_pred))

    # Print evaluation metric results
    print('Evaluation metric results:')
    print(f'MAE is : {round(np.mean(mae_values), dec)}')
    print(f'RMSE is : {round(np.mean(rmse_values), dec)}')
    print(f'MAPE is : {round(np.mean(mape_values), dec)}')
    print(f'NMAE is : {round(np.mean(nmae_values), dec)}')