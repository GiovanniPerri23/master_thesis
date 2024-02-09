import matplotlib.pyplot as plt

def plot_model_loss(history):
    """
    Plot training and validation loss from a Keras model's history.

    Args:
        history (keras.callbacks.History): The history object from training a Keras model.

    Returns:
        None

    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(16, 8))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')
    
    plt.show()

def plot_error_metrics_barchart(error_table):
    """
    Plots a bar chart to visualize error metrics for each column.

    Parameters:
    error_table (DataFrame): A DataFrame containing error metrics for each column.

    Returns:
    None
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    # Grafico a barre per NMAE
    ax1.bar(error_table['Column'], error_table['NMAE'], color='blue', alpha=0.7)
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('NMAE')
    ax1.set_title('NMAE for Each Column')

    # Grafico a barre per MAE, RMSE, MAPE
    # ax2.bar(error_table['Column'] - 0.2, error_table['MAE'], width=0.2, color='green', label='MAE', alpha=0.7)
    ax2.bar(error_table['Column'], error_table['RMSE'], width=0.2, color='orange', label='RMSE', alpha=0.7)
    ax2.bar(error_table['Column'] + 0.2, error_table['MAPE'], width=0.2, color='red', label='MAPE', alpha=0.7)
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Errors')
    ax2.set_title('RMSE and MAPE for Each Column')
    ax2.legend()

    # Trova il minimo di NMAE, RMSE e MAPE
    min_nmae_row = error_table.loc[error_table['NMAE'].idxmin()]
    min_rmse_row = error_table.loc[error_table['RMSE'].idxmin()]
    min_mape_row = error_table.loc[error_table['MAPE'].idxmin()]

    # Stampa i minimi e le relative colonne
    print("Min NMAE:", min_nmae_row['NMAE'], "for Column:", min_nmae_row['Column'])
    print("Min RMSE:", min_rmse_row['RMSE'], "for Column:", min_rmse_row['Column'])
    print("Min MAPE:", min_mape_row['MAPE'], "for Column:", min_mape_row['Column'])
    print((min_nmae_row['NMAE'], min_nmae_row['RMSE'], min_nmae_row['MAPE']))
    plt.tight_layout()
    plt.show()
    
def plot_model_rmse_and_loss(history):
    
    # Evaluate train and validation accuracies and losses
    
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Visualize epochs vs. train and validation accuracies and losses
    
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.legend()
    plt.title('Epochs vs. Training and Validation RMSE')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')
    
    plt.show()    

def plot_error_metrics(error_table):
    # Creazione dei subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    # Plot per NMAE
    ax1.scatter(error_table['Column'], error_table['NMAE'], label='NMAE', alpha=0.7, marker='o')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('NMAE')
    ax1.set_title('NMAE for Each Column')
    ax1.legend()

    # Plot per MAE, RMSE, MAPE
    ax2.scatter(error_table['Column'], error_table['RMSE'], label='RMSE', alpha=0.7, marker='o')
    ax2.scatter(error_table['Column'], error_table['MAPE'], label='MAPE', alpha=0.7, marker='o')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Errors')
    ax2.set_title('RMSE and MAPE for Each Column')
    ax2.legend()

    # Trova il minimo di NMAE, RMSE e MAPE
    min_nmae_row = error_table.loc[error_table['NMAE'].idxmin()]
    min_rmse_row = error_table.loc[error_table['RMSE'].idxmin()]
    min_mape_row = error_table.loc[error_table['MAPE'].idxmin()]

    # Stampa i minimi e le relative colonne
    print("Min NMAE:", min_nmae_row['NMAE'], "for Column:", min_nmae_row['Column'])
    print("Min RMSE:", min_rmse_row['RMSE'], "for Column:", min_rmse_row['Column'])
    print("Min MAPE:", min_mape_row['MAPE'], "for Column:", min_mape_row['Column'])


    plt.tight_layout()
    plt.show()
