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