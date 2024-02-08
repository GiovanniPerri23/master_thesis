import numpy as np

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    """
    Create a multivariate time series dataset from given dataset and target.

    Args:
        dataset (numpy.ndarray): The input dataset.
        target (numpy.ndarray): The target dataset.
        start_index (int): The starting index of the data to be used.
        end_index (int): The ending index of the data to be used.
        history_size (int): The number of time steps to consider as history.
        target_size (int): The number of time steps into the future to predict.
        step (int): The sampling step.
        single_step (bool, optional): If True, predicts only one step into the future. 
                                      If False, predicts multiple steps.

    Returns:
        numpy.ndarray: The processed data.
        numpy.ndarray: The corresponding labels.

    """
    data = []
    labels = []

    if start_index is None:
        start_index = 0

    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index, step):
        indices = range(i, i + history_size)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + history_size + target_size])
        else:
            labels.append(target[i + history_size: i + history_size + target_size])

        if i + history_size > len(dataset) - target_size - history_size:
            break
    
    return np.array(data), np.array(labels)

def prepare_data(dataset_input, univariate=False):
    """
    Prepare the dataset for training.

    Args:
        dataset_input (DataFrame): The input dataset.
        univariate (bool, optional): Indicates whether the data should be prepared as univariate or multivariate.
                                     Defaults to False.

    Returns:
        tuple: A tuple containing the input features X and the target variable y.
    """
    # Extract the target variable 'PUN'
    if univariate:
        # Prepare data as univariate
        X = dataset_input['PUN'].values.reshape(-1, 1)
    else:
        # Prepare data as multivariate
        X = dataset_input.values

    # Target variable y is the same as 'PUN'
    y = dataset_input['PUN'].values.reshape(-1, 1)

    return X, y
