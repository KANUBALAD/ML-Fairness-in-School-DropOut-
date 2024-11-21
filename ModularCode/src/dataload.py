import pandas as pd
import numpy as np
from .preprocess import preprocess_data
from sklearn.model_selection import train_test_split

def load_data(config):
    """
    Loads and preprocesses the dataset based on the configuration provided.

    Parameters:
    config (dict): A dictionary containing configuration parameters for loading and preprocessing the data.
        - 'dataname': The name of the dataset to load (e.g., 'brazil', 'africa').
        - 'datapath': The path to the dataset.
        - 'unawareness': Boolean indicating whether to exclude certain features.

    Returns:
    tuple: A tuple containing the features (X) and target (y).
    """
    # Load the dataset
    path = config['datapath']
    data = pd.read_csv(path)

    # Determine the target column and binary conversion based on dataset
    if config['dataname'] == 'brazil':
        target_column = 'Target'
        data[target_column] = np.where(data[target_column] == 'Dropout', 'YES', 'NO')
        data[target_column] = data[target_column].apply(lambda x: 1 if x == 'YES' else 0)
    elif config['dataname'] == 'africa':
        target_column = 'dropout'
        data[target_column] = data[target_column].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        raise ValueError(f"Unsupported dataset name: {config['dataname']}")

    # Separate the target variable and the features
    y = data[target_column]

    # Exclude specific columns if 'unawareness' is enabled
    if config.get('unawareness', False):
        if config['dataname'] == 'brazil':
            X = data.drop(columns=[target_column, 'Gender'])  # Excludes gender
        elif config['dataname'] == 'africa':
            X = data.drop(columns=[target_column, 'gender'])  # Excludes gender
    else:
        X = data.drop(columns=[target_column])  # Keeps all other features

    # Preprocess the features
    X_transformed = preprocess_data(X)

    return X_transformed, y