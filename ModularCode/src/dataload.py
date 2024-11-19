import pandas as pd
import numpy as np
from .preprocess import preprocess_data
from sklearn.model_selection import train_test_split


def load_data(config):
    """
    Loads and preprocesses the dataset
    Parameters:
    config (.yaml): A yaml file containing configuration parameters for loading and preprocessing the data. 
        - 'data_name': The name of the dataset to load (e.g., 'brazil').
        - 'datapath': The path to the dataset.

    Returns:
    pd.DataFrame: A DataFrame containing:
        - covariates (X).
        - Target (y).
    Raises:
    ValueError: If the specified 'data_name' is not supported.
    """
    # Load the dataset
    if config['dataname'] == 'brazil':
        path = config['datapath']
        data, ytarget = brazil_data(config, path)
        Xtransformed  = preprocess_data(data)
    else:
        path = config['datapath']
        data = pd.read_csv(path)
        data, ytarget  = preprocess_data(data)
        Xtransformed  = preprocess_data(data)
    return Xtransformed, ytarget
    
    

def brazil_data(config, path):
    data_brazil = pd.read_csv(path)
    data_brazil['Target'].unique()
    data_brazil['Target'] = np.where(data_brazil['Target'] == 'Dropout', 'YES', 'NO')
    # Convert the target variable to binary (0 or 1)
    data_brazil['Target'] = data_brazil['Target'].apply(lambda x: 1 if x == 'YES' else 0)
    # Separate the target variable and the features
    y = data_brazil['Target']
    
    if config['unawareness']:
        X = data_brazil.drop(columns=['Target', 'Gender']) #excludes gender from X features
    else:
        X = data_brazil.drop(columns=['Target']) # includes gender in X
    return X, y