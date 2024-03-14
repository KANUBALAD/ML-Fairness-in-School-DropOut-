import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocessing_split(path_to_csv, split_data=True, unawareness=True, 
                             test_size=None, random_state=None):
    """
    Load data from a CSV file, preprocess it, and split it into train and test sets.
    
    Args:
        path_to_csv (str): The path to the CSV file.
        split_data (bool): Whether to split the data into train and test sets. Default is True.
        awareness (bool): Whether to consider awareness attributes during preprocessing. Default is True.
        test_size (float or None): The proportion of the dataset to include in the test split. Default is None.
        random_state (int or None): Controls the shuffling applied to the data before splitting. Default is None.
    
    Returns:
        If split_data is True:
            X_train (DataFrame): The training features.
            X_test (DataFrame): The test features.
            y_train (Series): The training target variable.
            y_test (Series): The test target variable.
            sens_train (DataFrame): The sensitive attributes for the training set.
            sens_test (DataFrame): The sensitive attributes for the test set.
        If split_data is False:
            X (DataFrame): The features.
            y (Series): The target variable.
            sensitive_attribute (DataFrame): The sensitive attributes.
    """
    
    # Load the CSV file into a DataFrame
    df_school = pd.read_csv(path_to_csv)
    df_school['Target'].unique()
    df_school['Target'] = np.where(df_school['Target'] == 'Dropout', 'YES', 'NO')

    # Convert the target variable to binary (0 or 1)
    df_school['Target'] = df_school['Target'].apply(lambda x: 1 if x == 'YES' else 0)
    
    # Separate the target variable and the features
    y = df_school['Target']
    
    if unawareness:
        X = df_school.drop(columns=['Target', 'Gender']) #excludes gender from X features
    else:
        X = df_school.drop(columns=['Target']) # includes gender in X
    
    # Extract the sensitive attributes
    sensitive_attribute = df_school[['Marital status', 'Nacionality', 'Gender']]
    
    if split_data:
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(X, y, sensitive_attribute, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, sens_train, sens_test
    else:
        return df_school, X, y, sensitive_attribute