from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

def logistic_regression(X, y):
    lrmodel = LogisticRegression(max_iter=1000)  # Ensure convergence
    lrmodel.fit(X, y)
    return lrmodel

def decision_tree(X, y):
    decision_model = DecisionTreeClassifier()
    decision_model.fit(X, y)
    return decision_model

def random_forest(X, y):
    rfmodel = RandomForestClassifier()
    rfmodel.fit(X, y)
    return rfmodel

def run_cross_validation(config, X, y):
    """
    Perform 5x2 cross-validation and compute mean and standard deviation for each model.
    Parameters:
    config (dict): Configuration dictionary specifying the model and parameters.
    X (np.ndarray or pd.DataFrame): Feature matrix.
    y (np.ndarray or pd.Series): Target array.

    Returns:
    dict: A dictionary with models as keys and their mean/std accuracy as values.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['random_state'])
    
    # Define models
    model_functions = {
        'logistic_regression': logistic_regression,
        'decision_tree': decision_tree,
        'random_forest': random_forest,
    }
    
    results = {}
    if config['model'] == 'compare':  # Evaluate all models
        for model_name, model_function in model_functions.items():
            mean, std = cross_validate(model_function, X, y, skf)
            results[model_name] = {'mean': mean, 'std': std}
    else:  # Evaluate a single model
        model_function = model_functions[config['model']]
        mean, std = cross_validate(model_function, X, y, skf)
        results[config['model']] = {'mean': mean, 'std': std}
    
    return results

def cross_validate(model_function, X, y, skf):
    """
    Performs 5x2 cross-validation.
    Parameters:
    model_function (function): Function to train and return the model.
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target array.
    skf (StratifiedKFold): StratifiedKFold splitter instance.

    Returns:
    float, float: Mean and standard deviation of accuracy scores.
    """
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        # Use numpy indexing for train-test split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # First fold
        model_1 = model_function(X_train, y_train)
        scores.append(model_1.score(X_test, y_test))

        # Second fold (reverse roles)
        model_2 = model_function(X_test, y_test)
        scores.append(model_2.score(X_train, y_train))
    
    return np.mean(scores), np.std(scores)