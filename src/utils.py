import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X, y, sens=None, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets. Optionally splits sensitive attribute also.

    Parameters:
    X (array-like): Features
    y (array-like): Labels
    sens (array-like, optional): Sensitive attribute
    test_size (float): Proportion of test set
    random_state (int): Random state for reproducibility

    Returns:
    dict: Dictionary containing train/test splits (with 'sens_train' and 'sens_test' if sens provided)
    """
    if sens is not None:
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sens, test_size=test_size, random_state=random_state, stratify=y
        )
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "sens_train": sens_train, "sens_test": sens_test,
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }

def extract_sensitive_attribute(data, config):
    """
    Extract sensitive attribute based on dataset configuration.
    Returns np.array: 1 for privileged group (Male), 0 for unprivileged (Female).
    """
    if config['dataname'] == 'brazil':
        if 'Gender' in data.columns:
            # values are 0 or 1: let's check which means what
            val_counts = data['Gender'].value_counts()
            # If '0' is male and '1' is female as is conventional
            # Make privileged = 0 (male) --> 1, unpriv = 1 (female) --> 0
            # But if you prefer the opposite, reverse these.
            return (data['Gender'] == 0).astype(int).values
        else:
            return np.random.randint(0, 2, size=len(data))
    elif config['dataname'] == 'africa':
        if 'gender' in data.columns:
            # Already "Male"/"Female", so 'Male' = 1
            return (data['gender'].str.upper() == 'MALE').astype(int).values
        else:
            return np.random.randint(0, 2, size=len(data))
    elif config['dataname'] == 'india':
        if 'STUDENTGENDER' in data.columns:
            # 'M' for male, 'F' for female
            return (data['STUDENTGENDER'].str.upper().isin(['M', 'MALE'])).astype(int).values
        else:
            return np.random.randint(0, 2, size=len(data))
    else:
        return np.random.randint(0, 2, size=len(data))

def test_error(y_pred, y_true):
    return np.mean(np.array(y_pred) != np.array(y_true))

def demographic_parity(y_pred, mask):
    y_pred, mask = np.array(y_pred), np.array(mask)
    return np.mean(y_pred[mask]) if np.sum(mask) else 0.0

def equal_opportunity(y_pred, y_true, mask):
    y_pred, y_true, mask = map(np.array, (y_pred, y_true, mask))
    if not np.sum(mask) or not np.sum(y_true[mask]): return 0.0
    return np.sum((y_pred[mask] == 1) & (y_true[mask] == 1)) / np.sum(y_true[mask] == 1)

def false_positive_rate(y_pred, y_true, mask):
    y_pred, y_true, mask = map(np.array, (y_pred, y_true, mask))
    y_pred_group = y_pred[mask]
    y_true_group = y_true[mask]
    if len(y_pred_group) == 0: return 0.0
    fp = np.sum((y_pred_group == 1) & (y_true_group == 0))
    tn = np.sum((y_pred_group == 0) & (y_true_group == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def accuracy_rate(y_pred, y_true, mask):
    y_pred, y_true, mask = map(np.array, (y_pred, y_true, mask))
    if not np.sum(mask): return 0.0
    return np.mean(y_pred[mask] == y_true[mask])

def recall_rate(y_pred, y_true, mask):
    y_pred, y_true, mask = map(np.array, (y_pred, y_true, mask))
    if not np.sum(mask): return 0.0
    y_pred_group = y_pred[mask]
    y_true_group = y_true[mask]
    pos = np.sum(y_true_group == 1)
    if pos == 0: return 0.0
    return np.sum((y_pred_group == 1) & (y_true_group == 1)) / pos

def fairness_summary(y_pred, y_true, sensitive_attribute, model_name="Model"):
    """
    Generate a comprehensive fairness summary for reporting/comparison.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    sensitive_attribute = np.array(sensitive_attribute)
    privileged_mask = sensitive_attribute == 1
    unprivileged_mask = sensitive_attribute == 0

    metrics = {
        'model_name': model_name,
        'overall_accuracy': np.mean(y_pred == y_true),
        'privileged_accuracy': accuracy_rate(y_pred, y_true, privileged_mask),
        'unprivileged_accuracy': accuracy_rate(y_pred, y_true, unprivileged_mask),
        'privileged_tpr':    equal_opportunity(y_pred, y_true, privileged_mask),
        'unprivileged_tpr':  equal_opportunity(y_pred, y_true, unprivileged_mask),
        'privileged_fpr':    false_positive_rate(y_pred, y_true, privileged_mask),
        'unprivileged_fpr':  false_positive_rate(y_pred, y_true, unprivileged_mask),
        'privileged_selection_rate':    demographic_parity(y_pred, privileged_mask),
        'unprivileged_selection_rate':  demographic_parity(y_pred, unprivileged_mask),
    }
    metrics['accuracy_difference'] = metrics['privileged_accuracy'] - metrics['unprivileged_accuracy']
    metrics['tpr_difference'] = metrics['privileged_tpr'] - metrics['unprivileged_tpr']
    metrics['fpr_difference'] = metrics['privileged_fpr'] - metrics['unprivileged_fpr']
    metrics['demographic_parity_difference'] = metrics['privileged_selection_rate'] - metrics['unprivileged_selection_rate']
    return metrics

def print_fairness_report(metrics):
    """
    Print a formatted fairness report for the model.
    """
    print(f"\n=== Fairness Report for {metrics['model_name']} ===")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"\nGroup-wise Performance:")
    print(f"  Privileged Group Accuracy:   {metrics['privileged_accuracy']:.4f}")
    print(f"  Unprivileged Group Accuracy: {metrics['unprivileged_accuracy']:.4f}")
    print(f"  Accuracy Difference:         {metrics['accuracy_difference']:.4f}")
    print("\nTrue Positive Rate (Recall):")
    print(f"  Privileged Group TPR:        {metrics['privileged_tpr']:.4f}")
    print(f"  Unprivileged Group TPR:      {metrics['unprivileged_tpr']:.4f}")
    print(f"  TPR Difference:              {metrics['tpr_difference']:.4f}")
    print("\nFalse Positive Rate:")
    print(f"  Privileged Group FPR:        {metrics['privileged_fpr']:.4f}")
    print(f"  Unprivileged Group FPR:      {metrics['unprivileged_fpr']:.4f}")
    print(f"  FPR Difference:              {metrics['fpr_difference']:.4f}")
    print("\nSelection Rate (Demographic Parity):")
    print(f"  Privileged Group Selection Rate:   {metrics['privileged_selection_rate']:.4f}")
    print(f"  Unprivileged Group Selection Rate: {metrics['unprivileged_selection_rate']:.4f}")
    print(f"  Demographic Parity Difference:     {metrics['demographic_parity_difference']:.4f}")
    print("\nFairness Assessment:")
    if abs(metrics['demographic_parity_difference']) < 0.1:
        print("  ✓ Demographic Parity: FAIR (difference < 0.1)")
    else:
        print("  ✗ Demographic Parity: Potential Bias (difference >= 0.1)")