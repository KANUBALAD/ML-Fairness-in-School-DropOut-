import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc


# Calculate test error
def test_error(y_pred, y_true):
    """
    Calculates the test error rate.

    Parameters:
    y_pred (numpy array or torch.Tensor): Predicted labels
    y_true (numpy array or torch.Tensor): True labels

    Returns:
    float: Test error rate
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return np.mean(y_pred != y_true)

# Calculate demographic parity
def demographic_parity(y_pred, mask):
    """
    Calculates the demographic parity.

    Parameters:
    y_pred (numpy array): Predicted labels
    mask (numpy array): Mask indicating the protected group

    Returns:
    float: Demographic parity
    """
    y_cond = y_pred * mask.astype(np.float32)
    return y_cond.sum() / mask.sum()

# Calculate equality of opportunity
def equal_opportunity(y_pred, y_true, mask):
    """
    Calculates the equality of opportunity.

    Parameters:
    y_pred (numpy array): Predicted labels
    y_true (numpy array): True labels
    mask (numpy array): Mask indicating the protected group

    Returns:
    float: Equality of opportunity
    """
    y_cond = y_true * y_pred * mask.astype(np.float32)
    return y_cond.sum() / (y_true * mask.astype(np.float32)).sum()

# Calculate false positive rate
def false_positive_rate(y_pred, y_true, mask):
    """
    Calculates the false positive rate.

    Parameters:
    y_pred (numpy array): Predicted labels
    y_true (numpy array): True labels
    mask (numpy array): Mask indicating the protected group

    Returns:
    float: False positive rate
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = np.array(mask)
    mask_float = mask.astype(np.float32)
    false_positives = ((y_pred == 1) & (y_true == 0)).astype(np.float32) * mask_float
    true_negatives = ((y_pred == 0) & (y_true == 0)).astype(np.float32) * mask_float
    
    FP = false_positives.sum()
    TN = true_negatives.sum()
    
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    return fpr

# Calculate accuracy rate
def accuracy_rate(y_pred, y_true, mask):
    """
    Calculates the accuracy rate.

    Parameters:
    y_pred (numpy array): Predicted labels
    y_true (numpy array): True labels
    mask (numpy array): Mask indicating the protected group

    Returns:
    float: Accuracy rate
    """
    correct_predictions = (y_pred == y_true) * mask.astype(np.float32)
    accuracy = correct_predictions.sum() / mask.sum() if mask.sum() > 0 else 0
    return accuracy

# Calculate recall rate
def recall_rate(y_pred, y_true, mask):
    """
    Calculates the recall rate.

    Parameters:
    y_pred (numpy array or pandas Series): Predicted labels
    y_true (numpy array or pandas Series): True labels
    mask (numpy array or pandas Series): Mask indicating the protected group

    Returns:
    float: Recall rate
    """
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(mask, pd.Series):
        mask = mask.values
    mask_float = mask.astype(np.float32)
    true_positives = ((y_pred == 1) & (y_true == 1)).astype(float) * mask_float
    actual_positives = (y_true == 1).astype(float) * mask_float
    TP_sum = np.sum(true_positives)
    AP_sum = np.sum(actual_positives)
    recall = TP_sum / AP_sum if AP_sum > 0 else 0
    
    return recall

def model_metrics(model_name, Xtest_data, ytest_data, mask=None, fair_metrics=True):
    """
    Calculates various evaluation metrics for a given model.

    Parameters:
    model_name: The trained model
    Xtest_data: Test data features
    ytest_data: Test data labels
    mask: Mask indicating the protected group (default: None)
    fair_metrics: Flag to calculate fairness metrics (default: True)

    Returns:
    tuple: Tuple containing demographic parity, equality of opportunity, false positive rate, accuracy rate, and recall rate
    """
    y_preds = model_name.predict(Xtest_data)
    
    if fair_metrics:
        mydemographic_parity = demographic_parity(y_preds, mask)
        equality_opportunity = equal_opportunity(y_preds, ytest_data, mask)
        fpr = false_positive_rate(y_preds, ytest_data, mask)
        myaccuracy_rate = accuracy_rate(y_preds, ytest_data, mask)
        myrecall_rate = recall_rate(y_preds, ytest_data, mask)
        return mydemographic_parity, equality_opportunity, fpr, myaccuracy_rate, myrecall_rate, y_preds

        
    else:
        general_accuracy = accuracy_score(ytest_data, y_preds)
        general_f1_score = f1_score(ytest_data, y_preds, average='weighted')
        recall_scores= recall_score(ytest_data, y_preds, average='weighted')
        precision_scores= precision_score(ytest_data, y_preds, average='weighted')
        return general_accuracy, general_f1_score, recall_scores, precision_scores
    
    
    
    
def plot_roc_curve(ytest_data, y_pred_proba, model_name='Model'):
    """
    Plots the ROC curve for the given model's predictions.

    Parameters:
    ytest_data: True labels
    y_pred_proba: Predicted probabilities for the positive class
    model_name: Name of the model (default: 'Model')
    """
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(ytest_data, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

        
    
        
