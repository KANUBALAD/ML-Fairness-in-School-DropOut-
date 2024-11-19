# import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return split_data

# def test_error(y_pred, y_true):
#     """
#     Calculates the test error rate.

#     Parameters:
#     y_pred (numpy array or torch.Tensor): Predicted labels
#     y_true (numpy array or torch.Tensor): True labels

#     Returns:
#     float: Test error rate
#     """
#     if isinstance(y_pred, torch.Tensor):
#         y_pred = y_pred.numpy()
#         y_true = y_true.numpy()
#     return np.mean(y_pred != y_true)

# # Calculate demographic parity
# def demographic_parity(y_pred, mask):
#     """
#     Calculates the demographic parity.

#     Parameters:
#     y_pred (numpy array): Predicted labels
#     mask (numpy array): Mask indicating the protected group

#     Returns:
#     float: Demographic parity
#     """
#     y_cond = y_pred * mask.astype(np.float32)
#     return y_cond.sum() / mask.sum()

# # Calculate equality of opportunity
# def equal_opportunity(y_pred, y_true, mask):
#     """
#     Calculates the equality of opportunity.

#     Parameters:
#     y_pred (numpy array): Predicted labels
#     y_true (numpy array): True labels
#     mask (numpy array): Mask indicating the protected group

#     Returns:
#     float: Equality of opportunity
#     """
#     y_cond = y_true * y_pred * mask.astype(np.float32)
#     return y_cond.sum() / (y_true * mask.astype(np.float32)).sum()

# # Calculate false positive rate
# def false_positive_rate(y_pred, y_true, mask):
#     """
#     Calculates the false positive rate.

#     Parameters:
#     y_pred (numpy array): Predicted labels
#     y_true (numpy array): True labels
#     mask (numpy array): Mask indicating the protected group

#     Returns:
#     float: False positive rate
#     """
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     mask = np.array(mask)
#     mask_float = mask.astype(np.float32)
#     false_positives = ((y_pred == 1) & (y_true == 0)).astype(np.float32) * mask_float
#     true_negatives = ((y_pred == 0) & (y_true == 0)).astype(np.float32) * mask_float
    
#     FP = false_positives.sum()
#     TN = true_negatives.sum()
    
#     fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
#     return fpr

# # Calculate accuracy rate
# def accuracy_rate(y_pred, y_true, mask):
#     """
#     Calculates the accuracy rate.

#     Parameters:
#     y_pred (numpy array): Predicted labels
#     y_true (numpy array): True labels
#     mask (numpy array): Mask indicating the protected group

#     Returns:
#     float: Accuracy rate
#     """
#     correct_predictions = (y_pred == y_true) * mask.astype(np.float32)
#     accuracy = correct_predictions.sum() / mask.sum() if mask.sum() > 0 else 0
#     return accuracy

# # Calculate recall rate
# def recall_rate(y_pred, y_true, mask):
#     """
#     Calculates the recall rate.

#     Parameters:
#     y_pred (numpy array or pandas Series): Predicted labels
#     y_true (numpy array or pandas Series): True labels
#     mask (numpy array or pandas Series): Mask indicating the protected group

#     Returns:
#     float: Recall rate
#     """
#     if isinstance(y_pred, pd.Series):
#         y_pred = y_pred.values
#     if isinstance(y_true, pd.Series):
#         y_true = y_true.values
#     if isinstance(mask, pd.Series):
#         mask = mask.values
#     mask_float = mask.astype(np.float32)
#     true_positives = ((y_pred == 1) & (y_true == 1)).astype(float) * mask_float
#     actual_positives = (y_true == 1).astype(float) * mask_float
#     TP_sum = np.sum(true_positives)
#     AP_sum = np.sum(actual_positives)
#     recall = TP_sum / AP_sum if AP_sum > 0 else 0
    
#     return recall

    



        
    
        
