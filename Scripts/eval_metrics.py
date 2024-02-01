import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# test error
def test_error(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return np.mean(y_pred!=y_true)

# demographic parity
def demographic_parity(y_pred, mask):
    y_cond = y_pred * mask.astype(np.float32)
    return y_cond.sum() / mask.sum()

# equality of opportunity
def equal_opportunity(y_pred, y_true, mask):
    y_cond = y_true * y_pred * mask.astype(np.float32)
    return y_cond.sum() / (y_true * mask.astype(np.float32)).sum()


# false positive rate
def false_positive_rate(y_pred, y_true, mask):
    # Ensure all inputs are NumPy arrays to avoid pandas-related TypeError
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

# accuracy rate
def accuracy_rate(y_pred, y_true, mask):
    correct_predictions = (y_pred == y_true) * mask.astype(np.float32)
    accuracy = correct_predictions.sum() / mask.sum() if mask.sum() > 0 else 0
    return accuracy

# recall
def recall_rate(y_pred, y_true, mask):
    # Ensure inputs are numpy arrays if they're not already (if they're pandas Series)
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



def model_metrics(model_name, Xtest_data, ytest_data, mask=None,  fair_metrics= True):
    y_preds = model_name.predict(Xtest_data)
    print(f'Model accuracy {round(accuracy_score(y_preds, ytest_data), 2)}')
    print(f'Model F1score {round(f1_score(y_preds, ytest_data), 2)}')
    print(f'Model Recall {round(recall_score(y_preds, ytest_data), 2)} and precision {round(precision_score(y_preds, ytest_data), 2)}')
    print(f'Model Test error is {round(test_error(y_preds, ytest_data), 2)}')
    if fair_metrics:
        mydemographic_parity = demographic_parity(y_preds, mask)
        equality_opportunity = equal_opportunity(y_preds, ytest_data, mask)
        fpr = false_positive_rate(y_preds, ytest_data, mask)
        myaccuracy_rate= accuracy_rate(y_preds, ytest_data, mask)
        myrecall_rate= recall_rate(y_preds, ytest_data, mask)
        print(f'DP is {demographic_parity} EO is {equality_opportunity} fpr is {fpr} accuracy_rate is {myaccuracy_rate} recall rate is {myrecall_rate}')
    return mydemographic_parity, equality_opportunity, fpr, myaccuracy_rate, myrecall_rate

        
        
        