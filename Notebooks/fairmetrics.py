import torch
import numpy as np 
import texttable as tt

def statistical_parity(y_pred, mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    y_cond = y_pred * mask.astype(np.float32)
    return y_cond.sum() / mask.sum()

def equal_opportunity(y_pred, y, mask):
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask)
    y_cond = y * y_pred * mask.float()  
    return y_cond.sum() / (y * mask.float()).sum() 


def test_error(y_pred,y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return np.mean(y_pred!=y_true)


# def confusion_matrix(y_pred, y_true, mask=None):
#     y_pred = torch.tensor(y_pred)
#     y_true = torch.tensor(y_true)
#     size = y_true.size(0) if mask is None else mask.sum().item()
#     sum_and_int = lambda x: x.sum().long().item()
#     to_percentage = lambda l: [f'{float(y)* 100. / size :.2f}%' for y in l]
#     y_pred_binary = (y_pred > 0.5).float()
#     if mask is None:
#         mask = torch.ones_like(y_pred)  
#     if isinstance(mask, np.ndarray):
#         mask = torch.tensor(mask)
#     mask = mask.float()
#     true_positives = sum_and_int(y_true * y_pred_binary * mask)
#     false_positives = sum_and_int((1 - y_true) * y_pred_binary * mask)
#     true_negatives = sum_and_int((1 - y_true) * (1 - y_pred_binary) * mask)
#     false_negatives = sum_and_int(y_true * (1 - y_pred_binary) * mask)
#     total = true_positives + false_positives + true_negatives + false_negatives
#     table = tt.Texttable()
#     table.header(['Real/Pred', 'Positive', 'Negative', ''])
#     table.add_row(['Positive'] + to_percentage([true_positives, false_negatives, true_positives + false_negatives]))
#     table.add_row(['Negative'] + to_percentage([false_positives, true_negatives, false_positives + true_negatives]))
#     table.add_row([''] + to_percentage([true_positives + false_positives, false_negatives + true_negatives, total]))
    
#     return table



def get_statistics(model, X_test, y_test, pred, mask, confusion=True):
    print(f"Test error is {round(test_error(pred,y_test), 2)} and accuracy is {round(1-test_error(pred,y_test),2)} ")
    # if confusion == True:
    #     print("Confusion matrix all: ")
    #     print(confusion_matrix(pred,y_test).draw())
    #     print("Confusion matrix unprotected: ")
    #     print(confusion_matrix(pred,y_test,mask).draw())
    #     print("Confusion matrix protected: ")
    #     print(confusion_matrix(pred,y_test,1-mask).draw())
    pred = torch.tensor(pred)
    y_test = torch.tensor(y_test)
    print("statistical_parity for unprotected: %.3f, for protected: %.3f, difference: %.3f" % (statistical_parity(pred, mask),statistical_parity(pred,1-mask), statistical_parity(pred,mask)-statistical_parity(pred,1-mask)))
    print("equal_opportunity for unprotected: %.3f, for protected: %.3f, difference: %.3f" % (equal_opportunity(pred, y_test,mask),equal_opportunity(pred, y_test,1-mask), equal_opportunity(pred, y_test,mask) - equal_opportunity(pred, y_test,1-mask)))