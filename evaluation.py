from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score, roc_auc_score,
                               roc_curve)

def evaluate(y_test, y_pred, y_probs):
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    fpr, tpr, threshold  = roc_curve(y_test, y_probs)
    return f1, roc_auc, fpr, tpr, threshold 
