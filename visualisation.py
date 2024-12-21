from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def evaluate(y_test, y_pred, y_probs):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    return acc, prec, rec, f1, bal_acc, roc_auc


def print_fold_performance(y_test, y_pred, acc, prec, rec, f1, bal_acc, roc_auc, model_name):
    print(f'{model_name} | Fold {i + 1}:')
    print(f'  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')
    print(f'  Balanced Accuracy: {bal_acc:.4f}, ROC-AUC: {roc_auc:.4f}')
    print(confusion_matrix(y_test, y_pred))


def print_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name):
    print(f'{model_name} | Mean scores:')
    print(f'Mean Accuracy: {np.mean(accuracies):.4f}')
    print(f'Mean Precision: {np.mean(precisions):.4f}')
    print(f'Mean Recall: {np.mean(recalls):.4f}')
    print(f'Mean F1: {np.mean(f1s):.4f}')
    print(f'Mean Balanced Accuracy: {np.mean(balanced_accuracies):.4f}')
    print(f'Mean ROC-AUC: {np.mean(roc_aucs):.4f}')


def plot_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name):
    metrics = {
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s,
        'Balanced Accuracy': balanced_accuracies,
        'ROC-AUC': roc_aucs,
    }

    # Plot
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', label=metric_name)

    plt.title(model_name)
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()
