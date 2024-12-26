from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


def print_fold_performance(fold_index, y_test, y_pred, acc, prec, rec, f1, bal_acc, roc_auc, model_name):
    print(f'{model_name} | Fold {fold_index + 1}:')
    print(f'  Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}')
    print(f'  Balanced Accuracy: {bal_acc:.3f}, ROC-AUC: {roc_auc:.3f}')
    print(confusion_matrix(y_test, y_pred))


def print_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name):
    print(f'{model_name} | Mean scores:')
    print(f'Mean Accuracy: {np.mean(accuracies):.3f}')
    print(f'Mean Precision: {np.mean(precisions):.3f}')
    print(f'Mean Recall: {np.mean(recalls):.3f}')
    print(f'Mean F1: {np.mean(f1s):.3f}')
    print(f'Mean Balanced Accuracy: {np.mean(balanced_accuracies):.3f}')
    print(f'Mean ROC-AUC: {np.mean(roc_aucs):.3f}\n\n')


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


def plot_mean_roc_curve(mean_fpr, mean_tpr, mean_roc_auc, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Add a random classifier baseline
    plt.title(f'{model_name} - Mean ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def plot_combined_roc_curves(roc_curves):
    plt.figure(figsize=(12, 8))
    
    for model_name, (mean_fpr, mean_tpr, mean_roc_auc) in roc_curves.items():
        plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {mean_roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Add a random classifier baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Curves")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_learning_curve(estimator, X, y, model_name, cv, scoring):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, color="g", label="Cross-validation score")

    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def plot_combined_learning_curves(learning_curves):
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Training Scores Across Models")
    for model_name, (train_sizes, train_scores, test_scores) in learning_curves.items():
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label=f"{model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("Training Score")
    plt.legend(loc="best")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("Validation Scores Across Models")
    for model_name, (train_sizes, train_scores, test_scores) in learning_curves.items():
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label=f"{model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("Validation Score")
    plt.legend(loc="best")
    plt.grid()

    plt.tight_layout()
    plt.show()
