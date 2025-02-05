from sklearn.metrics import confusion_matrix, auc
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_in_2d(X, y, title='Classes'):
    plt.figure(figsize=(8, 8))

    colors = ['#0c6ac1', '#eb5141']
    for label, color in zip(np.unique(y), colors):
        plt.scatter(
            X[y==label, 0],
            X[y==label, 1],
            c=color, label=label, alpha=0.8
        )
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


def print_fold_performance(fold_index, f1, roc_auc, model_name):
    print(f'{model_name} | Fold {fold_index + 1}:')
    print(f'  F1: {f1:.3f}')
    print(f'  ROC-AUC: {roc_auc:.3f}')


def print_mean_performance(f1s, roc_aucs, model_name):
    print(f'{model_name} | Mean scores:')
    print(f'Mean F1: {np.mean(f1s):.3f}')
    print(f'Mean ROC-AUC: {np.mean(roc_aucs):.3f}\n\n')


def plot_mean_performance(model_name, results_data_dir, results_visualisations_dir):
    print(f"Plotting mean performance for {model_name}...\n")
    results = np.load(os.path.join(results_data_dir, f"{model_name}_results.npy"), allow_pickle=True).item()
    metrics = {
        'F1': results['f1s'],
        'ROC-AUC': results['roc_aucs'],
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
    plt.tight_layout()
    plt.savefig(os.path.join(results_visualisations_dir, f"{model_name}_mean_performance.png"))
    plt.close()


def plot_mean_roc_curve(model_name, results_data_dir, results_visualisations_dir):
    print(f"Plotting mean ROC curve for {model_name}...\n")
    results = np.load(os.path.join(results_data_dir, f"{model_name}_results.npy"), allow_pickle=True).item()
    mean_fpr = results["mean_fpr"]
    mean_tpr = results['mean_tpr']
    mean_roc_auc = auc(mean_fpr, mean_tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Add a random classifier baseline
    plt.title(f'{model_name} - Mean ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_visualisations_dir, f"{model_name}_mean_roc_curve.png"))
    plt.close()


def plot_combined_roc_curves(results_data_dir, results_visualisations_dir):
    print("Plotting combined ROC curves...\n")
    plt.figure(figsize=(12, 8))
    
    for file in os.listdir(results_data_dir):
        if file.endswith("_results.npy") and file != "all_experiment_results.npy":
            model_name = file.replace("_results.npy", "")
            results = np.load(os.path.join(results_data_dir, file), allow_pickle=True).item()
            mean_fpr = np.linspace(0, 1, 1000) # common range for fpr
            mean_tpr = results["mean_tpr"]
            mean_roc_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {mean_roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Add a random classifier baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparison of ROC Curves")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_visualisations_dir, "combined_roc_curves.png"))
    plt.close()


def plot_learning_curve(model_name, results_data_dir, results_visualisations_dir):
    print(f"Plotting learning curve for {model_name}...\n")
    results = np.load(os.path.join(results_data_dir, f"{model_name}_results.npy"), allow_pickle=True).item()
    # train_sizes, train_scores, test_scores = learning_curve(
    #     estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    # )
    learning_curve_data = results["learning_curve"]
    train_sizes = learning_curve_data["train_sizes"]
    train_scores = learning_curve_data["train_scores"]
    test_scores = learning_curve_data["test_scores"]

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
    plt.tight_layout()
    plt.savefig(os.path.join(results_visualisations_dir, f"{model_name}_learning_curve.png"))
    plt.close()


def plot_combined_learning_curves(results_data_dir, results_visualisations_dir):
    print("Plotting combined learning curves...\n")
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Training Scores Across Models")

    for file in os.listdir(results_data_dir):
        if file.endswith("_results.npy") and file != "all_experiment_results.npy":
            model_name = file.replace("_results.npy", "")
            
            results = np.load(os.path.join(results_data_dir, file), allow_pickle=True).item()
            learning_curve_data = results["learning_curve"]
            train_sizes = learning_curve_data["train_sizes"]
            train_scores = learning_curve_data["train_scores"]
            
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label=f"{model_name}")

    plt.xlabel("Training Examples")
    plt.ylabel("Training Score")
    plt.legend(loc="best")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title("Validation Scores Across Models")

    for file in os.listdir(results_data_dir):
        if file.endswith("_results.npy") and file != "all_experiment_results.npy":
            model_name = file.replace("_results.npy", "")
            
            results = np.load(os.path.join(results_data_dir, file), allow_pickle=True).item()
            learning_curve_data = results["learning_curve"]
            train_sizes = learning_curve_data["train_sizes"]
            test_scores = learning_curve_data["test_scores"]
            
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label=f"{model_name}")

    plt.xlabel("Training Examples")
    plt.ylabel("Validation Score")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_visualisations_dir, "combined_learning_curves.png"))
    plt.close()
    