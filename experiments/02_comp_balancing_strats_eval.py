import os, sys
from tabulate import tabulate
from scipy.stats import ttest_rel
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from experiment_parameters import classifiers_ex02, balancing_data_strategies_ex02
results_data_dir = 'results/data/comparing_balancing_strats_exp_02'
classifiers = list(classifiers_ex02.keys())
strategies = list(balancing_data_strategies_ex02.keys())
EXPECTED_FOLDS = 10  

# Helper function to pad or trim scores
def adjust_folds(scores, expected_folds):
    if len(scores) < expected_folds:
        scores.extend([np.nan] * (expected_folds - len(scores)))  # Pad with NaNs
    elif len(scores) > expected_folds:
        scores = scores[:expected_folds]  # Trim to expected folds
    return scores


def aggregate_results(file_pattern, expected_folds):
    f1s, roc_aucs = [], []
    files = [
        file for file in os.listdir(results_data_dir)
        if file.startswith(file_pattern) and file.endswith(".npy")
    ]
    files.sort()

    for file in files:
        results = np.load(os.path.join(results_data_dir, file), allow_pickle=True).item()
        f1s.extend(results["f1s"])
        roc_aucs.extend(results["roc_aucs"])
    
    f1s = adjust_folds(f1s, expected_folds*len(files))
    roc_aucs = adjust_folds(roc_aucs, expected_folds*len(files))
    return f1s, roc_aucs


results_dict = {strategy: {classifier: {"f1s": [], "roc_aucs": []} for classifier in classifiers} for strategy in strategies}

for strategy in strategies:
    for classifier in classifiers:
        file_pattern = f"{strategy}_{classifier}_results"
        f1s, roc_aucs = aggregate_results(file_pattern, EXPECTED_FOLDS)

        results_dict[strategy][classifier]["f1s"] = f1s
        results_dict[strategy][classifier]["roc_aucs"] = roc_aucs

# Generate tables and perform t-tests for each classifier
for classifier in classifiers:
    print(f"\n### Results for Classifier: {classifier} ###\n")

    f1_scores = np.array([results_dict[strategy][classifier]["f1s"] for strategy in strategies])
    auc_scores = np.array([results_dict[strategy][classifier]["roc_aucs"] for strategy in strategies])

    mean_f1_scores = np.nanmean(f1_scores, axis=1)
    std_f1_scores = np.nanstd(f1_scores, axis=1)
    mean_auc_scores = np.nanmean(auc_scores, axis=1)
    std_auc_scores = np.nanstd(auc_scores, axis=1)

    table_data = [
        ["Mean F1"] + mean_f1_scores.tolist(),
        ["Std Dev F1"] + std_f1_scores.tolist(),
        ["Mean AUC"] + mean_auc_scores.tolist(),
        ["Std Dev AUC"] + std_auc_scores.tolist(),
    ]

    table = tabulate(table_data, headers=strategies, tablefmt="grid", showindex=False)
    print(table)

    print("\n### Paired t-tests for Statistical Significance ###\n")
    f1_significance = np.zeros((len(strategies), len(strategies)), dtype=int)
    auc_significance = np.zeros((len(strategies), len(strategies)), dtype=int)

    for i in range(len(strategies)):
        for j in range(len(strategies)):
            if i != j:  
                _, p_value_f1 = ttest_rel(f1_scores[i, :], f1_scores[j, :], nan_policy='omit')
                f1_significance[i, j] = int(p_value_f1 < 0.05)

                _, p_value_auc = ttest_rel(auc_scores[i, :], auc_scores[j, :], nan_policy='omit')
                auc_significance[i, j] = int(p_value_auc < 0.05)

    # Display statistical significance matrices
    print("Statistical Significance Matrix for F1 (1: Significant, 0: Not Significant):\n")
    table_f1 = tabulate(f1_significance, headers=strategies, showindex=strategies, tablefmt="grid")
    print(table_f1)

    print("\nStatistical Significance Matrix for AUC (1: Significant, 0: Not Significant):\n")
    table_auc = tabulate(auc_significance, headers=strategies, showindex=strategies, tablefmt="grid")
    print(table_auc)
