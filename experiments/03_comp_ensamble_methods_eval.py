import os, sys
from tabulate import tabulate
from scipy.stats import ttest_rel
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from visualisation_methods import (plot_combined_learning_curves, plot_combined_roc_curves, plot_learning_curve, plot_mean_performance, plot_mean_roc_curve)
from experiment_parameters import classifiers_ex03

results_data_dir = 'results/data/comparing_ensamble_methods_exp_03'
results_visualisations_dir = 'results/visualisations/comparing_ensamble_methods_exp_03'

models = classifiers_ex03.keys()
n_models = len(classifiers_ex03)
n_folds = None

f1_scores = {model: [] for model in models}
auc_scores = {model: [] for model in models}

for model_name in models:
    model_files = [
        file for file in os.listdir(results_data_dir)
        if file.startswith(f"{model_name}_results") and file.endswith(".npy")
    ]
    model_files.sort()

    for file in model_files:
        results = np.load(os.path.join(results_data_dir, file), allow_pickle=True).item()
        f1_scores[model_name].extend(results["f1s"])
        auc_scores[model_name].extend(results["roc_aucs"])
    if n_folds is None:
        n_folds = len(f1_scores)

f1_scores = {model: np.array(scores) for model, scores in f1_scores.items()}
auc_scores = {model: np.array(scores) for model, scores in auc_scores.items()}

# Calculate mean and standard deviation for each model
mean_f1_scores = []
mean_auc_scores = []
std_f1_scores = []
std_auc_scores = []

for model_name in models:
    mean_f1_scores.append(np.mean(f1_scores[model_name]))
    mean_auc_scores.append(np.mean(auc_scores[model_name]))
    std_f1_scores.append(np.std(f1_scores[model_name]))
    std_auc_scores.append(np.std(auc_scores[model_name]))

print(f"\n### Statistical Evaluation for Dataset: Credit Card Fraud Detection ###\n")
table = tabulate([mean_f1_scores, mean_auc_scores], 
                 tablefmt="grid", 
                 headers=models, 
                 showindex=["Mean F1", "Mean AUC"])
print("\nMean Scores:\n", table)

table = tabulate([std_f1_scores, std_auc_scores], 
                 tablefmt="grid", 
                 headers=models, 
                 showindex=["Std Dev F1", "Std Dev AUC"])
print("\nStandard Deviations:\n", table)

# Perform paired t-tests
stat_mat_f1 = np.zeros((n_models, n_models))
stat_mat_auc = np.zeros((n_models, n_models))
model_list = list(models)

for j in range(n_models):
    for k in range(n_models):
        if j != k:  # Skip self-comparison
            _, p_value_f1 = ttest_rel(f1_scores[model_list[j]], f1_scores[model_list[k]])
            _, p_value_auc = ttest_rel(auc_scores[model_list[j]], auc_scores[model_list[k]])
            stat_mat_f1[j, k] = p_value_f1 < 0.05  # Significant if p < 0.05
            stat_mat_auc[j, k] = p_value_auc < 0.05

# Display statistical significance matrices
print("\nStatistical Significance Matrix for F1 (1: Significant, 0: Not Significant):\n")
table_f1 = tabulate(stat_mat_f1, 
                     tablefmt="grid", 
                     headers=models, 
                     showindex=models)
print(table_f1)

print("\nStatistical Significance Matrix for AUC (1: Significant, 0: Not Significant):\n")
table_auc = tabulate(stat_mat_auc, 
                      tablefmt="grid", 
                      headers=models, 
                      showindex=models)
print(table_auc)
