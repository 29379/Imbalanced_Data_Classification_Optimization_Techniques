import numpy as np
import os, sys, multiprocessing

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, RepeatedStratifiedKFold,
                                      learning_curve)
from sklearn.metrics import auc
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.pipeline import make_pipeline

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from read_data import read_data
from visualisation_methods import print_fold_performance, print_mean_performance
from evaluation import evaluate
from experiment_parameters import classifiers_ex01

def evaluate_classifier(model_name, model, X, y, rskf, results_data_dir, experiment_results):
    print(f"Running experiment for {model_name}...\n")

    classifier_results = {
        "accuracies": [],
        "precisions": [],
        "recalls": [],
        "f1s": [],
        "balanced_accuracies": [],
        "roc_aucs": [],
        "true_positive_rates": [],
        "mean_fpr": np.linspace(0, 1, 1000), # common range for fpr
        "mean_tpr": [],
        "learning_curve": {}
    }

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, bal_acc, roc_auc, fpr, tpr, _ = evaluate(y_test, y_pred, y_probs)
        
        classifier_results["accuracies"].append(acc)
        classifier_results["precisions"].append(prec)
        classifier_results["recalls"].append(rec)
        classifier_results["f1s"].append(f1)
        classifier_results["balanced_accuracies"].append(bal_acc)
        classifier_results["roc_aucs"].append(roc_auc)

        tpr_interp = np.interp(classifier_results["mean_fpr"], fpr, tpr)  # Interpolate tpr to the common fpr range
        tpr_interp[0] = 0.0  # Ensure curve starts at (0, 0)
        classifier_results["true_positive_rates"].append(tpr_interp)

        print_fold_performance(i, y_test, y_pred, acc, prec, rec, f1, bal_acc, roc_auc, model_name)

    mean_tpr = np.mean(classifier_results["true_positive_rates"], axis=0)
    mean_tpr[-1] = 1.0  # ensure curve ends at (1, 1)
    classifier_results["mean_tpr"] = mean_tpr

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=rskf, scoring="roc_auc", n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    classifier_results["learning_curve"] = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores
    }

    experiment_results[model_name] = classifier_results
    np.save(os.path.join(results_data_dir, f"{model_name}_results.npy"), classifier_results)

    print_mean_performance(classifier_results["accuracies"], classifier_results["precisions"], classifier_results["recalls"], 
                        classifier_results["f1s"], classifier_results["balanced_accuracies"], classifier_results["roc_aucs"], model_name)


def main():
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/data'):
        os.makedirs('results/data')
    if not os.path.exists('results/visualisations'):
        os.makedirs('results/visualisations')
    results_data_dir = 'results/data/comparing_classifiers_exp_01'
    results_visualisations_dir = 'results/visualisations/comparing_classifiers_exp_01'
    if not os.path.exists(results_data_dir):
        os.makedirs(results_data_dir)
    if not os.path.exists(results_visualisations_dir):
        os.makedirs(results_visualisations_dir)

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

    X, y = read_data()
    experiment_results = {}
    processes = []

    for model_name, model in classifiers_ex01.items():
        process = multiprocessing.Process(target=evaluate_classifier, args=(model_name, model, X, y, rskf, results_data_dir, experiment_results))
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    np.save(os.path.join(results_data_dir, "all_experiment_results.npy"), experiment_results)


if __name__ == "__main__":
    main()