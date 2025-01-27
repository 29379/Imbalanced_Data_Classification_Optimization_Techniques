import numpy as np
import os, sys, multiprocessing

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, RepeatedStratifiedKFold,
                                      learning_curve)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from read_data import read_data
from visualisation_methods import print_fold_performance, print_mean_performance
from evaluation import evaluate
from experiment_parameters import classifiers_ex03

from visualisation_methods import plot_in_2d

def evaluate_classifier(model_name, model, X, y, rskf, results_data_dir, experiment_results):
    print(f"Running experiment for {model_name}...\n")

    classifier_results = {
        "f1s": [],
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

        f1, roc_auc, fpr, tpr, _  = evaluate(y_test, y_pred, y_probs)
        
        classifier_results["f1s"].append(f1)
        classifier_results["roc_aucs"].append(roc_auc)

        tpr_interp = np.interp(classifier_results["mean_fpr"], fpr, tpr)  # Interpolate tpr to the common fpr range
        tpr_interp[0] = 0.0  # Ensure curve starts at (0, 0)
        classifier_results["true_positive_rates"].append(tpr_interp)

        print_fold_performance(i, f1, roc_auc, model_name)

    mean_tpr = np.mean(classifier_results["true_positive_rates"], axis=0)
    mean_tpr[-1] = 1.0  # ensure curve ends at (1, 1)
    classifier_results["mean_tpr"] = mean_tpr

    experiment_results[model_name] = classifier_results
    # np.save(os.path.join(results_data_dir, f"{model_name}_results.npy"), classifier_results)
    base_filename = os.path.join(results_data_dir, f"{model_name}_results")
    filename = base_filename + ".npy"
    i = 0
    while os.path.exists(filename):
        i += 1
        filename = f"{base_filename}_{i}.npy"
    np.save(filename, classifier_results)

    print_mean_performance(classifier_results["f1s"], classifier_results["roc_aucs"], model_name)


def main():
    X, y = read_data()
    # plot_in_2d(X, y, "Classes")

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/data'):
        os.makedirs('results/data')
    if not os.path.exists('results/visualisations'):
        os.makedirs('results/visualisations')
    results_data_dir = 'results/data/comparing_ensamble_methods_exp_03'
    results_visualisations_dir = 'results/visualisations/comparing_ensamble_methods_exp_03'
    if not os.path.exists(results_data_dir):
        os.makedirs(results_data_dir)
    if not os.path.exists(results_visualisations_dir):
        os.makedirs(results_visualisations_dir)

    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

    experiment_results = {}
    processes = []

    for model_name, model in classifiers_ex03.items():
        process = multiprocessing.Process(target=evaluate_classifier, args=(model_name, model, X, y, rskf, results_data_dir, experiment_results))
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    np.save(os.path.join(results_data_dir, "all_experiment_results.npy"), experiment_results)


if __name__ == "__main__":
    for i in range(10):
        main()
        print(f"Finsihed iteration {i+1} - experiment 03")