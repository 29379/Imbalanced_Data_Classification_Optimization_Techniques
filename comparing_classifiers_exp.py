import numpy as np

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, RepeatedStratifiedKFold,
                                      learning_curve)
from sklearn.metrics import auc
from sklearn.svm import SVC
# from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                BalancedBaggingClassifier, RUSBoostClassifier)

from read_data import read_data
from visualisation import (print_fold_performance, print_mean_performance,
                            plot_mean_performance, plot_mean_roc_curve, plot_combined_roc_curves,
                            plot_learning_curve, plot_combined_learning_curves)
from evaluation import evaluate
from custom_ada_boost import CustomAdaBoostClassifier


classifiers = {
    "Naive Bayes": GaussianNB(),
    # "SVM": SVC(probability=True),
    # "Random Forest": BalancedRandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    # "EasyEnsemble": EasyEnsembleClassifier(),
    # "BalancedBagging": BalancedBaggingClassifier(),
    # "RUSBoost": RUSBoostClassifier(),
    # "BalancedRandomForest": BalancedRandomForestClassifier(),
    "Custom AdaBoost": CustomAdaBoostClassifier()
}

balancing_data_strategies_names = [
    "None",
    "SMOTE",
    "RandomUnderSampler"
]


strategies = [
    None,
    SMOTE(sampling_strategy='minority', random_state=42),
    RandomUnderSampler(sampling_strategy='majority', random_state=42)
]

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

X, y = read_data()

learning_curves = {}
roc_curves = {}

for model_name, model in classifiers.items():
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    balanced_accuracies = []
    roc_aucs = []

    # roc_auc curve components
    true_positive_rates = []
    mean_fpr = np.linspace(0, 1, 1000)  # common range for fpr

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]  

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, bal_acc, roc_auc, fpr, tpr, _ = evaluate(y_test, y_pred, y_probs)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        balanced_accuracies.append(bal_acc)
        roc_aucs.append(roc_auc)
        true_positive_rates.append(np.interp(mean_fpr, fpr, tpr))   # Interpolate tpr to the common fpr range
        true_positive_rates[-1][0] = 0.0    # curve starts at 0.0

        print_fold_performance(i, y_test, y_pred, acc, prec, rec, f1, bal_acc, roc_auc, model_name)

    mean_tpr = np.mean(true_positive_rates, axis=0)
    mean_tpr[-1] = 1.0  # ensure curve ends at (1, 1)
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    roc_curves[model_name] = (mean_fpr, mean_tpr, mean_roc_auc)

    print_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name)
    plot_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name)
    plot_mean_roc_curve(mean_fpr, mean_tpr, mean_roc_auc, model_name)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=rskf, scoring="roc_auc",n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores)
    learning_curves[model_name] = (train_sizes, train_scores, test_scores)

    plot_learning_curve(model, X, y, model_name, cv=rskf, scoring="roc_auc")

plot_combined_roc_curves(roc_curves)
plot_combined_learning_curves(learning_curves)
