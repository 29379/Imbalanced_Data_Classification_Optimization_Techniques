import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier, RUSBoostClassifier

from read_data import read_data
from visualisation import evaluate, print_fold_performance, print_mean_performance, plot_mean_performance


classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Random Forest": BalancedRandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "EasyEnsemble": EasyEnsembleClassifier(),
    "BalancedBagging": BalancedBaggingClassifier(),
    "RUSBoost": RUSBoostClassifier(),
    "BalancedRandomForest": BalancedRandomForestClassifier(),
    "Custom AdaBoost": AdaBoostClassifier()
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


X, y = read_data()

for model_name, model in classifiers.items():
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    balanced_accuracies = []
    roc_aucs = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]   

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, bal_acc, roc_auc = evaluate(y_test, y_pred, y_probs)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        balanced_accuracies.append(bal_acc)
        roc_aucs.append(roc_auc)

        print_fold_performance(y_test, y_pred, acc, prec, rec, f1, bal_acc, roc_auc, model_name)

    print_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name)
    plot_mean_performance(accuracies, precisions, recalls, f1s, balanced_accuracies, roc_aucs, model_name)