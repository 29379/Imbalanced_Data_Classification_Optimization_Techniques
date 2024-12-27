from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                BalancedBaggingClassifier, RUSBoostClassifier)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from custom_ada_boost import CustomAdaBoostClassifier


classifiers = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Random Forest": BalancedRandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    "EasyEnsemble": EasyEnsembleClassifier(),
    "BalancedBagging": BalancedBaggingClassifier(),
    "RUSBoost": RUSBoostClassifier(),
    "BalancedRandomForest": BalancedRandomForestClassifier(),
    "Custom AdaBoost": CustomAdaBoostClassifier()
}

balancing_data_strategies = {
    "None": None,
    "SMOTE": SMOTE(sampling_strategy='minority', random_state=42),
    "RandomUnderSampler": RandomUnderSampler(sampling_strategy='majority', random_state=42)
}