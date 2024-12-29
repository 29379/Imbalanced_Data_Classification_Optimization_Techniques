from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                BalancedBaggingClassifier, RUSBoostClassifier)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_ada_boost import CustomAdaBoostClassifier
from sklearn.linear_model import LogisticRegression


classifiers_ex01 = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, class_weight='balanced_subsample', random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", n_estimators=25, learning_rate=0.1, random_state=42),
    "EasyEnsemble": EasyEnsembleClassifier(n_estimators=10, sampling_strategy='auto', random_state=42),
    "BalancedBagging": BalancedBaggingClassifier(n_estimators=10, sampling_strategy='auto', max_samples=0.5, random_state=42),
    "RUSBoost": RUSBoostClassifier(n_estimators=100, learning_rate=0.1, sampling_strategy='auto', random_state=42),
    "BalancedRandomForest": BalancedRandomForestClassifier(n_estimators=100, max_depth=5, sampling_strategy='all', random_state=42),
    "Custom AdaBoost": CustomAdaBoostClassifier(num_base_learners=25)
}

classifiers_ex02 = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, class_weight='balanced_subsample', random_state=42),
    "Custom AdaBoost": CustomAdaBoostClassifier(num_base_learners=25)
}    

balancing_data_strategies_ex02 = {
    "None": None,
    "SMOTE": SMOTE(sampling_strategy='minority', random_state=42),
    "RandomUnderSampler": RandomUnderSampler(sampling_strategy='majority', random_state=42)
}