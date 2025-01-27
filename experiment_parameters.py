from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids

from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                BalancedBaggingClassifier, RUSBoostClassifier)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from custom_ada_boost import CustomAdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


classifiers_ex01 = {
    "Naive Bayes": GaussianNB(),
    # "Logistic Regression0": LogisticRegression(class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight=None),
    # "Logistic Regression2": LogisticRegression(penalty="l2", C=0.1, solver="liblinear", class_weight="balanced", random_state=42),
    # "Logistic Regression3": LogisticRegression(penalty="l1", C=0.5, solver="saga", class_weight=None, random_state=42),
    # "Logistic Regression4": LogisticRegression(penalty=None, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42),
    # "Logistic Regression5": LogisticRegression(penalty="elasticnet", C=0.1, l1_ratio=0.5, solver="saga", class_weight=None, random_state=42),
    # "Logistic Regression6": LogisticRegression(penalty="l2", C=10.0, solver="saga", class_weight="balanced", random_state=42),
    # "Logistic Regression7": LogisticRegression(penalty="l2", C=1.0, solver="saga", class_weight=None, random_state=42),
    # "Logistic Regression9": LogisticRegression(penalty="l1", C=1.0, solver="saga", class_weight=None, random_state=42),
    # "Random Forest1": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2, class_weight='balanced_subsample', random_state=42),
    # "Random Forest2": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, class_weight=None, random_state=42),
    # "Random Forest8": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, class_weight=None, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=2, class_weight=None),
    # "Random Forest10": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, class_weight=None, random_state=42),
    # "Random Forest11": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, class_weight="balanced", random_state=42),
    # "Random Forest12": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", random_state=42),
    # "Random Forest3": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42),
    # "Random Forest4": RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=2, class_weight="balanced_subsample", random_state=42),
    # "Random Forest5": RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=10, class_weight=None, random_state=42),
    # "Random Forest6": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, class_weight="balanced", random_state=42),
    # "Random Forest7": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, criterion="entropy", class_weight="balanced_subsample", random_state=42),
    # "AdaBoost": AdaBoostClassifier(algorithm="SAMME", n_estimators=25, learning_rate=0.1, random_state=42),
    # "AdaBoost1": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME"),
    # "AdaBoost2": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm="SAMME"),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=1.0, algorithm="SAMME"),
    # "AdaBoost4": AdaBoostClassifier(n_estimators=200, learning_rate=0.1, algorithm="SAMME"),
    # "AdaBoost5": AdaBoostClassifier(n_estimators=50, learning_rate=0.05, algorithm="SAMME"),
    # "AdaBoost6": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, algorithm="SAMME"),
    "Custom AdaBoost": CustomAdaBoostClassifier(num_base_learners=25)
}

classifiers_ex02 = {
    "Logistic Regression": LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight=None),
    # "Custom AdaBoost": CustomAdaBoostClassifier(num_base_learners=25)
}    

balancing_data_strategies_ex02 = {
    "None": None,
    # "RandomOverSampler": RandomOverSampler(sampling_strategy="minority", random_state=42),
    # "RandomOverSampler1": RandomOverSampler(sampling_strategy=0.3, random_state=42),
    # "RandomOverSampler2": RandomOverSampler(sampling_strategy=0.5, random_state=100),
    "RandomOverSampler3": RandomOverSampler(sampling_strategy={0: 284315, 1: 2000}, random_state=42),
    # "SMOTE": SMOTE(sampling_strategy='minority', random_state=42),
    # "SMOTE1": SMOTE(sampling_strategy='minority', k_neighbors=3, random_state=42),
    # "SMOTE2": SMOTE(sampling_strategy=0.4, random_state=42),
    # "SMOTE3": BorderlineSMOTE(sampling_strategy="minority", random_state=42),
    "SMOTE4": SVMSMOTE(sampling_strategy="minority", random_state=42),
    # "RandomUnderSampler": RandomUnderSampler(sampling_strategy='majority', random_state=42),
    "RandomUnderSampler": RandomUnderSampler(sampling_strategy=0.7, random_state=42),
    # "RandomUnderSampler2": RandomUnderSampler(sampling_strategy={0: 10000, 1: 492}, random_state=42),
    # "RandomUnderSampler3": ClusterCentroids(sampling_strategy="majority", random_state=42),
    # "TomekLinks": TomekLinks(sampling_strategy="all"),
    "TomekLinks": TomekLinks(sampling_strategy="majority"),
}

classifiers_ex03 = {
    # "EasyEnsemble": EasyEnsembleClassifier(n_estimators=10, sampling_strategy='auto', random_state=42),
    # "EasyEnsemble1": EasyEnsembleClassifier(n_estimators=50, sampling_strategy='auto', random_state=42),
    "EasyEnsemble2": EasyEnsembleClassifier(n_estimators=10, sampling_strategy=0.5, random_state=42),
    # "RUSBoost": RUSBoostClassifier(n_estimators=100, learning_rate=0.1, sampling_strategy='auto', random_state=42),
    # "RUSBoost1": RUSBoostClassifier(n_estimators=200, learning_rate=0.1, sampling_strategy='auto', random_state=42),
    "RUSBoost2": RUSBoostClassifier(n_estimators=100, learning_rate=0.1, sampling_strategy=0.5, random_state=42),
    # "RUSBoost3": RUSBoostClassifier(n_estimators=100, learning_rate=0.05, sampling_strategy='auto', random_state=42),
    # "BalancedBagging": BalancedBaggingClassifier(n_estimators=10, sampling_strategy='auto', max_samples=0.5, random_state=42),
    # "BalancedBagging1": BalancedBaggingClassifier(n_estimators=50, sampling_strategy='auto', max_samples=0.5, random_state=42),
    # "BalancedBagging2": BalancedBaggingClassifier(n_estimators=10, sampling_strategy='auto', max_samples=0.3, random_state=42),
    "BalancedBagging3": BalancedBaggingClassifier(n_estimators=10, sampling_strategy=0.7, max_samples=0.5, random_state=42),
    # "BalancedRandomForest": BalancedRandomForestClassifier(n_estimators=100, max_depth=5, sampling_strategy='all', random_state=42),
    # "BalancedRandomForest1": BalancedRandomForestClassifier(n_estimators=100, max_depth=10, sampling_strategy='all', random_state=42),
    # "BalancedRandomForest2": BalancedRandomForestClassifier(n_estimators=100, max_depth=3, sampling_strategy='all', random_state=42),
    "BalancedRandomForest3": BalancedRandomForestClassifier(n_estimators=100, max_depth=5, sampling_strategy=0.4, random_state=42),

}
