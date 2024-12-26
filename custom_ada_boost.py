import numpy as np
from typing import Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomAdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_base_learners=10, epsilon=1e-10) -> None:
        self.num_base_learners = num_base_learners  # base learners - decision trees with max_depth=1
        self.X_train = None
        self.y_train = None
        self.label_count = 0
        self.base_learners = {} # dictionary of DecisionTrees and the amount of say (float) they have
        self.epsilon = epsilon

    def fit(self, X_train: np.array, y_train: np.array) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)  # Store the unique classes for learning curve plotting

        X_bootstrapped = X_train
        y_bootstrapped = y_train
        self.label_count = len(np.unique(y_train))

        for i in range(self.num_base_learners):
            base_learner, amount_of_say = self._fit_base_learner(X_bootstrapped, y_bootstrapped)
            self.base_learners[base_learner] = amount_of_say
            sample_weights = self._calculate_sample_weights(base_learner, amount_of_say)
            X_bootstrapped, y_bootstrapped = self._update_dataset(sample_weights)

    def predict(self, X: np.array) -> np.array:
        pred_probabilities = self.predict_proba(X)
        preds = np.argmax(pred_probabilities, axis=1)
        return preds
    
    def predict_proba(self, X: np.array) -> np.array:
        pred_probs = []
        base_learner_pred_scores = self._predict_scores_with_base_learners(X)

        # take the average scores and turn them to probabilities
        avg_base_learners_pred_scores = np.mean(base_learner_pred_scores, axis=0)
        column_sums = np.sum(avg_base_learners_pred_scores, axis=1)

        for i in range(avg_base_learners_pred_scores.shape[0]):
            pred_probs.append(avg_base_learners_pred_scores[i] / column_sums[i])

        return np.array(pred_probs)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def _fit_base_learner(self, X_bootstrapped: np.array, y_bootstrapped: np.array) -> tuple:
        base_learner = DecisionTreeClassifier(max_depth=1)
        base_learner.fit(X_bootstrapped, y_bootstrapped)
        amount_of_say = self._calculate_amount_of_say(base_learner, self.X_train, self.y_train)

        return base_learner, amount_of_say

    def _calculate_sample_weights(self, base_learner: DecisionTreeClassifier, amount_of_say: int) -> np.array:
        predictions = base_learner.predict(self.X_train)
        matches = (predictions == self.y_train)
        errors = (~matches).astype(int)
        sample_weights = (1/self.X_train.shape[0]) * np.exp(amount_of_say * errors)
        sample_weights = np.nan_to_num(sample_weights, nan=self.epsilon)  # replace NaNs with epsilon
        sample_weights = np.clip(sample_weights, self.epsilon, None)  # guard against zeros
        sample_weights /= np.sum(sample_weights)    # normalize
        return sample_weights

    def _update_dataset(self, sample_weights: np.array) -> tuple:  # creating bootstrap samples
        n_samples = self.X_train.shape[0]
        bootstrap_indicies = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
        X_bootstrapped = self.X_train[bootstrap_indicies]
        y_bootstrapped = self.y_train[bootstrap_indicies]
        return X_bootstrapped, y_bootstrapped
        
    def _calculate_amount_of_say(self, base_learner: DecisionTreeClassifier, X: np.array, y: np.array) -> float:
        K = self.label_count
        preds = base_learner.predict(X)
        err = 1 - np.sum(preds==y) / preds.shape[0]        
        amount_of_say = np.log((1 - err) / (err + self.epsilon)) + np.log(K-1)
        return amount_of_say

    def _predict_scores_with_base_learners(self, X:np.array) -> list:
        pred_scores = np.zeros(shape=(self.num_base_learners, X.shape[0], self.label_count))

        for idx, (base_learner, amount_of_say) in enumerate(self.base_learners.items()):
            if idx >= self.num_base_learners:
                break
            pred_probs = base_learner.predict_proba(X)
            pred_scores[idx] = pred_probs * amount_of_say
        
        return pred_scores

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def get_params(self, deep=True):
        return {'num_base_learners': self.num_base_learners}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    