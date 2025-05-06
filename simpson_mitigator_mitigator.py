import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .utils import preprocess_data, compute_fairness_metrics

class SimpsonMitigator:
    """A class to detect and mitigate Simpson's paradox in machine learning datasets."""
    
    def __init__(self, verbose=False):
        """
        Initialize the SimpsonMitigator.
        
        Parameters:
        - verbose: bool, whether to print detailed logs (default: False)
        """
        self.verbose = verbose

    def detect_confounders(self, data, x, y, categorical=False, threshold=0.5):
        """
        Detect confounders causing Simpson's paradox.
        
        Parameters:
        - data: pandas DataFrame, input dataset
        - x: str, column name of the impact factor
        - y: str, column name of the target variable
        - categorical: bool, whether to use categorical confounder detection
        - threshold: float, minimum confounding degree to consider (default: 0.5)
        
        Returns:
        - List of tuples (column, confounding_degree)
        """
        if x not in data.columns or y not in data.columns:
            raise ValueError(f"Columns {x} or {y} not found in dataset")
        
        data = preprocess_data(data)
        if self.verbose:
            print(f"Detecting confounders for {x} vs {y}, categorical={categorical}")
        
        if categorical:
            confounders = self._detect_confounders_categorical(data, x, y)
        else:
            confounders = self._detect_confounders_continuous(data, x, y)
        
        return [(col, degree) for col, degree in confounders if degree >= threshold]

    def _detect_confounders_continuous(self, data, x, y):
        """Detect confounders in continuous data (Algorithm 1)."""
        confounders = []
        try:
            aggregate_corr, _ = pearsonr(data[x], data[y])
        except ValueError as e:
            raise ValueError(f"Cannot compute correlation for {x} vs {y}: {e}")
        
        for col in data.columns:
            if col in [x, y] or data[col].dtype not in [np.float64, np.int64]:
                continue
            unique_values = data[col].unique()
            reversed_subgroups = 0
            for val in unique_values:
                subset = data[data[col] == val]
                if len(subset) < 2:
                    continue
                try:
                    corr, _ = pearsonr(subset[x], subset[y])
                    if np.sign(corr) != np.sign(aggregate_corr):
                        reversed_subgroups += 1
                except ValueError:
                    continue
            if len(unique_values) > 0:
                confounders.append((col, reversed_subgroups / len(unique_values)))
        
        return sorted(confounders, key=lambda x: x[1], reverse=True)

    def _detect_confounders_categorical(self, data, x, y):
        """Detect confounders in categorical data (Algorithm 2)."""
        confounders = []
        try:
            aggregate_corr, _ = pearsonr(data[x], data[y])
        except ValueError as e:
            raise ValueError(f"Cannot compute correlation for {x} vs {y}: {e}")
        
        for col in data.columns:
            if col in [x, y] or data[col].dtype.name != 'category':
                continue
            unique_values = data[col].unique()
            reversed_subgroups = 0
            for val in unique_values:
                subset = data[data[col] == val]
                if len(subset) < 2:
                    continue
                try:
                    corr, _ = pearsonr(subset[x], subset[y])
                    if np.sign(corr) != np.sign(aggregate_corr):
                        reversed_subgroups += 1
                except ValueError:
                    continue
            if len(unique_values) > 0:
                confounders.append((col, reversed_subgroups / len(unique_values)))
        
        return sorted(confounders, key=lambda x: x[1], reverse=True)

    def adjust_confounders(self, data, treatment, outcome, confounders):
        """
        Adjust confounders using Inverse Propensity Score Weighting (Algorithm 3).
        
        Parameters:
        - data: pandas DataFrame, input dataset
        - treatment: str, column name of the treatment variable
        - outcome: str, column name of the outcome variable
        - confounders: list of str, column names of confounders
        
        Returns:
        - pandas DataFrame with weighted outcome
        """
        if not all(col in data.columns for col in [treatment, outcome] + confounders):
            raise ValueError("Specified columns not found in dataset")
        
        data = preprocess_data(data)
        if self.verbose:
            print(f"Adjusting confounders: {confounders}")
        
        # Estimate propensity scores
        X = data[confounders]
        y = data[treatment]
        try:
            model = LogisticRegression(max_iter=1000).fit(X, y)
            propensity_scores = model.predict_proba(X)[:, 1]
        except ValueError as e:
            raise ValueError(f"Propensity score estimation failed: {e}")
        
        # Compute IPW weights
        weights = np.where(y == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))
        weights /= np.mean(weights)  # Normalize weights
        
        # Apply weights to outcome
        data[f'weighted_{outcome}'] = data[outcome] * weights
        return data

    def evaluate(self, data, model, features, target, protected_attribute):
        """
        Evaluate fairness and performance metrics after mitigation.
        
        Parameters:
        - data: pandas DataFrame, adjusted dataset
        - model: scikit-learn compatible model
        - features: list of str, feature columns
        - target: str, target column
        - protected_attribute: str, column for fairness evaluation
        
        Returns:
        - dict, containing accuracy, DPD, EOD
        """
        X = data[features]
        y = data[target]
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            **compute_fairness_metrics(data, y_pred, protected_attribute)
        }
        if self.verbose:
            print(f"Evaluation metrics: {metrics}")
        return metrics