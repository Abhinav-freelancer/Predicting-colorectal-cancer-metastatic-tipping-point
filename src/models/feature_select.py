"""
Mutual information feature selection for MPS model.
Selects top-k features by mutual information with the label.
Fit on training folds only to prevent selection leakage.
"""
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class MIFeatureSelector:
    def __init__(self, k: int = 30):
        self.k = k
        self.selector = None
        self.selected_indices = None
        self.selected_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        n_features = X.shape[1]
        k_actual = min(self.k, n_features)
        self.selector = SelectKBest(mutual_info_classif, k=k_actual)
        self.selector.fit(X, y)
        self.selected_indices = self.selector.get_support(indices=True)
        self.selected_names = [feature_names[i] for i in self.selected_indices]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return X
        return self.selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)
