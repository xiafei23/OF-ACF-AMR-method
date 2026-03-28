"""
OF-ACF Feature Extraction Module (Clean Version)
================================================
Core functionality only. Demo, synthetic data generation, and hard-coded
experiment settings have been removed.

This module provides:
  1. 52-dim HOC feature extraction
  2. Elastic Net + Domain Prior feature screening

Extensibility:
  - External dataset loaders should provide signals (N, L) and labels (N,)
  - Hyperparameters are fully configurable via class initialization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Section 1: Higher-Order Cumulant Computation
# =============================================================================

def compute_moments(signal: np.ndarray) -> Dict[str, complex]:
    x = signal
    xc = np.conj(signal)

    return {
        'M20': np.mean(x ** 2),
        'M21': np.mean(x * xc),
        'M40': np.mean(x ** 4),
        'M41': np.mean(x ** 3 * xc),
        'M42': np.mean(x ** 2 * xc ** 2),
        'M60': np.mean(x ** 6),
        'M61': np.mean(x ** 5 * xc),
        'M62': np.mean(x ** 4 * xc ** 2),
        'M63': np.mean(x ** 3 * xc ** 3),
    }


def compute_cumulants(signal: np.ndarray) -> Dict[str, complex]:
    M = compute_moments(signal)

    C = {}
    C['C20'] = M['M20']
    C['C21'] = M['M21']

    C['C40'] = M['M40'] - 3.0 * M['M20'] ** 2
    C['C41'] = M['M41'] - 3.0 * M['M21'] * M['M20']
    C['C42'] = M['M42'] - np.abs(M['M20']) ** 2 - 2.0 * M['M21'] ** 2

    C['C60'] = M['M60'] - 15.0 * M['M40'] * M['M20'] + 30.0 * M['M20'] ** 3
    C['C61'] = (M['M61'] - 5.0 * M['M42'] * M['M20']
                - 10.0 * M['M41'] * M['M21']
                + 30.0 * M['M21'] * M['M20'] ** 2)
    C['C62'] = (M['M62'] - 6.0 * M['M42'] * M['M20']
                - 8.0 * M['M41'] * M['M21']
                - M['M40'] * np.conj(M['M20'])
                + 6.0 * M['M20'] ** 2 * np.conj(M['M20'])
                + 24.0 * M['M21'] ** 2 * M['M20'])
    C['C63'] = (M['M63'] - 9.0 * M['M42'] * M['M21']
                + 12.0 * M['M21'] ** 3
                - 3.0 * M['M40'] * np.conj(M['M20'])
                - 3.0 * M['M20'] * np.conj(M['M40'])
                + 18.0 * M['M20'] * np.conj(M['M20']) * M['M21'])

    return C


# =============================================================================
# Section 2: Feature Construction
# =============================================================================

HOC_FEATURES = ['C20', 'C21', 'C40', 'C41', 'C42', 'C60', 'C61', 'C62', 'C63']

NORMALIZED_FEATURES = [
    ('C21', 'C20'), ('C40', 'C20'), ('C40', 'C21'), ('C41', 'C40'),
    ('C42', 'C20'), ('C42', 'C21'), ('C60', 'C20'), ('C60', 'C21'),
    ('C60', 'C40'), ('C60', 'C42'), ('C61', 'C20'), ('C61', 'C21'),
    ('C61', 'C60'), ('C62', 'C21'), ('C62', 'C41'), ('C62', 'C42'),
    ('C62', 'C60'), ('C62', 'C40'), ('C63', 'C41'), ('C63', 'C42'),
]

QUADRATIC_FEATURES = [
    ('C40', 'C20'), ('C41', 'C20'), ('C41', 'C40'), ('C42', 'C20'),
    ('C42', 'C40'), ('C42', 'C41'), ('C60', 'C20'), ('C60', 'C40'),
    ('C60', 'C41'), ('C61', 'C20'), ('C61', 'C21'), ('C61', 'C40'),
    ('C61', 'C41'), ('C61', 'C42'), ('C62', 'C20'), ('C62', 'C21'),
    ('C62', 'C40'), ('C62', 'C42'), ('C62', 'C60'), ('C62', 'C61'),
    ('C63', 'C20'), ('C63', 'C21'), ('C63', 'C61'),
]


def extract_52_features(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    C = compute_cumulants(signal)
    C_abs = {k: np.abs(v) for k, v in C.items()}

    features = []
    for name in HOC_FEATURES:
        features.append(C_abs[name])

    for (num, den) in NORMALIZED_FEATURES:
        features.append(C_abs[num] / (C_abs[den] + eps))

    for (num, den) in QUADRATIC_FEATURES:
        features.append(C_abs[num] / (C_abs[den] ** 2 + eps))

    return np.array(features, dtype=np.float64)


def extract_features_batch(signals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.array([extract_52_features(sig, eps) for sig in signals])


# =============================================================================
# Section 3: Elastic Net Screener (Core Only)
# =============================================================================

class ElasticNetScreener:
    def __init__(self,
                 l1_ratio: float = 0.5,
                 alpha: Optional[float] = None,
                 n_folds: int = 5,
                 group_quota: Tuple[int, int, int] = (9, 10, 16),
                 w_en: float = 0.1,
                 w_prior: float = 0.9,
                 random_state: int = 42):

        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.n_folds = n_folds
        self.group_quota = group_quota
        self.w_en = w_en
        self.w_prior = w_prior
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.selected_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler_.fit_transform(X)
        en_importance = self._compute_en_importance(X_scaled, y)
        prior = np.ones(X.shape[1])  # simplified hook

        composite = self._normalize(en_importance) * self.w_en + \
                    self._normalize(prior) * self.w_prior

        self.selected_indices_ = np.argsort(composite)[-sum(self.group_quota):]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X)[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _compute_en_importance(self, X, y):
        classes = np.unique(y)
        coef = np.zeros(X.shape[1])

        for c in classes:
            y_bin = (y == c).astype(float)
            model = ElasticNetCV(l1_ratio=self.l1_ratio, cv=self.n_folds,
                                 random_state=self.random_state)
            model.fit(X, y_bin)
            coef += np.abs(model.coef_)

        return coef / len(classes)

    @staticmethod
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-12)


# =============================================================================
# Section 4: External Interface (IMPORTANT)
# =============================================================================

class DatasetInterface:
    """
    Users should implement this interface to connect their own datasets.

    Required:
        load() -> signals: np.ndarray (N, L), labels: np.ndarray (N,)
    """

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# =============================================================================
# Example Usage (No Demo Code)
# =============================================================================
# dataset = YourDataset()
# signals, labels = dataset.load()
# X = extract_features_batch(signals)
# screener = ElasticNetScreener()
# X_selected = screener.fit_transform(X, labels)
