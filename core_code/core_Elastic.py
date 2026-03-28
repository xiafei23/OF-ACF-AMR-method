"""
OF-ACF Feature Extraction & Elastic Net Screening — Core Module
=================================================================
52-dim HOC Domain Feature Extraction + Hybrid Filter-Wrapper Screening

Core Components:
  - compute_cumulants():    Compute 2nd/4th/6th-order cumulants via Leonov-Shiryaev formulas
  - extract_52_features():  Single signal → 52-dim HOC feature vector
  - extract_features_batch(): Batch feature extraction
  - ElasticNetScreener:     Elastic Net + domain-prior hybrid screening (52 → 35)

External Interface:
  # Feature extraction — input complex baseband signals
  X = extract_features_batch(signals)   # signals: (N, L) complex

  # Feature screening
  screener = ElasticNetScreener(**params)
  screener.fit(X, y)
  X_sel = screener.transform(X)
  names = screener.get_selected_feature_names()
  info  = screener.summary()

52-dimensional Feature Space:
  Group A (9):  |C20|, |C21|, ..., |C63|        (HOC magnitudes)
  Group B (20): Cpq/Crs                          (normalized ratios)
  Group C (23): Cpq/Crs²                         (quadratic ratios, squared denominator)

Dependencies: numpy, scikit-learn
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

__all__ = [
    'compute_moments', 'compute_cumulants',
    'HOC_FEATURES', 'NORMALIZED_FEATURES', 'QUADRATIC_FEATURES',
    'N_HOC', 'N_NORM', 'N_QUAD', 'N_TOTAL',
    'build_feature_names', 'extract_52_features', 'extract_features_batch',
    'ElasticNetScreener',
]


# =====================================================================
# Section 1: Higher-Order Cumulant Computation
# =====================================================================

def compute_moments(signal: np.ndarray) -> Dict[str, complex]:
    """
    Mixed moments: M_{pq} = E[x^{p-q} · (x*)^q]

    Parameters
    ----------
    signal : (N,) complex

    Returns
    -------
    dict  'M20','M21','M40',...,'M63' → complex
    """
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
    """
    Leonov-Shiryaev equation: 2/4/6 order HOCs

    Parameters
    ----------
    signal : (N,) complex

    Returns
    -------
    dict  'C20','C21','C40',...,'C63' → complex
    """
    M = compute_moments(signal)
    C = {}

    # 2-order
    C['C20'] = M['M20']
    C['C21'] = M['M21']

    # 4-order
    C['C40'] = M['M40'] - 3.0 * M['M20'] ** 2
    C['C41'] = M['M41'] - 3.0 * M['M21'] * M['M20']
    C['C42'] = M['M42'] - np.abs(M['M20']) ** 2 - 2.0 * M['M21'] ** 2

    # 6-order
    C['C60'] = (M['M60']
                - 15.0 * M['M40'] * M['M20']
                + 30.0 * M['M20'] ** 3)
    C['C61'] = (M['M61']
                - 5.0 * M['M42'] * M['M20']
                - 10.0 * M['M41'] * M['M21']
                + 30.0 * M['M21'] * M['M20'] ** 2)
    C['C62'] = (M['M62']
                - 6.0 * M['M42'] * M['M20']
                - 8.0 * M['M41'] * M['M21']
                - M['M40'] * np.conj(M['M20'])
                + 6.0 * M['M20'] ** 2 * np.conj(M['M20'])
                + 24.0 * M['M21'] ** 2 * M['M20'])
    C['C63'] = (M['M63']
                - 9.0 * M['M42'] * M['M21']
                + 12.0 * M['M21'] ** 3
                - 3.0 * M['M40'] * np.conj(M['M20'])
                - 3.0 * M['M20'] * np.conj(M['M40'])
                + 18.0 * M['M20'] * np.conj(M['M20']) * M['M21'])

    return C


# =====================================================================
# Section 2: 52-Dimensional Feature Space Definition
# =====================================================================

HOC_FEATURES = ['C20', 'C21', 'C40', 'C41', 'C42',
                'C60', 'C61', 'C62', 'C63']

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

N_HOC   = len(HOC_FEATURES)          # 9
N_NORM  = len(NORMALIZED_FEATURES)   # 20
N_QUAD  = len(QUADRATIC_FEATURES)    # 23
N_TOTAL = N_HOC + N_NORM + N_QUAD    # 52


def build_feature_names() -> List[str]:
    """52-dimensional feature name list"""
    names = []
    for c in HOC_FEATURES:
        names.append(f"|{c}|")
    for (num, den) in NORMALIZED_FEATURES:
        names.append(f"{num}/{den}")
    for (num, den) in QUADRATIC_FEATURES:
        names.append(f"{num}/{den}^2")
    return names


# =====================================================================
# Section 3: Feature Extraction
# =====================================================================

def extract_52_features(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Single complex signal → 52-dimensional HOC features

    Parameters
    ----------
    signal : (N,) complex
    eps : float  Prevent division by zero

    Returns
    -------
    (52,) float64
    """
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


def extract_features_batch(signals: np.ndarray,
                           eps: float = 1e-12) -> np.ndarray:
    """
    Batch extraction of 52-dimensional features

    Parameters
    ----------
    signals : (n_samples, signal_length) complex
    eps : float

    Returns
    -------
    (n_samples, 52) float64
    """
    n_samples = signals.shape[0]
    X = np.zeros((n_samples, N_TOTAL), dtype=np.float64)
    for i in range(n_samples):
        X[i] = extract_52_features(signals[i], eps=eps)
    return X


# =====================================================================
# Section 4: Elastic Net Hybrid Screening (52 → 35)
# =====================================================================

class ElasticNetScreener:
    """
    Hybrid Feature Selection: Elastic Net with Domain Prior (Cumulant-Theoretic)

    This module implements a hybrid feature selection strategy that integrates
    data-driven importance (via Elastic Net) with domain-specific prior knowledge
    derived from cumulant theory.

    The composite score is defined as:
        composite = w_en * EN_norm + w_prior * Prior_norm

    where EN_norm and Prior_norm are independently normalized within each feature group.
    Features are ranked in descending order of the composite score, and top-k features
    are retained according to predefined group-wise quotas.

    Parameters
    ----------
    l1_ratio : float
        The mixing parameter between L1 and L2 regularization in Elastic Net.

    alpha : float or None
        Regularization strength. If None, it is automatically determined via cross-validation.

    n_folds : int
        Number of folds for cross-validation.

    group_quota : tuple of int
        Number of features to retain in each group, specified as (HOC, Normalized, Quadratic).

    w_en : float
        Weight assigned to the data-driven Elastic Net importance.

    w_prior : float
        Weight assigned to the domain prior importance.

    random_state : int
        Random seed for reproducibility.

    verbose : bool
        Whether to print training logs.

    Attributes (after fitting)
    --------------------------
    selected_indices_ : ndarray
        Indices of the selected features, sorted in ascending order.

    feature_importances_ : ndarray of shape (52,)
        Feature importance scores derived from Elastic Net coefficients.

    composite_scores_ : ndarray of shape (52,)
        Final composite scores used for ranking.

    prior_scores_ : ndarray of shape (52,)
        Domain prior scores based on cumulant-theoretic analysis.

    feature_names_ : list of str
        Names of the 52-dimensional feature set.
    """

    GROUP_RANGES = {
        'A_HOC':        (0, N_HOC),
        'B_Normalized': (N_HOC, N_HOC + N_NORM),
        'C_Quadratic':  (N_HOC + N_NORM, N_TOTAL),
    }

    # ---- Cumulant-theory-based parameter tables (Swami & Sadler 2000; Nandi & Azzouz 1998) ----
    _NUM_DISC = {
        'C20': 0.40, 'C21': 0.55,
        'C40': 0.88, 'C41': 0.22, 'C42': 0.78,
        'C60': 0.48, 'C61': 0.25, 'C62': 0.95, 'C63': 0.82,
    }
    _DEN_STAB = {
        'C20': 0.52, 'C21': 0.95, 'C40': 0.85, 'C41': 0.15,
        'C42': 0.60, 'C60': 0.50,
    }
    _DEN_STAB_SQ = {
        'C20': 0.88, 'C21': 0.85, 'C40': 0.52, 'C41': 0.28,
        'C42': 0.05, 'C60': 0.48, 'C61': 0.12,
    }
    _PAIR_ADJ_NORM = {
        ('C21', 'C20'): +0.20, ('C41', 'C40'): -0.10,
        ('C42', 'C20'): -0.10, ('C60', 'C20'): -0.18,
        ('C60', 'C21'): -0.25, ('C60', 'C40'): +0.10,
        ('C61', 'C20'): -0.18, ('C61', 'C21'): -0.18,
    }

    def __init__(self, l1_ratio: float = 0.5,
                 alpha: Optional[float] = None, n_folds: int = 5,
                 group_quota: Tuple[int, int, int] = (9, 10, 16),
                 w_en: float = 0.10, w_prior: float = 0.90,
                 random_state: int = 42, verbose: bool = True):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.n_folds = n_folds
        self.group_quota = group_quota
        self.n_select = sum(group_quota)
        self.w_en = w_en
        self.w_prior = w_prior
        self.random_state = random_state
        self.verbose = verbose

        self.scaler_ = StandardScaler()
        self.selected_indices_ = None
        self.feature_importances_ = None
        self.composite_scores_ = None
        self.prior_scores_ = None
        self.feature_names_ = build_feature_names()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Domain Prior
    # ------------------------------------------------------------------
    def _compute_domain_prior(self) -> np.ndarray:
        """Domain prior scoring (52,)"""
        scores = np.zeros(N_TOTAL)
        scores[:N_HOC] = 1.0  # HOC 全保留

        for i, (num, den) in enumerate(NORMALIZED_FEATURES):
            base = self._NUM_DISC[num] * self._DEN_STAB[den]
            base += self._PAIR_ADJ_NORM.get((num, den), 0.0)
            scores[N_HOC + i] = base

        for i, (num, den) in enumerate(QUADRATIC_FEATURES):
            p_n, q_n = int(num[1]), int(num[2])
            p_d, q_d = int(den[1]), int(den[2])
            base = self._NUM_DISC[num] * self._DEN_STAB_SQ[den]
            base += 0.05 * abs(p_n - p_d) / 2.0  # Cross-order gain
            if p_n == p_d:
                base += (-0.45 if p_n == 4 else -0.15)  # Same-order penalty
            if q_d % 2 == 1 and q_d > 0:
                base -= 0.08  # Odd-q penalty
            if q_n == q_d and q_n > 0 and q_n % 2 == 0:
                base -= 0.05  # Even-q redundancy penalty
            scores[N_HOC + N_NORM + i] = base

        return scores

    # ------------------------------------------------------------------
    # Elastic Net (Data-Driven)
    # ------------------------------------------------------------------
    def _compute_en_importance(self, X_scaled, y) -> np.ndarray:
        """Elastic Net importance (OvR → 52-dim importance)"""
        classes = np.unique(y)
        coef_imp = np.zeros(X_scaled.shape[1])
        for cls in classes:
            y_bin = (y == cls).astype(np.float64)
            if self.alpha is None:
                m = ElasticNetCV(l1_ratio=self.l1_ratio, cv=self.n_folds,
                                 random_state=self.random_state,
                                 max_iter=10000, n_alphas=100)
            else:
                m = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                               max_iter=10000, random_state=self.random_state)
            m.fit(X_scaled, y_bin)
            coef_imp += np.abs(m.coef_)
        return coef_imp / len(classes)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_scores(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-15:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _group_quota_select(self, composite) -> np.ndarray:
        """Top-quota selection within each group"""
        groups = {
            'A_HOC':        np.arange(0, N_HOC),
            'B_Normalized': np.arange(N_HOC, N_HOC + N_NORM),
            'C_Quadratic':  np.arange(N_HOC + N_NORM, N_TOTAL),
        }
        quotas = dict(zip(groups.keys(), self.group_quota))
        selected = []
        for gname, g_idx in groups.items():
            ranked = np.argsort(composite[g_idx])[::-1]
            for k in range(quotas[gname]):
                selected.append(g_idx[ranked[k]])
        return np.sort(selected)

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetScreener':
        """
        Fit the feature screener using labeled data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 52)
            Input feature matrix consisting of 52-dimensional HOC features.

        y : np.ndarray of shape (n_samples,)
            Class labels encoded as integers.

        Returns
        -------
        self : ElasticNetScreener
            Fitted screener instance.
        """
        assert X.shape[1] == N_TOTAL, \
            f"Expected {N_TOTAL} features, got {X.shape[1]}"

        X_scaled = self.scaler_.fit_transform(X)

        en_imp = self._compute_en_importance(X_scaled, y)
        self.feature_importances_ = en_imp

        prior = self._compute_domain_prior()
        self.prior_scores_ = prior

        composite = np.zeros(N_TOTAL)
        for gname, (start, end) in self.GROUP_RANGES.items():
            en_n = self._normalize_scores(en_imp[start:end])
            pr_n = self._normalize_scores(prior[start:end])
            composite[start:end] = self.w_en * en_n + self.w_prior * pr_n
        self.composite_scores_ = composite

        self.selected_indices_ = self._group_quota_select(composite)

        self._log(f"\nElastic Net: {N_TOTAL} -> {len(self.selected_indices_)}  "
                  f"({self.w_en:.0%} EN + {self.w_prior:.0%} Prior)")
        grp = self.get_group_distribution()
        for g, cnt in grp.items():
            self._log(f"  {g}: {cnt}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection and standardization to input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 52)
            Input feature matrix.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_selected_features)
            Standardized and feature-selected matrix.
        """
        assert self.selected_indices_ is not None, "Call fit() first"
        return self.scaler_.transform(X)[:, self.selected_indices_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_selected_feature_names(self) -> List[str]:
        """Retained feature names"""
        return [self.feature_names_[i] for i in self.selected_indices_]

    def get_group_distribution(self) -> Dict[str, int]:
        """Group-wise distribution of selected features"""
        sel = set(self.selected_indices_.tolist())
        return {
            gname: len(sel & set(range(start, end)))
            for gname, (start, end) in self.GROUP_RANGES.items()
        }

    def summary(self) -> dict:
        """Structured summary (JSON-serializable)"""
        return {
            'n_input': N_TOTAL,
            'n_output': len(self.selected_indices_),
            'selected_names': self.get_selected_feature_names(),
            'selected_indices': self.selected_indices_.tolist(),
            'group_distribution': self.get_group_distribution(),
            'params': {
                'l1_ratio': self.l1_ratio, 'n_folds': self.n_folds,
                'group_quota': self.group_quota,
                'w_en': self.w_en, 'w_prior': self.w_prior,
            },
        }

    def print_summary(self):
        """Print concise summary"""
        if self.selected_indices_ is None:
            print("Not fitted."); return
        sel = self.get_selected_feature_names()
        grp = self.get_group_distribution()
        print(f"\nElastic Net: {N_TOTAL} -> {len(sel)}")
        for g, cnt in grp.items():
            print(f"  {g}: {cnt}")
        print(f"Retained: {sel}")