"""
OF-ACF Feature Extraction Module
=================================
Optimal-Feature Driven Adversarial Adaptive Cascade Forest
Part 1: 52-dim HOC Domain Feature Extraction + Elastic Net Screening

Features:
  - Group A: 9 HOC (Higher-Order Cumulants): C20, C21, C40, C41, C42, C60, C61, C62, C63
  - Group B: 20 Normalized Proportional Combinations (Cpq / Crs)
  - Group C: 23 Quadratic Proportional Combinations (Cpq / Crs^2)
  - Total: 52 features

Pipeline: 52 features -> Elastic Net + Domain Prior -> 35 retained features
          (Group quota: HOC=9, Normalized=10, Quadratic=16)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Section 1: Higher-Order Cumulant Computation
# =============================================================================

def compute_moments(signal: np.ndarray) -> Dict[str, complex]:
    """
    Compute mixed moments of the signal:
    M_{pq} = E[x^{p-q} * (x*)^q]
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
    Compute higher-order cumulants (HOC) based on mixed moments,
    expanded up to the 6th order using the Leonov-Shiryaev formula.
    """
    M = compute_moments(signal)

    C = {}
    # --- 2nd order ---
    C['C20'] = M['M20']
    C['C21'] = M['M21']

    # --- 4th order ---
    C['C40'] = M['M40'] - 3.0 * M['M20'] ** 2
    C['C41'] = M['M41'] - 3.0 * M['M21'] * M['M20']
    C['C42'] = M['M42'] - np.abs(M['M20']) ** 2 - 2.0 * M['M21'] ** 2

    # --- 6th order (Leonov-Shiryaev) ---
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


# =============================================================================
# Section 2: 52-Dimensional Feature Vector Construction
# =============================================================================

# Group A: 9 HOC features
HOC_FEATURES = ['C20', 'C21', 'C40', 'C41', 'C42', 'C60', 'C61', 'C62', 'C63']

# Group B: 20 Normalized Proportional Combinations (Cpq / Crs)
NORMALIZED_FEATURES = [
    ('C21', 'C20'), ('C40', 'C20'), ('C40', 'C21'), ('C41', 'C40'),
    ('C42', 'C20'), ('C42', 'C21'), ('C60', 'C20'), ('C60', 'C21'),
    ('C60', 'C40'), ('C60', 'C42'), ('C61', 'C20'), ('C61', 'C21'),
    ('C61', 'C60'), ('C62', 'C21'), ('C62', 'C41'), ('C62', 'C42'),
    ('C62', 'C60'), ('C62', 'C40'), ('C63', 'C41'), ('C63', 'C42'),
]

# Group C: 23 Quadratic Proportional Combinations (Cpq / Crs^2)
QUADRATIC_FEATURES = [
    ('C40', 'C20'), ('C41', 'C20'), ('C41', 'C40'), ('C42', 'C20'),
    ('C42', 'C40'), ('C42', 'C41'), ('C60', 'C20'), ('C60', 'C40'),
    ('C60', 'C41'), ('C61', 'C20'), ('C61', 'C21'), ('C61', 'C40'),
    ('C61', 'C41'), ('C61', 'C42'), ('C62', 'C20'), ('C62', 'C21'),
    ('C62', 'C40'), ('C62', 'C42'), ('C62', 'C60'), ('C62', 'C61'),
    ('C63', 'C20'), ('C63', 'C21'), ('C63', 'C61'),
]


def build_feature_names() -> List[str]:
    """Construct the 52-dimensional feature name list."""
    names = []
    for c in HOC_FEATURES:
        names.append(f"|{c}|")
    for (num, den) in NORMALIZED_FEATURES:
        names.append(f"{num}/{den}")
    for (num, den) in QUADRATIC_FEATURES:
        names.append(f"{num}/{den}^2")
    return names


def extract_52_features(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Extract a 52-dimensional HOC-domain feature vector from a single complex signal segment."""
    C = compute_cumulants(signal)
    C_abs = {k: np.abs(v) for k, v in C.items()}

    features = []

    # Group A: 9 HOC
    for name in HOC_FEATURES:
        features.append(C_abs[name])

    # Group B: 20 Normalized (Cpq / Crs)
    for (num, den) in NORMALIZED_FEATURES:
        features.append(C_abs[num] / (C_abs[den] + eps))

    # Group C: 23 Quadratic (Cpq / Crs^2)
    for (num, den) in QUADRATIC_FEATURES:
        features.append(C_abs[num] / (C_abs[den] ** 2 + eps))

    return np.array(features, dtype=np.float64)


def extract_features_batch(signals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Batch extraction of 52-dimensional features."""
    n_samples = signals.shape[0]
    feature_matrix = np.zeros((n_samples, 52), dtype=np.float64)
    for i in range(n_samples):
        feature_matrix[i] = extract_52_features(signals[i], eps=eps)
    return feature_matrix


# =============================================================================
# Section 3: Elastic Net Feature Screening (52 -> 35)
#             Hybrid Filter-Wrapper with Domain Knowledge Prior
# =============================================================================

class ElasticNetScreener:
    """
    Elastic Net Feature Preselector (Hybrid-Criterion Version)

    Adopts a Hybrid Filter-Wrapper framework, integrating both data-driven
    and domain-knowledge-based scoring:

      composite = w_en * EN_importance_normalized + w_prior * Prior_normalized

    Data-driven component:
      Absolute values of Elastic Net OvR coefficients --
      leverage L1 sparsity to identify redundant features irrelevant to classification

    Domain knowledge prior (based on cumulant theory):
      Normalized ratio scoring =
          disc(num) x stab(den) + pairing adjustment

      Quadratic ratio scoring =
          disc(num) x stab_sq(den) + cross-order gain
          + same-order penalty + odd-q penalty + q-matching redundancy penalty

    Core theoretical basis:
    +------------------------------------------------------------------+
    |  Numerator Discriminability                                      |
    |  - C40, C42: kurtosis-related -> core indicators for PSK/QAM    |
    |  - C62, C63: high-order fine discriminators, highly sensitive    |
    |    to constellation geometry                                     |
    |  - C41, C61: odd-order (odd q) terms asymptotically vanish for  |
    |    symmetric modulations (PSK, QAM), low discriminability        |
    |                                                                  |
    |  Denominator Stability                                           |
    |  - C21 = E[|x|^2]: strictly positive (signal power), most       |
    |    stable normalization reference                                 |
    |  - C20 = E[x^2]: may approach zero under circular symmetry,     |
    |    less stable                                                    |
    |  - C41, C61: approach zero under symmetric modulation -> highly  |
    |    unstable as denominators; worse after squaring                 |
    |  - C42^2: variance is amplified after squaring, resulting in     |
    |    extremely low reliability as a quadratic denominator           |
    |                                                                  |
    |  Cross-Order Information Gain                                    |
    |  - Cross-order ratios (e.g., 6th/2nd) provide more complementary|
    |    information than same-order ratios (e.g., 4th/4th)            |
    |  - Severe redundancy in 4th-order same-order ratios (covered     |
    |    by HOC features) -> strong penalty                            |
    |  - 6th-order same-order ratios retain partial information due    |
    |    to q differences -> mild penalty                              |
    |                                                                  |
    |  Even-q Matching Redundancy                                      |
    |  - When numerator and denominator share the same even q          |
    |    (e.g., C62/C42^2, both q=2), quadratic ratios become highly  |
    |    linearly correlated with normalized ratios -> redundancy       |
    |    penalty                                                       |
    +------------------------------------------------------------------+

    Group quota:
      HOC = 9 (fully retained),
      Normalized = 10/20,
      Quadratic = 16/23
    """

    N_HOC = len(HOC_FEATURES)                    # 9
    N_NORM = len(NORMALIZED_FEATURES)             # 20
    N_QUAD = len(QUADRATIC_FEATURES)              # 23
    GROUP_RANGES = {
        'A_HOC':        (0, 9),
        'B_Normalized': (9, 29),
        'C_Quadratic':  (29, 52),
    }

    # ------------------------------------------------------------------
    # Cumulant attribute parameter table
    # References:
    #   Swami & Sadler, IEEE Trans. SP, 2000
    #   Nandi & Azzouz, IEEE Trans. Commun, 1998
    # ------------------------------------------------------------------

    # Discriminability of each cumulant when used as numerator in ratios
    _NUM_DISC = {
        'C20': 0.40, 'C21': 0.55,
        'C40': 0.88, 'C41': 0.22, 'C42': 0.78,
        'C60': 0.48, 'C61': 0.25, 'C62': 0.95, 'C63': 0.82,
    }

    # Estimation stability of each cumulant as denominator in normalized ratios (Cpq/Crs)
    _DEN_STAB = {
        'C20': 0.52, 'C21': 0.95, 'C40': 0.85, 'C41': 0.15,
        'C42': 0.60, 'C60': 0.50,
    }

    # Estimation stability of each cumulant as denominator in quadratic ratios (Cpq/Crs^2)
    # Squaring operation amplifies low-order estimation errors:
    # variance of C42^2 is significantly higher than that of C42
    _DEN_STAB_SQ = {
        'C20': 0.88, 'C21': 0.85, 'C40': 0.52, 'C41': 0.28,
        'C42': 0.05, 'C60': 0.48, 'C61': 0.12,
    }

    # Adjustment terms for specific normalized ratio pairings
    # Physical rationale:
    # High-order cumulants (e.g., C60, C61) exhibit significantly increased
    # estimation variance under low SNR conditions
    _PAIR_ADJ_NORM = {
        ('C21', 'C20'): +0.20,  # Power ratio -- universal normalization baseline across modulations
        ('C41', 'C40'): -0.10,  # Same-order (4th/4th) redundancy + odd-q numerator,
                                 # highly overlapping with |C41| feature
        ('C42', 'C20'): -0.10,  # Redundancy: C42/C21 is more reliable (C21 strictly positive)
        ('C60', 'C20'): -0.18,  # 6th/2nd: large order gap -> unstable variance under finite samples
        ('C60', 'C21'): -0.25,  # Same as above; despite stable C21,
                                 # 6th-order variance O(N^-3) dominates
        ('C60', 'C40'): +0.10,  # 6th/4th: mid-order normalization suppresses variance effectively
        ('C61', 'C20'): -0.18,  # Low discriminability of C61 + high variance (6th/2nd)
        ('C61', 'C21'): -0.18,
    }

    def __init__(self, l1_ratio: float = 0.5,
                 alpha: Optional[float] = None, n_folds: int = 5,
                 group_quota: Tuple[int, int, int] = (9, 10, 16),
                 w_en: float = 0.10, w_prior: float = 0.90,
                 random_state: int = 42):
        """
        Parameters
        ----------
        group_quota : (int, int, int)
            Retention quota for three groups (HOC, Normalized, Quadratic)
        w_en : float
            Weight for the Elastic Net data-driven component
        w_prior : float
            Weight for the domain knowledge prior component
        """
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.n_folds = n_folds
        self.group_quota = group_quota
        self.n_select = sum(group_quota)
        self.w_en = w_en
        self.w_prior = w_prior
        self.random_state = random_state

        self.scaler_ = StandardScaler()
        self.selected_indices_ = None
        self.feature_importances_ = None
        self.composite_scores_ = None
        self.prior_scores_ = None
        self.feature_names_ = build_feature_names()

    # ------------------------------------------------------------------
    # Domain Knowledge Prior
    # ------------------------------------------------------------------
    def _compute_domain_prior(self) -> np.ndarray:
        """
        Compute domain knowledge prior scores based on cumulant theory.

        Normalized ratio (Cpq/Crs):
            score = disc(num) x stab(den) + pair_adjustment

        Quadratic ratio (Cpq/Crs^2):
            score = disc(num) x stab_sq(den)
                    + cross-order information gain
                    + same-order redundancy penalty
                    + odd-q denominator instability penalty
                    + even-q matching redundancy penalty
        """
        scores = np.zeros(52)

        # --- Group A: HOC fully retained, prior = 1.0 ---
        scores[:self.N_HOC] = 1.0

        # --- Group B: Normalized ---
        for i, (num, den) in enumerate(NORMALIZED_FEATURES):
            base = self._NUM_DISC[num] * self._DEN_STAB[den]
            adj = self._PAIR_ADJ_NORM.get((num, den), 0.0)
            scores[self.N_HOC + i] = base + adj

        # --- Group C: Quadratic ---
        for i, (num, den) in enumerate(QUADRATIC_FEATURES):
            p_n, q_n = int(num[1]), int(num[2])
            p_d, q_d = int(den[1]), int(den[2])

            base = self._NUM_DISC[num] * self._DEN_STAB_SQ[den]

            # (a) Cross-order information gain: larger order gap -> more complementary info
            gap_bonus = 0.05 * abs(p_n - p_d) / 2.0

            # (b) Same-order redundancy penalty
            #     4th/4th: information fully covered by HOC direct features -> strong penalty
            #     6th/6th: q-value differences still provide incremental info -> mild penalty
            same_order_pen = 0.0
            if p_n == p_d:
                same_order_pen = -0.45 if p_n == 4 else -0.15

            # (c) Odd-q denominator instability: C41(q=1), C61(q=1) approach zero
            #     for symmetric modulations
            odd_q_pen = -0.08 if (q_d % 2 == 1 and q_d > 0) else 0.0

            # (d) Even-q matching redundancy: when numerator and denominator share
            #     the same even q (e.g., C62/C42^2, both q=2), the quadratic ratio
            #     is highly linearly correlated with the normalized ratio -> redundant
            even_q_pen = 0.0
            if q_n == q_d and q_n > 0 and q_n % 2 == 0:
                even_q_pen = -0.05

            scores[self.N_HOC + self.N_NORM + i] = (
                base + gap_bonus + same_order_pen + odd_q_pen + even_q_pen
            )

        return scores

    # ------------------------------------------------------------------
    # Elastic Net Importance (Data-Driven)
    # ------------------------------------------------------------------
    def _compute_en_importance(self, X_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OvR Elastic Net training -> 52-dim importance vector"""
        classes = np.unique(y)
        n_features = X_scaled.shape[1]
        coef_importance = np.zeros(n_features)

        for cls in classes:
            y_binary = (y == cls).astype(np.float64)
            if self.alpha is None:
                model = ElasticNetCV(
                    l1_ratio=self.l1_ratio, cv=self.n_folds,
                    random_state=self.random_state, max_iter=10000, n_alphas=100
                )
            else:
                model = ElasticNet(
                    alpha=self.alpha, l1_ratio=self.l1_ratio,
                    max_iter=10000, random_state=self.random_state
                )
            model.fit(X_scaled, y_binary)
            coef_importance += np.abs(model.coef_)

        return coef_importance / len(classes)

    # ------------------------------------------------------------------
    # Composite Score & Group-Quota Selection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_scores(arr: np.ndarray) -> np.ndarray:
        """Min-Max normalization to [0, 1]"""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-15:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _group_quota_select(self, composite: np.ndarray) -> np.ndarray:
        """Group-quota selection: retain top-quota features within each group by descending composite score"""
        groups = {
            'A_HOC':        np.arange(0, self.N_HOC),
            'B_Normalized': np.arange(self.N_HOC, self.N_HOC + self.N_NORM),
            'C_Quadratic':  np.arange(self.N_HOC + self.N_NORM,
                                      self.N_HOC + self.N_NORM + self.N_QUAD),
        }
        quotas = dict(zip(groups.keys(), self.group_quota))
        selected = []
        for gname, g_indices in groups.items():
            g_scores = composite[g_indices]
            ranked = np.argsort(g_scores)[::-1]
            for k in range(quotas[gname]):
                selected.append(g_indices[ranked[k]])
        return np.sort(selected)

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetScreener':
        """
        Train Elastic Net + compute prior -> composite score -> group-quota selection
        """
        n_features = X.shape[1]
        assert n_features == 52, f"Expected 52 features, got {n_features}"

        X_scaled = self.scaler_.fit_transform(X)

        # Step 1: Elastic Net importance (data-driven)
        en_imp = self._compute_en_importance(X_scaled, y)
        self.feature_importances_ = en_imp

        # Step 2: Domain prior (theory-driven)
        prior = self._compute_domain_prior()
        self.prior_scores_ = prior

        # Step 3: Per-group normalization -> weighted fusion
        composite = np.zeros(52)
        for gname, (start, end) in self.GROUP_RANGES.items():
            en_norm = self._normalize_scores(en_imp[start:end])
            pr_norm = self._normalize_scores(prior[start:end])
            composite[start:end] = self.w_en * en_norm + self.w_prior * pr_norm

        self.composite_scores_ = composite

        # Step 4: Group-quota selection
        self.selected_indices_ = self._group_quota_select(composite)

        # ------ Print report ------
        self._print_report(en_imp, prior, composite)

        return self

    def _print_report(self, en_imp, prior, composite):
        """Print screening report"""
        groups = {
            'A_HOC':        np.arange(0, self.N_HOC),
            'B_Normalized': np.arange(self.N_HOC, self.N_HOC + self.N_NORM),
            'C_Quadratic':  np.arange(self.N_HOC + self.N_NORM,
                                      self.N_HOC + self.N_NORM + self.N_QUAD),
        }
        quotas = dict(zip(groups.keys(), self.group_quota))

        print(f"\n[Hybrid Screening] 52 -> {self.n_select} features")
        print(f"  Criterion: {self.w_en:.0%} Elastic Net + {self.w_prior:.0%} Domain Prior")
        print(f"  {'Group':<18s} {'Total':>6s} {'Quota':>6s} {'Retained':>9s}")
        print(f"  {'-'*42}")
        for gname, g_idx in groups.items():
            n_ret = len(set(g_idx) & set(self.selected_indices_))
            print(f"  {gname:<18s} {len(g_idx):>6d} {quotas[gname]:>6d} {n_ret:>9d}")

        for gname, g_idx in groups.items():
            retained = [i for i in self.selected_indices_ if i in g_idx]
            eliminated = [i for i in g_idx if i not in self.selected_indices_]
            print(f"\n  [{gname}] Retained ({len(retained)}):")
            for idx in retained:
                print(f"    [{idx:2d}] {self.feature_names_[idx]:20s}  "
                      f"composite={composite[idx]:.4f}  "
                      f"(EN={en_imp[idx]:.4f}, prior={prior[idx]:.3f})")
            if eliminated:
                print(f"  [{gname}] Eliminated ({len(eliminated)}):")
                for idx in eliminated:
                    print(f"    [{idx:2d}] {self.feature_names_[idx]:20s}  "
                          f"composite={composite[idx]:.4f}  "
                          f"(EN={en_imp[idx]:.4f}, prior={prior[idx]:.3f})")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply screening: 52 -> 35"""
        assert self.selected_indices_ is not None, "Must call fit() first"
        X_scaled = self.scaler_.transform(X)
        return X_scaled[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_selected_feature_names(self) -> List[str]:
        return [self.feature_names_[i] for i in self.selected_indices_]

    def get_group_distribution(self) -> Dict[str, int]:
        groups = {
            'A_HOC':        np.arange(0, self.N_HOC),
            'B_Normalized': np.arange(self.N_HOC, self.N_HOC + self.N_NORM),
            'C_Quadratic':  np.arange(self.N_HOC + self.N_NORM,
                                      self.N_HOC + self.N_NORM + self.N_QUAD),
        }
        return {
            gname: len(set(g_idx) & set(self.selected_indices_))
            for gname, g_idx in groups.items()
        }


# =============================================================================
# Section 4: Signal Generation Utilities
# =============================================================================

def generate_modulated_signal(mod_type: str, n_samples: int = 1024,
                               snr_db: float = 10.0,
                               random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate complex baseband signals of common modulation types (for testing).
    Supported: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, AM-DSB, AM-SSB
    """
    rng = np.random.RandomState(random_state)

    if mod_type == 'BPSK':
        signal = rng.choice([-1.0, 1.0], size=n_samples).astype(np.complex128)
    elif mod_type == 'QPSK':
        phase = rng.choice([0, 1, 2, 3], size=n_samples) * np.pi / 2 + np.pi / 4
        signal = np.exp(1j * phase)
    elif mod_type == '8PSK':
        phase = rng.choice(range(8), size=n_samples) * 2 * np.pi / 8
        signal = np.exp(1j * phase)
    elif mod_type == '16QAM':
        c = np.array([-3, -1, 1, 3])
        signal = (rng.choice(c, n_samples) + 1j * rng.choice(c, n_samples)) / np.sqrt(10)
    elif mod_type == '64QAM':
        c = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        signal = (rng.choice(c, n_samples) + 1j * rng.choice(c, n_samples)) / np.sqrt(42)
    elif mod_type == '2FSK':
        freq = rng.choice([-0.1, 0.1], size=n_samples)
        signal = np.exp(1j * 2 * np.pi * np.cumsum(freq))
    elif mod_type == '4FSK':
        freq = rng.choice([-0.3, -0.1, 0.1, 0.3], size=n_samples)
        signal = np.exp(1j * 2 * np.pi * np.cumsum(freq))
    elif mod_type == 'AM-DSB':
        t = np.arange(n_samples)
        msg = rng.randn(n_samples)
        signal = (1 + 0.5 * msg) * np.exp(1j * 2 * np.pi * 0.1 * t)
    elif mod_type == 'AM-SSB':
        from scipy.signal import hilbert
        t = np.arange(n_samples)
        signal = hilbert(rng.randn(n_samples)) * np.exp(1j * 2 * np.pi * 0.1 * t)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

    # Power normalization + AWGN
    signal = signal / np.sqrt(np.mean(np.abs(signal) ** 2))
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (rng.randn(n_samples) + 1j * rng.randn(n_samples))
    return signal + noise


# =============================================================================
# Section 5: Demo Pipeline
# =============================================================================

def demo_pipeline():
    print("=" * 70)
    print("OF-ACF Feature Extraction Demo")
    print("52-dim HOC Feature Extraction + Hybrid Screening (52 -> 35)")
    print("=" * 70)

    mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '2FSK', '4FSK']
    n_per_class = 200
    signal_length = 1024
    snr_db = 10.0

    print(f"\n[1] Generating {len(mod_types)} modulation types x "
          f"{n_per_class} samples @ SNR={snr_db}dB")

    signals, labels = [], []
    for cls_idx, mod in enumerate(mod_types):
        for i in range(n_per_class):
            sig = generate_modulated_signal(
                mod, n_samples=signal_length, snr_db=snr_db,
                random_state=cls_idx * 10000 + i)
            signals.append(sig)
            labels.append(cls_idx)
    signals = np.array(signals)
    labels = np.array(labels)
    print(f"  Signal matrix: {signals.shape}, Labels: {len(mod_types)} classes")

    print(f"\n[2] Extracting 52-dim HOC features ...")
    X = extract_features_batch(signals)
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    print(f"  Feature matrix: {X.shape}")

    print(f"\n[3] Hybrid Screening (Elastic Net + Domain Prior) ...")
    screener = ElasticNetScreener(
        l1_ratio=0.5, n_folds=5,
        group_quota=(9, 10, 16),
        w_en=0.10, w_prior=0.90,
        random_state=42,
    )
    X_selected = screener.fit_transform(X, labels)
    print(f"\n  Selected feature matrix: {X_selected.shape}")

    selected_names = screener.get_selected_feature_names()
    print(f"\n[4] Final 35 retained features:")
    for i, name in enumerate(selected_names):
        print(f"  {i + 1:2d}. {name}")

    print("\n" + "=" * 70)
    print("Feature extraction complete. Ready for ARFE + AACF pipeline.")
    print("=" * 70)

    return X_selected, labels, screener


if __name__ == '__main__':
    X_selected, labels, screener = demo_pipeline()