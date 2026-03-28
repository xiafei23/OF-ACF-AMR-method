"""
OF-ACF ARFE Module
==================
Augmented Recursive Feature Elimination with Feature-level Multi-Head Attention

Pipeline: 35 features (Elastic Net screening) → T rounds FMHA + Soft-SVM → 27 optimal features

Core Components:
  1. FMHA (Feature-level Multi-Head Attention):
     Does not explicitly construct Q, K, V matrices. Instead, directly utilizes the self-similarity matrix
     of feature vectors to implicitly implement the attention mechanism,
     learning a set of attention weights in the feature dimension and focusing on high-order cumulant
     features with strong discriminative power.

  2. Soft-SVM Evaluator:
     Linear-kernel soft-margin SVM, which quantifies the contribution of each feature to the classification
     boundary via the hyperplane normal vector |w_j|.

  3. Multi-Criteria Importance Scoring:
     I(j) = w_α · ā_j  +  w_s · |w_j^svm|  +  w_F · F_j
           + w_ρ · (1 - ρ_j)  +  w_σ · σ_j

     ā_j  : FMHA attention weight — feature-level attention focus
     |w_j| : Soft-SVM weight — contribution to the classification boundary
     F_j   : Fisher discriminant ratio — between-class variance / within-class variance
     ρ_j   : maximum mutual correlation — redundancy with other features
     σ_j   : cumulant-theoretic stability prior — numerator discriminability × denominator estimation stability

References:
  - Swami & Sadler, "Hierarchical digital modulation classification using
    cumulants," IEEE Trans. Commun., 2000.
  - Nandi & Azzouz, "Algorithms for automatic modulation recognition of
    communication signals," IEEE Trans. Commun., 1998.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Section 1: Feature Name Parser
# =====================================================================

def parse_feature(name: str) -> dict:
    """
    Parse feature name into structured information.

    Examples:
        '|C20|'       → {'type': 'hoc', 'cumulant': 'C20', 'order': 2, 'q': 0}
        'C62/C21'     → {'type': 'normalized', 'num': 'C62', 'den': 'C21', ...}
        'C62/C40^2'   → {'type': 'quadratic', 'num': 'C62', 'den': 'C40', ...}
    """
    name = name.strip()
    if name.startswith('|') and name.endswith('|'):
        cum = name.strip('|')
        return {'type': 'hoc', 'cumulant': cum,
                'order': int(cum[1]), 'q': int(cum[2])}
    elif '^2' in name:
        parts = name.replace('^2', '').split('/')
        num, den = parts[0].strip(), parts[1].strip()
        return {'type': 'quadratic', 'num': num, 'den': den,
                'num_order': int(num[1]), 'num_q': int(num[2]),
                'den_order': int(den[1]), 'den_q': int(den[2])}
    elif '/' in name:
        num, den = name.split('/')
        num, den = num.strip(), den.strip()
        return {'type': 'normalized', 'num': num, 'den': den,
                'num_order': int(num[1]), 'num_q': int(num[2]),
                'den_order': int(den[1]), 'den_q': int(den[2])}
    else:
        return {'type': 'unknown', 'name': name}


# =====================================================================
# Section 2: Feature-level Multi-Head Attention (FMHA)
# =====================================================================

class FMHA:
    """
    Feature-level Multi-Head Attention (FMHA)

    Does not explicitly construct Q, K, V matrices. Instead, directly performs
    self-similarity analysis on the derived feature matrix to implicitly build
    the attention mechanism, learning attention weight vectors alpha_h^(t) in
    the feature dimension.

    Algorithm steps (round t, p_t active features, H attention heads):
    ────────────────────────────────────────────────────────────────────
    (1) Randomly assign p_t features to H heads, each with dimension d_k = p_t / H
    (2) For each sample n, each head h:
        · Extract sub-vector X_n^(h) ∈ R^{d_k}
        · Compute feature self-similarity matrix:
              S_{n,h} = softmax( X_n^(h) · X_n^(h)^T / sqrt(d_k) )
        · Row aggregation: s_j = sum_l S_{n,h}[j, l] → total affinity of feature j
        · Attention weights: alpha_{h,j} = softmax(s)[j]
    (3) Sample average: alpha_bar_h = (1/N) sum_n alpha_{n,h}
    (4) Hadamard weighting: Z_{n,h} = alpha_bar_h ⊙ X_n^(h)
    (5) Multi-head concatenation: Z_n = [ Z_{n,1}, Z_{n,2}, ..., Z_{n,H} ] ∈ R^{p_t}

    Parameters
    ----------
    n_heads : int
        Number of attention heads H (default 5)
    random_state : int
        Controls random assignment of features to heads for reproducibility
    """

    def __init__(self, n_heads: int = 5, random_state: int = 42):
        self.n_heads = n_heads
        self.random_state = random_state
        self.head_assignment_ = None
        self.attention_weights_ = None

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax"""
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

    def _assign_heads(self, p_t: int) -> List[np.ndarray]:
        """
        Randomly assign p_t features to H attention heads.

        If p_t is not divisible by H, the first (p_t % H) heads each receive
        one extra feature.
        """
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(p_t)
        d_k = p_t // self.n_heads
        remainder = p_t % self.n_heads

        heads = []
        start = 0
        for h in range(self.n_heads):
            size = d_k + (1 if h < remainder else 0)
            heads.append(np.sort(perm[start:start + size]))
            start += size
        return heads

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        FMHA forward pass.

        Parameters
        ----------
        X : ndarray, shape (n_samples, p_t)
            Standardized feature matrix for the current round

        Returns
        -------
        Z : ndarray, shape (n_samples, p_t)
            Attention-weighted aggregated feature matrix
        alpha_full : ndarray, shape (p_t,)
            Mean attention weight per feature (mapped back to full p_t space)
        """
        n_samples, p_t = X.shape
        heads = self._assign_heads(p_t)
        self.head_assignment_ = heads

        Z = np.zeros_like(X)
        alpha_full = np.zeros(p_t)

        for h, h_idx in enumerate(heads):
            d_k = len(h_idx)
            X_h = X[:, h_idx]                        # (N, d_k)

            # ---- Per-sample attention computation, then average ----
            alpha_accum = np.zeros(d_k)
            for n in range(n_samples):
                x_n = X_h[n]                          # (d_k,)

                # Feature self-similarity matrix (implicit Q=K=V=feature vector)
                sim = np.outer(x_n, x_n) / (np.sqrt(d_k) + 1e-12)
                S_nh = self._softmax(sim, axis=1)     # (d_k, d_k)

                # Row aggregation → total affinity of each feature
                row_agg = S_nh.sum(axis=1)            # (d_k,)

                # Softmax → attention weights
                alpha_n = self._softmax(row_agg)      # (d_k,)
                alpha_accum += alpha_n

            alpha_h = alpha_accum / n_samples         # Sample-averaged attention

            # Hadamard weighting
            Z[:, h_idx] = alpha_h[np.newaxis, :] * X_h

            # Map back to full feature space
            alpha_full[h_idx] = alpha_h

        self.attention_weights_ = alpha_full
        return Z, alpha_full


# =====================================================================
# Section 3: Multi-Criteria Importance Scoring
# =====================================================================

class ImportanceScorer:
    """
    Multi-Criteria Feature Importance Scorer

    Fuses five evaluation dimensions:
      (1) FMHA attention weight alpha_bar_j  — feature interaction affinity
      (2) Soft-SVM coefficient |w_j|         — contribution to classification boundary
      (3) Fisher discriminant ratio F_j      — statistical separability
      (4) Redundancy penalty rho_j           — information complementarity
      (5) Cumulant-theoretic prior sigma_j   — estimation stability

    Core prior parameter tables are based on:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Numerator Discriminability                                      │
    │  · After power normalization, C21 = E[|x|^2] ≈ 1 → near-       │
    │    constant, lowest discriminability                              │
    │  · C40, C42: kurtosis/ellipticity, key PSK/QAM discriminators   │
    │  · C62, C63: high-order fine classifiers, highly sensitive to    │
    │    constellation geometry                                        │
    │  · C41, C61: odd-q values, asymptotically zero for symmetric    │
    │    modulations (PSK, QAM)                                        │
    │                                                                  │
    │  Denominator Estimation Stability                                │
    │  · Quadratic-ratio denominators: Var(C_hat^2) ≈ 4*C_hat^2 *    │
    │    Var(C_hat) (delta method) — squaring amplifies estimation     │
    │    error                                                         │
    │  · C41^2: C41→0 for symmetric modulations, squared denominator  │
    │    approaches zero → ratio diverges                               │
    │  · C42^2: 4th-order variance O(N^-2), after squaring O(N^-4)    │
    │  · C60^2: 6th-order variance O(N^-3), after squaring O(N^-6)    │
    │                                                                  │
    │  Redundancy detection: quadratic ratios sharing the same         │
    │  numerator but different denominators are highly correlated       │
    │  · If Cpq/Crs^2 and Cpq/Crt^2 exist with stab(Crt)>>stab(Crs),│
    │    the former is redundant, detected via Pearson correlation      │
    └───────────────────────────────────────────────────────────────────┘
    """

    # ---- Numerator discriminability ----
    # Based on: theoretical discriminative power of each cumulant after
    # power normalization.
    # C21 ≈ 1 (constant) → lowest discriminability
    # C41, C61: odd-q, approach zero for symmetric modulations, but the
    #   "zero/non-zero" binary property provides key discrimination between
    #   symmetric (PSK, QAM) vs asymmetric (FSK, AM) modulations.
    # C61 additional contribution: 6th-order odd-q carries third-order
    #   phase coupling information that 4th-order C41 cannot provide,
    #   hence C61 has higher discriminability than C41.
    _NUM_DISC = {
        'C20': 0.42, 'C21': 0.05,
        'C40': 0.92, 'C41': 0.55, 'C42': 0.85,
        'C60': 0.65, 'C61': 0.72, 'C62': 0.95, 'C63': 0.88,
    }

    # ---- HOC direct-feature adjustment ----
    # After signal power normalization E[|x|^2]=1:
    #   C21 = E[|x|^2] ≡ 1 → degenerates to a constant, carries no
    #   modulation information. SVM/Fisher overestimate its score due
    #   to finite-sample noise fluctuation, not reflecting true discriminability.
    #   A strong correction is needed to counteract inflated data-driven scores.
    _HOC_ADJ = {
        'C21': -0.15,
    }

    # ---- Normalized-ratio denominator stability ----
    _DEN_STAB_NORM = {
        'C20': 0.70, 'C21': 0.95, 'C40': 0.88, 'C41': 0.15,
        'C42': 0.78, 'C60': 0.50,
    }

    # ---- Normalized-ratio pair-specific adjustment ----
    # C62/C60: same-order (6th/6th) ratio, the q-value difference (q=2 vs q=0)
    #   provides limited incremental information, and is highly collinear with
    #   the cross-order ratio C62/C40 → redundant.
    _PAIR_ADJ_NORM = {
        ('C62', 'C60'): -0.30,  # 6th/6th: same-order redundancy + highly collinear
                                 # with C62/C40 (rho > 0.90); C60 estimation variance
                                 # O(N^-3) makes the ratio extremely unstable at low SNR;
                                 # the cross-order ratio C62/C40 already captures the
                                 # same information via a stable 4th-order baseline
    }

    # ---- Quadratic-ratio denominator stability (squaring amplification) ----
    # Var(C_hat^2) ≈ 4*C_hat^2*Var(C_hat) → variance inflates drastically
    # after squaring.
    # C41^2: C41→0 for symmetric modulations, squared denominator approaches
    #        zero → ratio diverges. Note: divergent values artificially inflate
    #        the Fisher ratio but do not reflect true separability.
    _DEN_STAB_SQ = {
        'C20': 0.68, 'C21': 0.90, 'C40': 0.52, 'C41': 0.02,
        'C42': 0.02, 'C60': 0.10, 'C61': 0.05,
    }

    # Same-order redundancy penalty
    _SAME_ORDER_PEN = {4: -0.25, 6: -0.22}

    # ---- Quadratic-ratio pair-specific adjustment ----
    # (C41,C20): C41/C20^2 is linearly redundant with |C41| and |C20|
    # (C63,C20): strictly dominated by C63/C21^2
    # (C60,C41): C41^2 denominator diverges for symmetric modulations,
    #            artificially inflating Fisher; C60/C40^2 already provides
    #            the same cross-order information (C40 is the even-q counterpart of C41)
    _PAIR_ADJ_QUAD = {
        ('C41', 'C20'): -0.18,   # 4th-order odd-q / 2nd-order^2: linear redundancy
        ('C63', 'C20'): -0.30,   # Strictly dominated by C63/C21^2: C20=E[x^2] approaches
                                  # zero for circularly symmetric signals, C20^2→0+ causes
                                  # ratio divergence; C21=E[|x|^2] is always positive
                                  # → C21^2 is far more stable than C20^2
        ('C60', 'C41'): -0.15,   # C41^2 denominator diverges + redundant with C60/C40^2
        ('C61', 'C42'): -0.12,   # C42^2 estimation variance O(N^-4) makes ratio unreliable;
                                  # SVM/Fisher inflation originates from finite-sample
                                  # divergence rather than true separability;
                                  # C61/C40^2 (even-q stable denominator) provides same
                                  # cross-order information
        ('C61', 'C40'): +0.08,   # 6th/4th-order cross-order ratio: mid-order stable
                                  # denominator (even-q, C40^2 variance controllable)
                                  # effectively captures C61's third-order phase coupling
                                  # information; closer to 6th-order signal characteristics
                                  # than C61/C20^2 and C61/C21^2 (denominator itself
                                  # contains 4th-order nonlinear structure)
    }

    def __init__(self, feature_names: List[str],
                 w_attn: float = 0.15, w_svm: float = 0.10,
                 w_fisher: float = 0.10, w_redundancy: float = 0.20,
                 w_stability: float = 0.45):
        self.feature_names = feature_names
        self.w_attn = w_attn
        self.w_svm = w_svm
        self.w_fisher = w_fisher
        self.w_redundancy = w_redundancy
        self.w_stability = w_stability

        # Pre-compute theoretical prior (depends only on feature names, not data)
        self._stability_cache = self._compute_stability_prior(feature_names)

    # ------------------------------------------------------------------
    # Theoretical Stability Prior
    # ------------------------------------------------------------------
    def _compute_stability_prior(self, names: List[str]) -> np.ndarray:
        """
        Compute stability prior scores based on cumulant estimation theory.

        HOC:           score = discriminability(Cpq)
        Normalized:    score = disc(num) * stab(den) + pair_adj
        Quadratic:     score = disc(num) * stab_sq(den) + same_order_pen
        """
        scores = np.zeros(len(names))
        for i, name in enumerate(names):
            info = parse_feature(name)

            if info['type'] == 'hoc':
                # HOC direct feature: discriminability + specific adjustment
                cum = info['cumulant']
                scores[i] = (self._NUM_DISC.get(cum, 0.5)
                             + self._HOC_ADJ.get(cum, 0.0))

            elif info['type'] == 'normalized':
                num, den = info['num'], info['den']
                base = (self._NUM_DISC.get(num, 0.5) *
                        self._DEN_STAB_NORM.get(den, 0.5))

                # Same-order penalty
                if info['num_order'] == info['den_order']:
                    pen = self._SAME_ORDER_PEN.get(info['num_order'], -0.10)
                    base += pen

                # 2nd/2nd (pure power ratio) → additional penalty
                if info['num_order'] == 2 and info['den_order'] == 2:
                    base -= 0.10

                # Pair-specific adjustment
                pair_adj = self._PAIR_ADJ_NORM.get((num, den), 0.0)
                base += pair_adj

                scores[i] = base

            elif info['type'] == 'quadratic':
                num, den = info['num'], info['den']
                base = (self._NUM_DISC.get(num, 0.5) *
                        self._DEN_STAB_SQ.get(den, 0.3))

                # Same-order penalty (more severe for quadratic ratios
                # because squaring amplifies redundancy)
                if info['num_order'] == info['den_order']:
                    pen = self._SAME_ORDER_PEN.get(info['num_order'], -0.10)
                    base += pen * 1.2   # 1.2x amplification for quadratic ratios

                # Odd-q denominator: approaches zero for symmetric modulations,
                # approaches zero even faster after squaring
                if info['den_q'] % 2 == 1 and info['den_q'] > 0:
                    base -= 0.05

                # Pair-specific adjustment (physical constraints of specific
                # numerator-denominator combinations)
                pair_adj = self._PAIR_ADJ_QUAD.get((num, den), 0.0)
                base += pair_adj

                scores[i] = base

        return scores

    # ------------------------------------------------------------------
    # Fisher Discriminant Ratio
    # ------------------------------------------------------------------
    @staticmethod
    def fisher_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fisher discriminant ratio: F_j = var_between / var_within

        A classic filter-type feature scoring metric that measures the
        statistical between-class separability of individual features.
        """
        classes = np.unique(y)
        n_features = X.shape[1]
        global_mean = X.mean(axis=0)

        between_var = np.zeros(n_features)
        within_var = np.zeros(n_features)

        for c in classes:
            mask = (y == c)
            n_c = mask.sum()
            class_mean = X[mask].mean(axis=0)
            between_var += n_c * (class_mean - global_mean) ** 2
            within_var += X[mask].var(axis=0) * n_c

        return between_var / (within_var + 1e-12)

    # ------------------------------------------------------------------
    # Redundancy via Max Correlation
    # ------------------------------------------------------------------
    @staticmethod
    def max_abs_correlation(X: np.ndarray) -> np.ndarray:
        """
        Feature redundancy: max |Pearson correlation| with all other features.

        High correlation → the information carried by this feature can be
        replaced by another feature → redundant.
        """
        n_features = X.shape[1]
        if n_features < 2:
            return np.zeros(n_features)
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0.0)
        return np.max(np.abs(corr), axis=1)

    # ------------------------------------------------------------------
    # Composite Score
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-15:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    def score(self, X: np.ndarray, y: np.ndarray,
              alpha: np.ndarray, svm_coef: np.ndarray,
              active_indices: np.ndarray) -> np.ndarray:
        """
        Compute composite importance scores for currently active features.

        Parameters
        ----------
        X : (n_samples, p_t) current feature matrix
        y : (n_samples,) labels
        alpha : (p_t,) FMHA attention weights
        svm_coef : (p_t,) absolute SVM coefficients
        active_indices : (p_t,) feature indices in the original 35-dim space

        Returns
        -------
        composite : (p_t,) composite importance scores (higher = more important)
        """
        p_t = X.shape[1]

        # (1) FMHA attention
        s_attn = self._normalize_01(alpha)

        # (2) SVM coefficients
        s_svm = self._normalize_01(svm_coef)

        # (3) Fisher discriminant ratio
        fisher = self.fisher_ratio(X, y)
        s_fisher = self._normalize_01(fisher)

        # (4) Redundancy penalty (higher correlation = more redundant → invert)
        rho = self.max_abs_correlation(X)
        s_nonredundancy = self._normalize_01(1.0 - rho)

        # (5) Theoretical stability prior
        stability = self._stability_cache[active_indices]
        s_stability = self._normalize_01(stability)

        # Weighted fusion
        composite = (self.w_attn * s_attn
                     + self.w_svm * s_svm
                     + self.w_fisher * s_fisher
                     + self.w_redundancy * s_nonredundancy
                     + self.w_stability * s_stability)

        return composite


# =====================================================================
# Section 4: ARFE — Augmented Recursive Feature Elimination
# =====================================================================

class ARFE:
    """
    Augmented Recursive Feature Elimination (ARFE)

    Overall pipeline:
      Input: X in R^{N x 35}, y in R^N (from Elastic Net screened features)
      For t = 1, 2, ..., T:
        (1) FMHA attention weighting:   Z^(t) = FMHA(X^(t))
        (2) Soft-SVM training:          w^(t) = SVM.fit(Z^(t), y).coef_
        (3) Multi-criteria scoring:     I_j = composite(alpha, |w|, Fisher, rho, sigma)
        (4) Eliminate weakest feature:  S_{t+1} = S_t \\ {argmin I_j}
      Output: 27-dimensional optimal feature subset

    Parameters
    ----------
    n_heads : int
        Number of FMHA attention heads (default 5)
    svm_C : float
        Soft-SVM regularization parameter (penalty coefficient)
    n_eliminate_per_round : int
        Number of features to eliminate per round (default 1)
    target_n_features : int
        Target number of features (default 27)
    w_attn, w_svm, w_fisher, w_redundancy, w_stability : float
        Weights for each component of the multi-criteria scoring
    random_state : int
        Random seed
    """

    def __init__(self, n_heads: int = 5, svm_C: float = 1.0,
                 n_eliminate_per_round: int = 1,
                 target_n_features: int = 27,
                 w_attn: float = 0.15, w_svm: float = 0.10,
                 w_fisher: float = 0.10, w_redundancy: float = 0.20,
                 w_stability: float = 0.45,
                 random_state: int = 42):
        self.n_heads = n_heads
        self.svm_C = svm_C
        self.n_eliminate = n_eliminate_per_round
        self.target_n = target_n_features
        self.w_attn = w_attn
        self.w_svm = w_svm
        self.w_fisher = w_fisher
        self.w_redundancy = w_redundancy
        self.w_stability = w_stability
        self.random_state = random_state

        # Training results
        self.selected_indices_ = None
        self.elimination_history_ = []
        self.round_details_ = []
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'ARFE':
        """
        Execute ARFE recursive elimination.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 35)
            35-dim feature matrix from Elastic Net screening
        y : ndarray, shape (n_samples,)
            Class labels
        feature_names : list of str
            Names of the 35 features
        """
        self.feature_names_ = list(feature_names)
        n_features_init = X.shape[1]
        T = (n_features_init - self.target_n) // self.n_eliminate

        # Initialize scorer (prior cache depends only on feature names)
        scorer = ImportanceScorer(
            feature_names,
            w_attn=self.w_attn, w_svm=self.w_svm,
            w_fisher=self.w_fisher, w_redundancy=self.w_redundancy,
            w_stability=self.w_stability,
        )

        # Active feature indices (positions in the original 35-dim space)
        active = np.arange(n_features_init)
        scaler = StandardScaler()

        print(f"\n{'='*70}")
        print(f"ARFE: Augmented Recursive Feature Elimination")
        print(f"  Initial features: {n_features_init}")
        print(f"  Target features:  {self.target_n}")
        print(f"  Rounds:           {T} (eliminate {self.n_eliminate}/round)")
        print(f"  FMHA heads:       {self.n_heads}")
        print(f"  Soft-SVM C:       {self.svm_C}")
        print(f"  Scoring weights:  attn={self.w_attn:.2f}, svm={self.w_svm:.2f}, "
              f"fisher={self.w_fisher:.2f}, redund={self.w_redundancy:.2f}, "
              f"stab={self.w_stability:.2f}")
        print(f"{'='*70}")

        for t in range(T):
            p_t = len(active)
            X_t = X[:, active]

            # ---- Standardization ----
            X_scaled = scaler.fit_transform(X_t)

            # ---- Step 1: FMHA ----
            fmha = FMHA(n_heads=self.n_heads,
                        random_state=self.random_state + t)
            Z_t, alpha = fmha.forward(X_scaled)

            # ---- Step 2: Soft-SVM (OvR) ----
            svm = SVC(kernel='linear', C=self.svm_C, decision_function_shape='ovr',
                      random_state=self.random_state)
            svm.fit(Z_t, y)

            # Multi-class: coef_ shape = (n_classes*(n_classes-1)/2, p_t)
            svm_coef = np.mean(np.abs(svm.coef_), axis=0)

            # ---- Step 3: Multi-criteria scoring ----
            composite = scorer.score(X_scaled, y, alpha, svm_coef, active)

            # ---- Step 4: Eliminate weakest ----
            n_elim = min(self.n_eliminate, p_t - self.target_n)
            weakest = np.argsort(composite)[:n_elim]

            elim_names = [self.feature_names_[active[w]] for w in weakest]
            elim_scores = composite[weakest]

            # Record round details
            round_info = {
                'round': t + 1,
                'p_before': p_t,
                'eliminated': elim_names,
                'eliminated_scores': elim_scores.tolist(),
                'all_scores': {self.feature_names_[active[j]]: composite[j]
                               for j in range(p_t)},
            }
            self.round_details_.append(round_info)
            self.elimination_history_.extend(elim_names)

            # Print round results
            print(f"\n  Round {t+1}/{T}: {p_t} -> {p_t - n_elim} features")
            for w in weakest:
                idx_orig = active[w]
                fname = self.feature_names_[idx_orig]
                print(f"    x Eliminated: {fname:20s}  "
                      f"composite={composite[w]:.4f}  "
                      f"(alpha={alpha[w]:.3f}, svm={svm_coef[w]:.4f}, "
                      f"stab={scorer._stability_cache[idx_orig]:.3f})")

            # Update active set
            active = np.delete(active, weakest)

        self.selected_indices_ = active
        print(f"\n{'='*70}")
        print(f"ARFE Complete: {n_features_init} -> {len(active)} features")
        print(f"{'='*70}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection: 35 -> 27"""
        assert self.selected_indices_ is not None, "Must call fit() first"
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str]) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_feature_names(self) -> List[str]:
        return [self.feature_names_[i] for i in self.selected_indices_]

    def get_elimination_order(self) -> List[str]:
        return list(self.elimination_history_)

    def print_summary(self):
        """Print ARFE screening result summary."""
        if self.selected_indices_ is None:
            print("ARFE has not been fitted yet.")
            return

        selected = self.get_selected_feature_names()

        # Group statistics
        hoc = [n for n in selected if n.startswith('|')]
        norm = [n for n in selected if '/' in n and '^2' not in n]
        quad = [n for n in selected if '^2' in n]

        print(f"\n  ARFE Retained Features ({len(selected)} total):")
        print(f"  {'Group':<18s} {'Count':>6s}")
        print(f"  {'-'*26}")
        print(f"  {'HOC':<18s} {len(hoc):>6d}")
        print(f"  {'Normalized':<18s} {len(norm):>6d}")
        print(f"  {'Quadratic':<18s} {len(quad):>6d}")

        print(f"\n  HOC ({len(hoc)}):")
        for n in hoc:
            print(f"    {n}")
        print(f"  Normalized ({len(norm)}):")
        for n in norm:
            print(f"    {n}")
        print(f"  Quadratic ({len(quad)}):")
        for n in quad:
            print(f"    {n}")

        print(f"\n  Elimination order:")
        for i, name in enumerate(self.elimination_history_):
            print(f"    Round {i+1}: {name}")


# =====================================================================
# Section 5: Demo Pipeline
# =====================================================================

def demo_arfe():
    """
    ARFE demo: Import Elastic Net screening results -> FMHA + Soft-SVM -> 27 features
    """
    from Elastic import (
        generate_modulated_signal, extract_features_batch,
        ElasticNetScreener
    )

    print("=" * 70)
    print("OF-ACF ARFE Demo")
    print("35-dim -> FMHA + Soft-SVM Recursive Elimination -> 27-dim")
    print("=" * 70)

    # ---- Step 1: Generate data ----
    mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '2FSK', '4FSK']
    n_per_class = 200
    snr_db = 10.0

    print(f"\n[1] Generating {len(mod_types)} modulations x "
          f"{n_per_class} samples @ SNR={snr_db}dB")

    signals, labels = [], []
    for cls_idx, mod in enumerate(mod_types):
        for i in range(n_per_class):
            sig = generate_modulated_signal(
                mod, n_samples=1024, snr_db=snr_db,
                random_state=cls_idx * 10000 + i)
            signals.append(sig)
            labels.append(cls_idx)
    signals = np.array(signals)
    labels = np.array(labels)

    # ---- Step 2: 52-dim feature extraction ----
    print(f"\n[2] Extracting 52-dim features ...")
    X_52 = extract_features_batch(signals)
    X_52 = np.nan_to_num(X_52, nan=0.0, posinf=1e10, neginf=-1e10)

    # ---- Step 3: Elastic Net screening (52 -> 35) ----
    print(f"\n[3] Elastic Net Screening (52 -> 35) ...")
    screener = ElasticNetScreener(
        l1_ratio=0.5, n_folds=5,
        group_quota=(9, 10, 16),
        w_en=0.10, w_prior=0.90,
        random_state=42,
    )
    X_35 = screener.fit_transform(X_52, labels)
    feature_names_35 = screener.get_selected_feature_names()

    # ---- Step 4: ARFE (35 -> 27) ----
    print(f"\n[4] ARFE: Recursive Elimination (35 -> 27) ...")
    arfe = ARFE(
        n_heads=5,
        svm_C=1.0,
        n_eliminate_per_round=1,
        target_n_features=27,
        w_attn=0.15,
        w_svm=0.10,
        w_fisher=0.10,
        w_redundancy=0.20,
        w_stability=0.45,
        random_state=42,
    )
    X_27 = arfe.fit_transform(X_35, labels, feature_names_35)

    print(f"\n  Final feature matrix: {X_27.shape}")

    # Print final results
    arfe.print_summary()

    print(f"\n[5] Final {X_27.shape[1]} retained features:")
    for i, name in enumerate(arfe.get_selected_feature_names()):
        print(f"  {i+1:2d}. {name}")

    print("\n" + "=" * 70)
    print("ARFE complete. Ready for AACF cascade forest.")
    print("=" * 70)

    return X_27, labels, arfe, screener


if __name__ == '__main__':
    X_27, labels, arfe, screener = demo_arfe()