"""
OF-ACF ARFE Core Module
========================
Augmented Recursive Feature Elimination with Feature-level Multi-Head Attention

Core Components:
  - FMHA:             Feature-level multi-head attention (implicit QKV-free formulation)
  - ImportanceScorer: Multi-criteria feature importance scoring (FMHA + SVM + Fisher + redundancy + prior)
  - ARFE:             Augmented recursive feature elimination (iterative FMHA → SVM → scoring → elimination)

External Interface:
  arfe = ARFE(**params)
  arfe.fit(X, y, feature_names)     # X: feature matrix of arbitrary dimension
  X_sel = arfe.transform(X)         # return selected feature matrix
  names = arfe.get_selected_feature_names()

Dependencies: numpy, scikit-learn
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

__all__ = ['FMHA', 'ImportanceScorer', 'ARFE', 'parse_feature']


# =====================================================================
# Utility Functions
# =====================================================================

def parse_feature(name: str) -> dict:
    """
        Parse feature name into structured information.

        Returns
        -------
        dict with keys:
          type : 'hoc' | 'normalized' | 'quadratic' | 'unknown'
          cumulant / num / den : str   cumulant identifiers
          order / num_order / den_order : int   order
          q / num_q / den_q : int      conjugation count
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
# FMHA: Feature-level Multi-Head Attention
# =====================================================================

class FMHA:
    """
    Feature-level multi-head attention mechanism.

    Instead of explicitly constructing Q, K, V, attention is implicitly
    modeled via the feature self-similarity matrix:

      S_{n,h} = softmax( X_n^(h) · X_n^(h)^T / sqrt(d_k) )
      alpha_{h,j} = softmax( sum_l S[j,l] )
      Z_{n,h} = alpha_h ⊙ X_n^(h)   (Hadamard product)
      Z_n = [Z_{n,1}, ..., Z_{n,H}]

    Parameters
    ----------
    n_heads : int
        Number of attention heads H.
    random_state : int
        Random seed (controls feature-to-head assignment).
    """

    def __init__(self, n_heads: int = 5, random_state: int = 42):
        self.n_heads = n_heads
        self.random_state = random_state
        self.head_assignment_ = None
        self.attention_weights_ = None

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

    def _assign_heads(self, p_t: int) -> List[np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(p_t)
        d_k = p_t // self.n_heads
        remainder = p_t % self.n_heads
        heads, start = [], 0
        for h in range(self.n_heads):
            size = d_k + (1 if h < remainder else 0)
            heads.append(np.sort(perm[start:start + size]))
            start += size
        return heads

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        X : (n_samples, p_t)
            Standardized feature matrix.

        Returns
        -------
        Z : (n_samples, p_t)
            Attention-weighted feature representation.
        alpha : (p_t,)
            Average attention weight of each feature.
        """
        n_samples, p_t = X.shape
        heads = self._assign_heads(p_t)
        self.head_assignment_ = heads

        Z = np.zeros_like(X)
        alpha_full = np.zeros(p_t)

        for h, h_idx in enumerate(heads):
            d_k = len(h_idx)
            X_h = X[:, h_idx]
            alpha_accum = np.zeros(d_k)

            for n in range(n_samples):
                x_n = X_h[n]
                sim = np.outer(x_n, x_n) / (np.sqrt(d_k) + 1e-12)
                S_nh = self._softmax(sim, axis=1)
                row_agg = S_nh.sum(axis=1)
                alpha_accum += self._softmax(row_agg)

            alpha_h = alpha_accum / n_samples
            Z[:, h_idx] = alpha_h[np.newaxis, :] * X_h
            alpha_full[h_idx] = alpha_h

        self.attention_weights_ = alpha_full
        return Z, alpha_full


# =====================================================================
# ImportanceScorer: Multi-criteria feature importance scoring
# =====================================================================

class ImportanceScorer:
    """
        Multi-criteria scorer:
        FMHA attention + SVM coefficients + Fisher ratio + redundancy penalty + prior stability

        Score:
            I(j) = w_a * a_j + w_s * |w_j| + w_F * F_j + w_r * (1 - rho_j) + w_sig * sig_j

        The prior parameters are derived from cumulant estimation theory
        (Swami & Sadler 2000; Nandi & Azzouz 1998).
        """

    # ---- Cumulant-theory-based parameter tables ----
    _NUM_DISC = {
        'C20': 0.42, 'C21': 0.05,
        'C40': 0.92, 'C41': 0.55, 'C42': 0.85,
        'C60': 0.65, 'C61': 0.72, 'C62': 0.95, 'C63': 0.88,
    }
    _HOC_ADJ = {'C21': -0.15}
    _DEN_STAB_NORM = {
        'C20': 0.70, 'C21': 0.95, 'C40': 0.88, 'C41': 0.15,
        'C42': 0.78, 'C60': 0.50,
    }
    _PAIR_ADJ_NORM = {('C62', 'C60'): -0.30}
    _DEN_STAB_SQ = {
        'C20': 0.68, 'C21': 0.90, 'C40': 0.52, 'C41': 0.02,
        'C42': 0.02, 'C60': 0.10, 'C61': 0.05,
    }
    _SAME_ORDER_PEN = {4: -0.25, 6: -0.22}
    _PAIR_ADJ_QUAD = {
        ('C41', 'C20'): -0.18,
        ('C63', 'C20'): -0.30,
        ('C60', 'C41'): -0.15,
        ('C61', 'C42'): -0.12,
        ('C61', 'C40'): +0.08,
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
        self._stability_cache = self._compute_stability_prior(feature_names)

    def _compute_stability_prior(self, names: List[str]) -> np.ndarray:
        """Compute stability prior based on cumulant theory"""
        scores = np.zeros(len(names))
        for i, name in enumerate(names):
            info = parse_feature(name)

            if info['type'] == 'hoc':
                cum = info['cumulant']
                scores[i] = (self._NUM_DISC.get(cum, 0.5)
                             + self._HOC_ADJ.get(cum, 0.0))

            elif info['type'] == 'normalized':
                num, den = info['num'], info['den']
                base = (self._NUM_DISC.get(num, 0.5) *
                        self._DEN_STAB_NORM.get(den, 0.5))
                if info['num_order'] == info['den_order']:
                    base += self._SAME_ORDER_PEN.get(info['num_order'], -0.10)
                if info['num_order'] == 2 and info['den_order'] == 2:
                    base -= 0.10
                base += self._PAIR_ADJ_NORM.get((num, den), 0.0)
                scores[i] = base

            elif info['type'] == 'quadratic':
                num, den = info['num'], info['den']
                base = (self._NUM_DISC.get(num, 0.5) *
                        self._DEN_STAB_SQ.get(den, 0.3))
                if info['num_order'] == info['den_order']:
                    pen = self._SAME_ORDER_PEN.get(info['num_order'], -0.10)
                    base += pen * 1.2
                if info['den_q'] % 2 == 1 and info['den_q'] > 0:
                    base -= 0.05
                base += self._PAIR_ADJ_QUAD.get((num, den), 0.0)
                scores[i] = base

        return scores

    @staticmethod
    def fisher_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fisher discriminant ratio: F_j = var_between / var_within"""
        classes = np.unique(y)
        global_mean = X.mean(axis=0)
        between = np.zeros(X.shape[1])
        within = np.zeros(X.shape[1])
        for c in classes:
            mask = (y == c)
            n_c = mask.sum()
            between += n_c * (X[mask].mean(axis=0) - global_mean) ** 2
            within += X[mask].var(axis=0) * n_c
        return between / (within + 1e-12)

    @staticmethod
    def max_abs_correlation(X: np.ndarray) -> np.ndarray:
        """Redundancy: max |corr(f_j, f_k)|, k ≠ j"""
        if X.shape[1] < 2:
            return np.zeros(X.shape[1])
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0.0)
        return np.max(np.abs(corr), axis=1)

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-15:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    def score(self, X, y, alpha, svm_coef, active_indices):
        """Composite importance score (higher is more important)"""
        return (self.w_attn * self._normalize_01(alpha)
                + self.w_svm * self._normalize_01(svm_coef)
                + self.w_fisher * self._normalize_01(self.fisher_ratio(X, y))
                + self.w_redundancy * self._normalize_01(
                    1.0 - self.max_abs_correlation(X))
                + self.w_stability * self._normalize_01(
                    self._stability_cache[active_indices]))


# =====================================================================
# ARFE: Augmented Recursive Feature Elimination
# =====================================================================

class ARFE:
    """
        Augmented Recursive Feature Elimination

        Procedure:
            X^(t) → FMHA weighting → Soft-SVM → multi-criteria scoring → eliminate weakest → iterate

        Parameters
        ----------
        n_heads : int               Number of attention heads
        svm_C : float               Soft-SVM regularization parameter
        n_eliminate_per_round : int Number of features eliminated per round
        target_n_features : int     Target number of retained features
        w_attn : float              Attention weight
        w_svm : float               SVM weight
        w_fisher : float            Fisher weight
        w_redundancy : float        Redundancy penalty weight
        w_stability : float         Stability prior weight
        random_state : int          Random seed
        verbose : bool              Whether to print progress

        Attributes (after fitting)
        ----------
        selected_indices_ : ndarray   Indices of retained features
        elimination_history_ : list   Elimination order
        round_details_ : list[dict]   Per-round details
        feature_names_ : list         Input feature names
        """

    def __init__(self, n_heads: int = 5, svm_C: float = 1.0,
                 n_eliminate_per_round: int = 1,
                 target_n_features: int = 27,
                 w_attn: float = 0.15, w_svm: float = 0.10,
                 w_fisher: float = 0.10, w_redundancy: float = 0.20,
                 w_stability: float = 0.45,
                 random_state: int = 42, verbose: bool = True):
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
        self.verbose = verbose

        self.selected_indices_ = None
        self.elimination_history_ = []
        self.round_details_ = []
        self.feature_names_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'ARFE':
        """
                Perform recursive feature elimination.

                Parameters
                ----------
                X : (n_samples, n_features)
                    Feature matrix
                y : (n_samples,)
                    Class labels (integer-encoded)
                feature_names : list of length n_features
        """
        self.feature_names_ = list(feature_names)
        self.elimination_history_ = []
        self.round_details_ = []

        n_init = X.shape[1]
        assert n_init == len(feature_names)
        assert n_init > self.target_n

        T = (n_init - self.target_n) // self.n_eliminate
        scorer = ImportanceScorer(
            feature_names, self.w_attn, self.w_svm,
            self.w_fisher, self.w_redundancy, self.w_stability)

        active = np.arange(n_init)
        scaler = StandardScaler()

        self._log(f"\nARFE: {n_init} -> {self.target_n} "
                  f"({T} rounds, eliminate {self.n_eliminate}/round)")

        for t in range(T):
            p_t = len(active)
            X_scaled = scaler.fit_transform(X[:, active])

            fmha = FMHA(self.n_heads, self.random_state + t)
            Z_t, alpha = fmha.forward(X_scaled)

            svm = SVC(kernel='linear', C=self.svm_C,
                      decision_function_shape='ovr',
                      random_state=self.random_state)
            svm.fit(Z_t, y)
            svm_coef = np.mean(np.abs(svm.coef_), axis=0)

            composite = scorer.score(X_scaled, y, alpha, svm_coef, active)

            n_elim = min(self.n_eliminate, p_t - self.target_n)
            weakest = np.argsort(composite)[:n_elim]
            elim_names = [self.feature_names_[active[w]] for w in weakest]
            self.elimination_history_.extend(elim_names)

            self.round_details_.append({
                'round': t + 1, 'p_before': p_t,
                'eliminated': elim_names,
                'scores': {self.feature_names_[active[j]]: float(composite[j])
                           for j in range(p_t)},
            })

            self._log(f"  R{t+1}/{T}: {p_t}->{p_t-n_elim}  "
                      f"x {', '.join(elim_names)}")

            active = np.delete(active, weakest)

        self.selected_indices_ = active
        self._log(f"  => {len(active)} features retained")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the selected feature submatrix"""
        assert self.selected_indices_ is not None, "Call fit() first"
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y, feature_names):
        return self.fit(X, y, feature_names).transform(X)

    def get_selected_feature_names(self) -> List[str]:
        return [self.feature_names_[i] for i in self.selected_indices_]

    def get_elimination_order(self) -> List[str]:
        return list(self.elimination_history_)

    def get_group_distribution(self) -> Dict[str, int]:
        """Distribution of retained features: HOC / Normalized / Quadratic"""
        sel = self.get_selected_feature_names()
        return {
            'HOC': sum(1 for n in sel if n.startswith('|')),
            'Normalized': sum(1 for n in sel if '/' in n and '^2' not in n),
            'Quadratic': sum(1 for n in sel if '^2' in n),
        }

    def summary(self) -> dict:
        """Structured summary (for JSON serialization / experiment logging)"""
        return {
            'n_input': len(self.feature_names_),
            'n_output': len(self.selected_indices_),
            'n_rounds': len(self.round_details_),
            'selected_names': self.get_selected_feature_names(),
            'selected_indices': self.selected_indices_.tolist(),
            'elimination_order': self.elimination_history_,
            'group_distribution': self.get_group_distribution(),
            'params': {
                'n_heads': self.n_heads, 'svm_C': self.svm_C,
                'n_eliminate': self.n_eliminate, 'target_n': self.target_n,
                'w_attn': self.w_attn, 'w_svm': self.w_svm,
                'w_fisher': self.w_fisher, 'w_redundancy': self.w_redundancy,
                'w_stability': self.w_stability,
            },
        }

    def print_summary(self):
        """Print concise summary"""
        if self.selected_indices_ is None:
            print("ARFE not fitted."); return
        sel = self.get_selected_feature_names()
        grp = self.get_group_distribution()
        print(f"\nARFE: {len(self.feature_names_)} -> {len(sel)}")
        for g, cnt in grp.items():
            print(f"  {g}: {cnt}")
        print(f"Retained: {sel}")
        print(f"Eliminated: {self.elimination_history_}")