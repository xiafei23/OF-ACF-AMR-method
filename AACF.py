"""
OF-ACF AACF Module (v2 — Enhanced)
====================================
Adversarial Adaptive Cascade Forest for Automatic Modulation Recognition

Pipeline:
  Optimal HOC feature set (27-dim) → Structured multi-granularity scanning
  → Adaptive cascade forest → Modulation recognition

v2 Enhancements:
  [E1] Structured Grouped Scanning
       Physically meaningful grouping (HOC / normalized / quadratic features)
       combined with global sliding windows.

  [E2] Residual Feature Injection
       Original features are concatenated at every cascade layer to prevent
       information degradation in deeper layers.

  [E3] SNR Perturbation Augmentation
       Simulates varying channel SNR conditions to improve robustness,
       especially under low-SNR scenarios.

  [E4] Confusable-Class-Oriented Adversarial Training
       PGD-based perturbations focused on the most easily confused modulation
       pairs to enhance discriminability where it matters most.

  [E5] Multi-Layer Weighted Fusion
       Outputs from all cascade layers are aggregated using exponentially
       decaying weights to exploit complementary representations.

  [E6] Inter-Layer Feature Importance Feedback
       Random Forest-based feature importance is fed back to reweight features
       in subsequent layers for progressive refinement.

References:
  - Zhou & Feng, "Deep Forest," National Science Review, 2019.
  - Madry et al., "Towards Deep Learning Models Resistant to
    Adversarial Attacks," ICLR 2018.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# #####################################################################
#                 HYPERPARAMETER CONFIGURATION (CENTRALIZED)
# #####################################################################

AACF_CONFIG = {

    # =================================================================
    # 1. Structured Multi-Granularity Scanning
    # =================================================================
    # [E1] 27-dimensional features are grouped by physical semantics:
    #   Group-HOC (8 dims), Group-Norm (8 dims), Group-Quad (11 dims)
    # Each group is scanned independently + global scanning (27 dims),
    # resulting in 4 granularities in total.
    # Sliding window parameters apply within each group.
    'scan_window_length':   64,      # Window length within each group (auto-clipped if exceeding group size)
    'scan_window_stride':   32,      # Sliding stride within each group
    'scan_n_windows':       31,      # Number of windows (reference value, auto-adjusted in practice)
    'scan_n_estimators':    50,      # Number of trees per forest in scanning stage
    'scan_use_groups':      True,    # [E1] Enable grouped scanning (False = conventional sliding window)

    # =================================================================
    # 2. Random Forests (RFs) — Adaptive Growth
    # =================================================================
    'rf_min':               20,      # Minimum number of RFs (M1_min)
    'rf_max':               100,     # Maximum number of RFs (M1_max)
    'rf_init':              50,      # Initial number of RFs (first layer)
    'rf_growth_rate':       0.6,     # Growth rate μ1

    # =================================================================
    # 3. Extra Trees (ETs) — Adaptive Growth
    # =================================================================
    'et_min':               15,      # Minimum number of ETs (M2_min)
    'et_max':               60,      # Maximum number of ETs (M2_max)
    'et_init':              40,      # Initial number of ETs (first layer)
    'et_growth_rate':       0.4,     # Growth rate μ2

    # =================================================================
    # 4. Adversarial Training
    # =================================================================
    'at_epsilon':           0.02,    # Adversarial constraint ε_AT
    'at_pgd_steps':         5,       # Number of PGD iterations
    'at_step_size':         0.008,   # PGD step size α
    'at_delta':             1e-3,    # Finite-difference step for numerical gradient
    'at_confusion_focus':   True,    # [E4] Confusable-class-oriented adversarial training
    'at_confusion_weight':  2.0,     # [E4] Loss reweighting factor for confused samples

    # =================================================================
    # 5. SNR Perturbation Augmentation
    # =================================================================
    # [E3] Add Gaussian noise to simulate varying SNR conditions during training
    'snr_augment':          True,    # Enable SNR perturbation augmentation
    'snr_aug_levels':       3,       # Number of augmentation levels
    'snr_aug_sigma_min':    0.01,    # Minimum noise standard deviation
    'snr_aug_sigma_max':    0.10,    # Maximum noise standard deviation

    # =================================================================
    # 6. Cascade Structure
    # =================================================================
    'cascade_max_layers':   15,      # Maximum number of layers L_max
    'cascade_es_threshold': 0.005,   # Early stopping threshold
    'cascade_patience':     2,       # Early stopping patience
    'cascade_cv_folds':     5,       # Number of folds for K-fold CV
    'cascade_residual':     True,    # [E2] Residual feature injection (concatenate original features at each layer)

    # =================================================================
    # 7. Multi-Layer Weighted Ensemble
    # =================================================================
    # [E5] Use weighted fusion across all layers instead of only the best layer
    'ensemble_weighted':    True,    # Enable multi-layer weighted ensemble
    'ensemble_decay':       0.7,     # Exponential decay factor (farther from best layer → lower weight)

    # =================================================================
    # 8. Inter-Layer Feature Importance Feedback
    # =================================================================
    # [E6] Use RF feature_importances_ from previous layer to reweight next layer input
    'importance_feedback':  True,    # Enable feature importance feedback
    'importance_smooth':    0.3,     # Smoothing factor: w_new = (1-λ)·1 + λ·importance

    # =================================================================
    # 9. Base Learner Hyperparameters
    # =================================================================
    'tree_max_depth':       None,
    'tree_min_samples_leaf': 2,
    'tree_max_features':    'sqrt',

    # =================================================================
    # 10. General Settings
    # =================================================================
    'random_state':         42,
}

# #####################################################################
#                 END OF HYPERPARAMETER CONFIGURATION
# #####################################################################


def print_config(cfg: dict = None):
    """Print current hyperparameter configuration"""
    cfg = cfg or AACF_CONFIG
    print(f"\n{'='*60}")
    print(f"  AACF Hyperparameter Configuration (v2)")
    print(f"{'='*60}")
    sections = [
        ("Multi-Granularity Scanning", ['scan_window_length', 'scan_window_stride',
                        'scan_n_windows', 'scan_n_estimators',
                        'scan_use_groups']),
        ("Random Forests (RFs)", ['rf_min', 'rf_max', 'rf_init', 'rf_growth_rate']),
        ("Extra Trees (ETs)", ['et_min', 'et_max', 'et_init', 'et_growth_rate']),
        ("Adversarial Training", ['at_epsilon', 'at_pgd_steps', 'at_step_size',
                        'at_delta', 'at_confusion_focus',
                        'at_confusion_weight']),
        ("SNR Perturbation Augmentation", ['snr_augment', 'snr_aug_levels',
                         'snr_aug_sigma_min', 'snr_aug_sigma_max']),
        ("Cascade Forest", ['cascade_max_layers', 'cascade_es_threshold',
                       'cascade_patience', 'cascade_cv_folds',
                       'cascade_residual']),
        ("Multi-Layer Ensemble", ['ensemble_weighted', 'ensemble_decay']),
        ("Importance Feedback", ['importance_feedback', 'importance_smooth']),
        ("Base Learners", ['tree_max_depth', 'tree_min_samples_leaf',
                       'tree_max_features']),
        ("General", ['random_state']),
    ]
    for sec_name, keys in sections:
        print(f"\n  [{sec_name}]")
        for k in keys:
            print(f"    {k:30s} = {cfg[k]}")
    print(f"\n{'='*60}")


# =====================================================================
# [E1] Structured Grouped Scanning: HOC / Normalized / Quadratic groups
# =====================================================================
# Feature grouping indices for the 27-dim optimal feature set
# (strictly aligned with ARFE output order)
FEATURE_GROUPS = {
    'HOC':        list(range(0, 8)),    # |C20|...|C63| (8 dims)
    'Normalized': list(range(8, 16)),   # C40/C20...C63/C42 (8 dims)
    'Quadratic':  list(range(16, 27)),  # C40/C20²...C63/C21² (11 dims)
}


# =====================================================================
# Section 1: Structured Multi-Granularity Scanner
# =====================================================================

class MultiGranularityScanner:
    """
    [E1] Structured Multi-Granularity Scanning

    Conventional approach:
      Sliding window over 27-dim features → degenerates into a single window
      when window size > feature dimension, losing multi-granularity benefits.

    Proposed approach:
      Physically meaningful grouping + global view → 4 granularities:
        (a) HOC group (8 dims)        — coarse discrimination (raw cumulant magnitudes)
        (b) Normalized group (8 dims) — mid-level discrimination (ratios)
        (c) Quadratic group (11 dims) — fine-grained discrimination (higher-order ratios)
        (d) Global (27 dims)          — cross-group interactions

    Within each granularity:
      If group size ≤ window length → single global window;
      otherwise apply sliding windows.

    Each window trains RF + ET and outputs class probabilities,
    concatenated into A^(0).
    """

    def __init__(self, cfg: dict = None):
        cfg = cfg or AACF_CONFIG
        self.window_length = cfg['scan_window_length']
        self.window_stride = cfg['scan_window_stride']
        self.n_estimators = cfg['scan_n_estimators']
        self.use_groups = cfg['scan_use_groups']
        self.random_state = cfg['random_state']

        self.tree_params = {
            'max_depth': cfg['tree_max_depth'],
            'min_samples_leaf': cfg['tree_min_samples_leaf'],
            'max_features': cfg['tree_max_features'],
        }

        self.forests_ = {}
        self.scan_plan_ = []       # [(granularity_name, feature_index_list), ...]
        self.n_classes_ = None
        self.scan_dim_ = None

    def _build_scan_plan(self, p: int) -> list:
        """
        Build scanning plan: window slicing strategy for each granularity

        Returns: [(granularity_name, [feature indices]), ...]
        """
        plan = []

        if self.use_groups:
            # Grouped scanning
            for group_name, group_idx in FEATURE_GROUPS.items():
                g_len = len(group_idx)
                w = min(self.window_length, g_len)
                s = min(self.window_stride, max(1, w // 2))
                if w >= g_len:
                    # Group size ≤ window length: single global window
                    plan.append((f"{group_name}(Global)", group_idx))
                else:
                    n_win = (g_len - w) // s + 1
                    for wi in range(n_win):
                        start = wi * s
                        end = start + w
                        win_idx = group_idx[start:end]
                        plan.append((f"{group_name}[{start}:{end}]",
                                     win_idx))

            # Global window (cross-group interaction)
            plan.append(("Global", list(range(p))))

        else:
            # Conventional sliding window mode
            w = min(self.window_length, p)
            s = min(self.window_stride, max(1, w // 2))
            if w >= p:
                plan.append(("Global", list(range(p))))
            else:
                n_win = (p - w) // s + 1
                for wi in range(n_win):
                    start = wi * s
                    end = start + w
                    plan.append((f"Win[{start}:{end}]",
                                 list(range(start, end))))

        return plan

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train scanner and generate enhanced features A^(0)"""
        n_samples, p = X.shape
        self.n_classes_ = len(np.unique(y))
        self.scan_plan_ = self._build_scan_plan(p)

        print(f"\n  [Structured Multi-Granularity Scanning]")
        print(f"    Input dim: {p}, Groups enabled: {self.use_groups}")
        print(f"    Scan plan: {len(self.scan_plan_)} granularities")
        for name, idx in self.scan_plan_:
            print(f"      {name:25s} → {len(idx)} features")

        all_proba = []
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=self.random_state)

        for g_idx, (g_name, feat_idx) in enumerate(self.scan_plan_):
            X_g = X[:, feat_idx]

            for forest_type in ['RF', 'ET']:
                if forest_type == 'RF':
                    model = RandomForestClassifier(
                        n_estimators=self.n_estimators, **self.tree_params,
                        random_state=self.random_state + g_idx)
                else:
                    model = ExtraTreesClassifier(
                        n_estimators=self.n_estimators, **self.tree_params,
                        random_state=self.random_state + g_idx + 500)

                proba = cross_val_predict(
                    model, X_g, y, cv=cv, method='predict_proba')
                all_proba.append(proba)

                model.fit(X_g, y)
                self.forests_[(g_idx, forest_type)] = model

        A0 = np.hstack(all_proba)
        self.scan_dim_ = A0.shape[1]
        print(f"    Scan output: {A0.shape[1]} dims "
              f"= {len(self.scan_plan_)} × 2 forests × "
              f"{self.n_classes_} classes")
        return A0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply scanning to new data"""
        all_proba = []
        for g_idx, (_, feat_idx) in enumerate(self.scan_plan_):
            X_g = X[:, feat_idx]
            for forest_type in ['RF', 'ET']:
                model = self.forests_[(g_idx, forest_type)]
                all_proba.append(model.predict_proba(X_g))
        return np.hstack(all_proba)

# =====================================================================
# [E3] SNR Perturbation Augmenter
# =====================================================================

class SNRAugmenter:
    """
    SNR Perturbation Augmentation

    Adds Gaussian noise with varying intensity levels to the feature matrix
    during training, simulating channel conditions under different
    signal-to-noise ratios (SNRs).

    This improves model robustness to SNR variations commonly observed in
    real-world communication systems.

    Mechanism:
      - Generate L noise levels
      - σ_l are evenly spaced within [σ_min, σ_max]
      - Each level produces an augmented copy of the dataset
      - Final output = original data + all augmented samples
    """

    def __init__(self, cfg: dict = None):
        cfg = cfg or AACF_CONFIG
        self.enabled = cfg['snr_augment']
        self.n_levels = cfg['snr_aug_levels']
        self.sigma_min = cfg['snr_aug_sigma_min']
        self.sigma_max = cfg['snr_aug_sigma_max']
        self.random_state = cfg['random_state']

    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented data.

        Returns
        -------
        (X_aug, y_aug) : tuple
            Concatenation of original and augmented datasets.
        """
        if not self.enabled or self.n_levels <= 0:
            return X, y

        rng = np.random.RandomState(self.random_state)
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.n_levels)

        X_parts = [X]
        y_parts = [y]

        for sigma in sigmas:
            noise = rng.randn(*X.shape) * sigma
            X_parts.append(X + noise)
            y_parts.append(y.copy())

        return np.vstack(X_parts), np.concatenate(y_parts)


# =====================================================================
# [E4] Enhanced Adversarial Trainer
# =====================================================================

class AdversarialTrainer:
    """
    Adversarial Training (PGD + Confusion-Aware Targeting)

    Standard PGD:
      ζ^(k+1) ← Π_{||ζ||₂≤ε} { ζ^(k) + α · ∇_ζL / ||∇_ζL||₂ }

    [E4] Confusion-Aware Targeting:
      Samples belonging to highly confused class pairs are assigned
      larger weights during loss computation.

      Procedure:
        1. Compute confusion matrix
        2. Identify the most confused target class per class
        3. Select pairs with error rate > threshold (e.g., 3%)
        4. Apply higher loss weight to these samples

      Effect:
        Forces PGD to generate stronger adversarial perturbations
        on the model's weakest decision boundaries.
    """

    def __init__(self, cfg: dict = None):
        cfg = cfg or AACF_CONFIG
        self.epsilon = cfg['at_epsilon']
        self.n_pgd_steps = cfg['at_pgd_steps']
        self.step_size = cfg['at_step_size']
        self.delta = cfg['at_delta']
        self.confusion_focus = cfg['at_confusion_focus']
        self.confusion_weight = cfg['at_confusion_weight']
        self.random_state = cfg['random_state']

        # [E4] Runtime-updated confusion mask
        self._confusion_mask = None

    def update_confusion_info(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        [E4] Update confusion-aware target samples using confusion matrix.

        Identifies the most frequently confused class pairs and marks
        samples belonging to these classes.
        """
        if not self.confusion_focus:
            return

        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        cm_offdiag = cm.astype(float).copy()
        np.fill_diagonal(cm_offdiag, 0)

        confused_classes = set()
        for c in range(n_classes):
            if cm_offdiag[c].sum() > 0:
                target = np.argmax(cm_offdiag[c])
                err_rate = cm_offdiag[c, target] / max(cm[c].sum(), 1)

                if err_rate > 0.03:
                    confused_classes.add(c)
                    confused_classes.add(target)

        self._confusion_mask = np.isin(y_true, list(confused_classes))

    def _weighted_cross_entropy(self, proba, y, n_classes):
        """
        [E4] Weighted cross-entropy loss.

        Samples belonging to confusion-prone classes receive higher weights.
        """
        proba = np.clip(proba, 1e-12, 1 - 1e-12)

        loss = 0.0
        total_weight = 0.0

        for i in range(len(y)):
            w = 1.0
            if (
                self.confusion_focus
                and self._confusion_mask is not None
                and i < len(self._confusion_mask)
                and self._confusion_mask[i]
            ):
                w = self.confusion_weight

            loss -= w * np.log(proba[i, y[i]])
            total_weight += w

        return loss / max(total_weight, 1e-12)

    def _predict_proba_ensemble(self, H, rf_models, et_models, n_classes):
        """Ensemble prediction: average of RF and ET outputs."""
        proba_sum = np.zeros((H.shape[0], n_classes))
        n_models = 0

        for model in rf_models + et_models:
            proba_sum += model.predict_proba(H)
            n_models += 1

        return proba_sum / max(n_models, 1)

    def _l2_project(self, zeta, epsilon):
        """Projection onto L2 ball."""
        norms = np.linalg.norm(zeta, axis=1, keepdims=True)
        return zeta * np.minimum(1.0, epsilon / (norms + 1e-12))

    def generate_adversarial(self, H, y, rf_models, et_models, n_classes):
        """
        PGD with confusion-aware adversarial targeting.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, d = H.shape

        zeta = rng.randn(n_samples, d) * (self.epsilon * 0.1)
        zeta = self._l2_project(zeta, self.epsilon)

        for _ in range(self.n_pgd_steps):
            n_coords = min(d, max(5, d // 3))
            coord_idx = rng.choice(d, n_coords, replace=False)

            grad = np.zeros_like(zeta)

            for j in coord_idx:
                e_j = np.zeros(d)
                e_j[j] = self.delta

                H_plus = H + zeta + e_j
                p_plus = self._predict_proba_ensemble(H_plus, rf_models, et_models, n_classes)
                l_plus = self._weighted_cross_entropy(p_plus, y, n_classes)

                H_minus = H + zeta - e_j
                p_minus = self._predict_proba_ensemble(H_minus, rf_models, et_models, n_classes)
                l_minus = self._weighted_cross_entropy(p_minus, y, n_classes)

                grad[:, j] = (l_plus - l_minus) / (2 * self.delta)

            grad_norm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
            zeta = zeta + self.step_size * grad / grad_norm
            zeta = self._l2_project(zeta, self.epsilon)

        return zeta

    def adversarial_train_layer(self, H, y, n_rf, n_et, rf_params, et_params, n_classes):
        """
        Adversarial training pipeline:
        clean training → PGD perturbation → retraining on mixed data.
        """
        rf_init = [RandomForestClassifier(n_estimators=n_rf, **rf_params).fit(H, y)]
        et_init = [ExtraTreesClassifier(n_estimators=n_et, **et_params).fit(H, y)]

        y_pred = np.argmax(
            self._predict_proba_ensemble(H, rf_init, et_init, n_classes), axis=1
        )
        self.update_confusion_info(y, y_pred)

        zeta_star = self.generate_adversarial(H, y, rf_init, et_init, n_classes)

        H_adv = H + zeta_star
        H_combined = np.vstack([H, H_adv])
        y_combined = np.concatenate([y, y])

        rf_models = [RandomForestClassifier(n_estimators=n_rf, **rf_params).fit(H_combined, y_combined)]
        et_models = [ExtraTreesClassifier(n_estimators=n_et, **et_params).fit(H_combined, y_combined)]

        return rf_models, et_models


# =====================================================================
# Section 3: Cascade Layer
# =====================================================================

class CascadeLayer:
    """
    Single layer of the cascade forest.

    Includes [E6] feature importance extraction, which is fed into
    subsequent layers for adaptive feature reweighting.
    """

    def __init__(self, layer_id, n_rf, n_et, rf_params, et_params,
                 adv_trainer, n_classes, cv_folds=5):
        self.layer_id = layer_id
        self.n_rf = n_rf
        self.n_et = n_et
        self.rf_params = rf_params
        self.et_params = et_params
        self.adv_trainer = adv_trainer
        self.n_classes = n_classes
        self.cv_folds = cv_folds

        self.rf_models_ = None
        self.et_models_ = None
        self.residual_rate_ = None
        self.accuracy_ = None
        self.feature_importances_ = None

    def fit_predict(self, H, y):
        """
        Train the layer + generate K-fold probability outputs + extract feature importance.
        """
        self.rf_models_, self.et_models_ = self.adv_trainer.adversarial_train_layer(
            H, y, self.n_rf, self.n_et,
            self.rf_params, self.et_params, self.n_classes
        )

        # [E6] Feature importance (average of RF and ET)
        imp = np.zeros(H.shape[1])
        for m in self.rf_models_ + self.et_models_:
            imp += m.feature_importances_

        self.feature_importances_ = imp / (len(self.rf_models_) + len(self.et_models_))

        # K-fold cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=42 + self.layer_id)

        proba_cv = np.zeros((len(y), self.n_classes))

        for train_idx, val_idx in skf.split(H, y):
            rf_fold, et_fold = self.adv_trainer.adversarial_train_layer(
                H[train_idx], y[train_idx], self.n_rf, self.n_et,
                self.rf_params, self.et_params, self.n_classes
            )

            p_rf = np.mean([m.predict_proba(H[val_idx]) for m in rf_fold], axis=0)
            p_et = np.mean([m.predict_proba(H[val_idx]) for m in et_fold], axis=0)

            proba_cv[val_idx] = (p_rf + p_et) / 2.0

        y_pred = np.argmax(proba_cv, axis=1)

        self.residual_rate_ = np.mean(y_pred != y)
        self.accuracy_ = np.mean(y_pred == y)

        return proba_cv

    def predict_proba(self, H):
        """Predict class probabilities for input features."""
        p_rf = np.mean([m.predict_proba(H) for m in self.rf_models_], axis=0)
        p_et = np.mean([m.predict_proba(H) for m in self.et_models_], axis=0)

        return (p_rf + p_et) / 2.0

# =====================================================================
# Section 4: AACF v2
# =====================================================================

class AACF:
    """
        Adversarial Adaptive Cascade Forest v2

        Integrates all six improvements:
          [1] Structured grouped scanning     [2] Residual input
          [3] SNR perturbation augmentation   [4] Confusion-class-pair targeted adversarial training
          [5] Multi-layer weighted fusion     [6] Inter-layer feature importance feedback
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or AACF_CONFIG
        self.scanner_ = None
        self.snr_aug_ = None
        self.layers_ = []
        self.layer_metrics_ = []
        self.layer_proba_ = []    # [改进5] 保存各层概率
        self.n_classes_ = None
        self.best_layer_idx_ = None
        self.scaler_ = StandardScaler()
        self.X_original_ = None   # [改进2] 缓存原始特征

    def _adaptive_tree_count(self, rho, layer):
        """Adaptive tree count: use initial values for layer 1, then M = M_min + μ·ρ·(M_max-M_min)"""
        c = self.cfg
        if layer == 1:
            return c['rf_init'], c['et_init']
        M1 = int(c['rf_min'] + c['rf_growth_rate'] * rho
                 * (c['rf_max'] - c['rf_min']))
        M2 = int(c['et_min'] + c['et_growth_rate'] * rho
                 * (c['et_max'] - c['et_min']))
        return max(M1, c['rf_min']), max(M2, c['et_min'])

    def _apply_importance_weighting(self, H: np.ndarray,
                                    importances: np.ndarray) -> np.ndarray:
        """
        [Improvement 6] Inter-layer feature importance feedback

        w = (1-λ)·1 + λ·normalize(importance)
        H_weighted = H ⊙ w
        """
        lam = self.cfg['importance_smooth']
        imp_norm = importances / (importances.max() + 1e-12)
        weights = (1.0 - lam) * np.ones_like(imp_norm) + lam * imp_norm
        return H * weights[np.newaxis, :]

    def fit(self, X, y, feature_names=None):
        """Train AACF v2"""
        c = self.cfg
        self.n_classes_ = len(np.unique(y))

        print(f"\n{'='*70}")
        print(f"AACF v2: Adversarial Adaptive Cascade Forest (Enhanced)")
        print(f"  Input: {X.shape[1]}-dim, {X.shape[0]} samples, "
              f"{self.n_classes_} classes")
        print_config(c)

        # Standardization
        X_scaled = self.scaler_.fit_transform(X)
        self.X_original_ = X_scaled.copy()  # [Improvement 2]

        # [Improvement 3] SNR perturbation augmentation
        self.snr_aug_ = SNRAugmenter(cfg=c)
        X_aug, y_aug = self.snr_aug_.augment(X_scaled, y)
        if c['snr_augment']:
            print(f"\n  [SNR Augmentation] {X_scaled.shape[0]} → "
                  f"{X_aug.shape[0]} samples "
                  f"(+{c['snr_aug_levels']} noise levels)")

        # Multi-granularity scanning
        self.scanner_ = MultiGranularityScanner(cfg=c)
        A0_aug = self.scanner_.fit_transform(X_aug, y_aug)

        # Only keep scan features of original samples (augmented samples are only used for scanner training)
        n_orig = X_scaled.shape[0]
        A0 = A0_aug[:n_orig]

        # Cascade growth
        adv = AdversarialTrainer(cfg=c)
        base_params = {
            'max_depth': c['tree_max_depth'],
            'min_samples_leaf': c['tree_min_samples_leaf'],
            'max_features': c['tree_max_features'],
            'random_state': c['random_state'],
        }

        prev_rho = 1.0
        prev_proba = None
        prev_importance = None
        best_acc = 0.0
        no_improve = 0
        self.layers_ = []
        self.layer_metrics_ = []
        self.layer_proba_ = []

        print(f"\n  {'L':<4s} {'M1':>5s} {'M2':>5s} {'ρ':>8s} "
              f"{'Acc':>8s} {'Δ':>8s} {'Hdim':>6s} {'Status'}")
        print(f"  {'-'*62}")

        for l in range(1, c['cascade_max_layers'] + 1):
            M1, M2 = self._adaptive_tree_count(prev_rho, l)

            # ---- [Improvement 2] Residual input: always concatenate original features ----
            parts = [A0]
            if c['cascade_residual']:
                parts.append(X_scaled)
            if prev_proba is not None:
                parts.append(prev_proba)
            H_l = np.hstack(parts)

            # ---- [Improvement 6] Inter-layer feature importance feedback ----
            if (c['importance_feedback'] and prev_importance is not None
                    and prev_importance.shape[0] == H_l.shape[1]):
                H_l = self._apply_importance_weighting(H_l, prev_importance)

            # ---- 训练该层 ----
            layer = CascadeLayer(
                layer_id=l, n_rf=M1, n_et=M2,
                rf_params=base_params, et_params=base_params,
                adv_trainer=adv, n_classes=self.n_classes_,
                cv_folds=c['cascade_cv_folds'])

            proba_l = layer.fit_predict(H_l, y)

            rho_l = layer.residual_rate_
            acc_l = layer.accuracy_
            d_acc = acc_l - best_acc

            if d_acc >= c['cascade_es_threshold']:
                best_acc = acc_l
                self.best_layer_idx_ = l - 1
                no_improve = 0
                status = "✓ improved"
            else:
                no_improve += 1
                status = f"✗ ({no_improve}/{c['cascade_patience']})"

            self.layers_.append(layer)
            self.layer_metrics_.append({
                'layer': l, 'M1': M1, 'M2': M2,
                'rho': rho_l, 'accuracy': acc_l, 'delta_acc': d_acc,
                'H_dim': H_l.shape[1]})
            self.layer_proba_.append(proba_l)  # [Improvement 5]

            print(f"  {l:<4d} {M1:>5d} {M2:>5d} {rho_l:>8.4f} "
                  f"{acc_l:>8.4f} {d_acc:>+8.4f} {H_l.shape[1]:>6d} "
                  f"{status}")

            prev_rho = rho_l
            prev_proba = proba_l
            prev_importance = layer.feature_importances_  # [Improvement 6]

            if no_improve >= c['cascade_patience']:
                print(f"\n  Early stop at L{l} "
                      f"(best=L{self.best_layer_idx_+1}, "
                      f"acc={best_acc:.4f})")
                break

        print(f"\n{'='*70}")
        print(f"AACF v2 Done: {len(self.layers_)} layers, "
              f"best=L{self.best_layer_idx_+1} (acc={best_acc:.4f})")
        print(f"{'='*70}")
        return self

    # ------------------------------------------------------------------
    # [Improvement 5] Multi-layer weighted fusion prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Predict probabilities

        When ensemble_weighted=True, perform weighted fusion across all layers:
          w_l = decay^|l - best_layer| (exponential decay)
          P = Σ w_l · P_l / Σ w_l
        """
        c = self.cfg
        X_scaled = self.scaler_.transform(X)
        A0 = self.scanner_.transform(X_scaled)

        all_proba = []
        prev_proba = None

        for l_idx, layer in enumerate(self.layers_):
            parts = [A0]
            if c['cascade_residual']:
                parts.append(X_scaled)
            if prev_proba is not None:
                parts.append(prev_proba)
            H_l = np.hstack(parts)

            prev_proba = layer.predict_proba(H_l)
            all_proba.append(prev_proba)

        if c['ensemble_weighted'] and len(all_proba) > 1:
            # Exponential decay weighting
            decay = c['ensemble_decay']
            best = self.best_layer_idx_
            weights = np.array([
                decay ** abs(i - best) for i in range(len(all_proba))])

            # Only use layers up to best+1
            weights[best + 1:] *= 0.3  # later layers are down-weighted but not removed

            weighted_sum = sum(w * p for w, p in zip(weights, all_proba))
            return weighted_sum / weights.sum()
        else:
            return all_proba[self.best_layer_idx_]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def print_summary(self):
        if not self.layers_:
            print("AACF not trained."); return
        print(f"\n  AACF v2 Layer Summary:")
        print(f"  {'L':<4s} {'M1':>5s} {'M2':>5s} {'ρ':>8s} "
              f"{'Acc':>8s} {'Δ':>8s} {'Hdim':>6s}")
        print(f"  {'-'*50}")
        for m in self.layer_metrics_:
            mk = " ◄" if m['layer'] == self.best_layer_idx_ + 1 else ""
            print(f"  {m['layer']:<4d} {m['M1']:>5d} {m['M2']:>5d} "
                  f"{m['rho']:>8.4f} {m['accuracy']:>8.4f} "
                  f"{m['delta_acc']:>+8.4f} {m['H_dim']:>6d}{mk}")
        # 启用的改进项
        c = self.cfg
        enabled = []
        if c['scan_use_groups']:      enabled.append("Grouped scanning")
        if c['cascade_residual']:     enabled.append("Residual input")
        if c['snr_augment']:          enabled.append("SNR augmentation")
        if c['at_confusion_focus']:   enabled.append("Confusion-aware adversarial")
        if c['ensemble_weighted']:    enabled.append("Multi-layer fusion")
        if c['importance_feedback']:  enabled.append("Importance feedback")
        print(f"\n  Active improvements: {', '.join(enabled)}")


# =====================================================================
# Section 5: Demo
# =====================================================================

def demo_aacf():
    """Whole pipeline: 52 → 35 → 27 → AACF v2"""
    from Elastic import (
        generate_modulated_signal, extract_features_batch,
        ElasticNetScreener
    )
    from ARFE import ARFE

    print("=" * 70)
    print("OF-ACF Complete Pipeline (v2 Enhanced)")
    print("52 → 35 → 27 → AACF v2 → Modulation Recognition")
    print("=" * 70)

    mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '2FSK', '4FSK']
    n_per_class = 200
    snr_db = 10.0

    print(f"\n[1] Signal generation: {len(mod_types)} mods × "
          f"{n_per_class} samples @ SNR={snr_db}dB")
    signals, labels = [], []
    for ci, mod in enumerate(mod_types):
        for i in range(n_per_class):
            sig = generate_modulated_signal(
                mod, n_samples=1024, snr_db=snr_db,
                random_state=ci * 10000 + i)
            signals.append(sig)
            labels.append(ci)
    signals = np.array(signals)
    labels = np.array(labels)

    print(f"\n[2] Feature extraction (52-dim) ...")
    X_52 = extract_features_batch(signals)
    X_52 = np.nan_to_num(X_52, nan=0.0, posinf=1e10, neginf=-1e10)

    print(f"\n[3] Elastic Net (52 → 35) ...")
    screener = ElasticNetScreener(
        l1_ratio=0.5, n_folds=5, group_quota=(9, 10, 16),
        w_en=0.10, w_prior=0.90, random_state=42)
    X_35 = screener.fit_transform(X_52, labels)
    names_35 = screener.get_selected_feature_names()

    print(f"\n[4] ARFE (35 → 27) ...")
    arfe = ARFE(n_heads=5, svm_C=1.0, target_n_features=27,
                random_state=42)
    X_27 = arfe.fit_transform(X_35, labels, names_35)
    names_27 = arfe.get_selected_feature_names()

    print(f"\n[5] AACF v2 ...")
    aacf = AACF(cfg=AACF_CONFIG)
    aacf.fit(X_27, labels, feature_names=names_27)

    train_acc = aacf.score(X_27, labels)
    print(f"\n  Train accuracy: {train_acc:.4f}")
    aacf.print_summary()

    print(f"\n[6] Optimal features ({len(names_27)}):")
    for i, name in enumerate(names_27):
        print(f"  {i+1:2d}. {name}")

    print("\n" + "=" * 70)
    print("OF-ACF v2 pipeline complete.")
    print("=" * 70)
    return aacf, arfe, screener


if __name__ == '__main__':
    aacf, arfe, screener = demo_aacf()