"""
OF-ACF AACF Core Module
========================
Adversarial Adaptive Cascade Forest for Automatic Modulation Recognition

Core Components:
  - MultiGranularityScanner: Structured multi-granularity scanning (group-wise + global)
  - SNRAugmenter:            SNR perturbation augmentation
  - AdversarialTrainer:      PGD-based adversarial training (with confusion-pair targeting)
  - CascadeLayer:            Single cascade layer (adversarial training + feature importance extraction)
  - AACF:                    Full cascade forest (adaptive growth, early stopping, multi-layer fusion)

External Interface:
  aacf = AACF(cfg={...})         # or use default AACF_CONFIG
  aacf.fit(X, y)
  y_pred = aacf.predict(X_test)
  proba  = aacf.predict_proba(X_test)
  acc    = aacf.score(X_test, y_test)
  info   = aacf.summary()

Dependencies: numpy, scikit-learn
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

__all__ = [
    'AACF_CONFIG', 'DEFAULT_FEATURE_GROUPS',
    'MultiGranularityScanner', 'SNRAugmenter',
    'AdversarialTrainer', 'CascadeLayer', 'AACF',
]


# #####################################################################
#           Default hyperparameter configuration (overridable)
# #####################################################################
# Usage:
#   cfg = AACF_CONFIG.copy()      # Copy default configuration
#   cfg['rf_max'] = 200           # Modify as needed
#   aacf = AACF(cfg=cfg)          # Initialize with custom configuration

AACF_CONFIG = {
    # --- Multi-granularity scanning ---
    'scan_window_length':   64,
    'scan_window_stride':   32,
    'scan_n_estimators':    50,
    'scan_use_groups':      True,     # Group-wise scanning based on physical semantics

    # --- Random Forests (RFs) adaptive growth ---
    'rf_min':               20,
    'rf_max':               100,
    'rf_init':              50,       # Cold start at the first layer
    'rf_growth_rate':       0.6,      # μ1

    ## --- Extremely Randomized Trees (ETs) adaptive growth ---
    'et_min':               15,
    'et_max':               60,
    'et_init':              40,
    'et_growth_rate':       0.4,      # μ2

    # --- Adversarial training ---
    'at_epsilon':           0.02,     # ε_AT
    'at_pgd_steps':         5,
    'at_step_size':         0.008,
    'at_delta':             1e-3,
    'at_confusion_focus':   True,     # 混淆类对定向
    'at_confusion_weight':  2.0,

    # --- SNR perturbation augmentation ---
    'snr_augment':          True,
    'snr_aug_levels':       3,
    'snr_aug_sigma_min':    0.01,
    'snr_aug_sigma_max':    0.10,

    # --- Cascade architecture ---
    'cascade_max_layers':   15,
    'cascade_es_threshold': 0.005,
    'cascade_patience':     2,
    'cascade_cv_folds':     5,
    'cascade_residual':     True,     # Concatenate original features at each layer

    # --- Multi-layer fusion ---
    'ensemble_weighted':    True,
    'ensemble_decay':       0.7,

    # --- Inter-layer importance feedback ---
    'importance_feedback':  True,
    'importance_smooth':    0.3,

    # --- Base learners ---
    'tree_max_depth':       None,
    'tree_min_samples_leaf': 2,
    'tree_max_features':    'sqrt',

    # --- General settings ---
    'random_state':         42,
    'verbose':              True,
}

# Index groups of the 27-dimensional optimal feature set
# (aligned with ARFE output ordering; externally overridable)
DEFAULT_FEATURE_GROUPS = {
    'HOC':        list(range(0, 8)),
    'Normalized': list(range(8, 16)),
    'Quadratic':  list(range(16, 27)),
}


# =====================================================================
# MultiGranularityScanner
# =====================================================================

class MultiGranularityScanner:
    """
    Structured multi-granularity scanning.

    Feature groups are constructed based on physical semantics,
    combined with a global window, resulting in G+1 granularities.
    For each granularity: RF + ET → K-class probabilities → concatenated as A^(0).

    Parameters
    ----------
    cfg : dict
        Hyperparameter configuration.
    feature_groups : dict, optional
        Feature group indices (default configuration is used if None).
    """

    def __init__(self, cfg: dict = None,
                 feature_groups: dict = None):
        cfg = cfg or AACF_CONFIG
        self.window_length = cfg['scan_window_length']
        self.window_stride = cfg['scan_window_stride']
        self.n_estimators = cfg['scan_n_estimators']
        self.use_groups = cfg['scan_use_groups']
        self.random_state = cfg['random_state']
        self.verbose = cfg.get('verbose', True)
        self.feature_groups = feature_groups or DEFAULT_FEATURE_GROUPS

        self.tree_params = {
            'max_depth': cfg['tree_max_depth'],
            'min_samples_leaf': cfg['tree_min_samples_leaf'],
            'max_features': cfg['tree_max_features'],
        }

        self.forests_ = {}
        self.scan_plan_ = []
        self.n_classes_ = None
        self.scan_dim_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _build_scan_plan(self, p: int) -> list:
        """Construct [(granularity_name, [feature_indices]), ...]"""
        plan = []
        if self.use_groups:
            for gname, gidx in self.feature_groups.items():
                g_len = len(gidx)
                w = min(self.window_length, g_len)
                s = min(self.window_stride, max(1, w // 2))
                if w >= g_len:
                    plan.append((f"{gname}", gidx))
                else:
                    n_win = (g_len - w) // s + 1
                    for wi in range(n_win):
                        st = wi * s
                        plan.append((f"{gname}[{st}:{st+w}]",
                                     gidx[st:st+w]))
            plan.append(("Global", list(range(p))))
        else:
            w = min(self.window_length, p)
            s = min(self.window_stride, max(1, w // 2))
            if w >= p:
                plan.append(("Global", list(range(p))))
            else:
                for wi in range((p - w) // s + 1):
                    st = wi * s
                    plan.append((f"W[{st}:{st+w}]",
                                 list(range(st, st+w))))
        return plan

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train scanning forests and generate A^(0) using K-fold probabilities"""
        n_samples, p = X.shape
        self.n_classes_ = len(np.unique(y))
        self.scan_plan_ = self._build_scan_plan(p)

        self._log(f"  Scanner: {len(self.scan_plan_)} granularities, "
                  f"groups={self.use_groups}")

        all_proba = []
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=self.random_state)

        for gi, (gname, fidx) in enumerate(self.scan_plan_):
            Xg = X[:, fidx]
            for ftype in ['RF', 'ET']:
                Cls = (RandomForestClassifier if ftype == 'RF'
                       else ExtraTreesClassifier)
                model = Cls(n_estimators=self.n_estimators,
                            **self.tree_params,
                            random_state=self.random_state + gi +
                            (500 if ftype == 'ET' else 0))
                proba = cross_val_predict(model, Xg, y, cv=cv,
                                          method='predict_proba')
                all_proba.append(proba)
                model.fit(Xg, y)
                self.forests_[(gi, ftype)] = model

        A0 = np.hstack(all_proba)
        self.scan_dim_ = A0.shape[1]
        self._log(f"  Scan output: {A0.shape[1]} dims")
        return A0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the trained scanning mechanism to new data"""
        all_proba = []
        for gi, (_, fidx) in enumerate(self.scan_plan_):
            Xg = X[:, fidx]
            for ftype in ['RF', 'ET']:
                all_proba.append(
                    self.forests_[(gi, ftype)].predict_proba(Xg))
        return np.hstack(all_proba)


# =====================================================================
# SNRAugmenter
# =====================================================================

class SNRAugmenter:
    """
    SNR perturbation augmentation: simulate channel variations
    by injecting Gaussian noise with different intensities.

    Parameters
    ----------
    cfg : dict
    """

    def __init__(self, cfg: dict = None):
        cfg = cfg or AACF_CONFIG
        self.enabled = cfg['snr_augment']
        self.n_levels = cfg['snr_aug_levels']
        self.sigma_min = cfg['snr_aug_sigma_min']
        self.sigma_max = cfg['snr_aug_sigma_max']
        self.random_state = cfg['random_state']

    def augment(self, X: np.ndarray,
                y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_aug, y_aug): concatenation of original and augmented data"""
        if not self.enabled or self.n_levels <= 0:
            return X, y
        rng = np.random.RandomState(self.random_state)
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.n_levels)
        Xs, ys = [X], [y]
        for sigma in sigmas:
            Xs.append(X + rng.randn(*X.shape) * sigma)
            ys.append(y.copy())
        return np.vstack(Xs), np.concatenate(ys)


# =====================================================================
# AdversarialTrainer
# =====================================================================

class AdversarialTrainer:
    """
    PGD-based adversarial training with confusion-pair-aware weighting.

    Inner optimization:
        ζ* = argmax_{||ζ||₂ ≤ ε} L(O[H + ζ; Θ], y)

    Outer optimization:
        Θ* = argmin (1/N) Σ L(O[H + ζ*; Θ], y)

    Parameters
    ----------
    cfg : dict
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
        self._confusion_mask = None

    def update_confusion_info(self, y_true, y_pred):
        """Identify samples belonging to confusion class pairs"""
        if not self.confusion_focus:
            return
        cm = confusion_matrix(y_true, y_pred)
        cm_off = cm.copy().astype(float)
        np.fill_diagonal(cm_off, 0)
        confused = set()
        for c in range(cm.shape[0]):
            if cm_off[c].sum() > 0:
                tgt = np.argmax(cm_off[c])
                if cm_off[c, tgt] / max(cm[c].sum(), 1) > 0.03:
                    confused.update([c, tgt])
        self._confusion_mask = np.isin(y_true, list(confused))

    def _weighted_ce(self, proba, y, n_classes):
        """Weighted cross-entropy (higher weights for confusing samples)"""
        pc = np.clip(proba, 1e-12, 1 - 1e-12)
        loss, tw = 0.0, 0.0
        for i in range(len(y)):
            w = 1.0
            if (self.confusion_focus and self._confusion_mask is not None
                    and i < len(self._confusion_mask)
                    and self._confusion_mask[i]):
                w = self.confusion_weight
            loss -= w * np.log(pc[i, y[i]])
            tw += w
        return loss / max(tw, 1e-12)

    def _ensemble_proba(self, H, rf_models, et_models, nc):
        """Average class probabilities from RF and ET"""
        ps = np.zeros((H.shape[0], nc))
        nm = 0
        for m in rf_models + et_models:
            ps += m.predict_proba(H)
            nm += 1
        return ps / max(nm, 1)

    def _l2_project(self, zeta, eps):
        norms = np.linalg.norm(zeta, axis=1, keepdims=True)
        return zeta * np.minimum(1.0, eps / (norms + 1e-12))

    def generate_adversarial(self, H, y, rf_models, et_models, nc):
        """PGD inner loop: generate adversarial perturbation ζ*"""
        rng = np.random.RandomState(self.random_state)
        n, d = H.shape
        zeta = rng.randn(n, d) * (self.epsilon * 0.1)
        zeta = self._l2_project(zeta, self.epsilon)

        for _ in range(self.n_pgd_steps):
            nc_ = min(d, max(5, d // 3))
            cidx = rng.choice(d, nc_, replace=False)
            grad = np.zeros_like(zeta)
            for j in cidx:
                ej = np.zeros(d); ej[j] = self.delta
                lp = self._weighted_ce(
                    self._ensemble_proba(H + zeta + ej, rf_models,
                                         et_models, nc), y, nc)
                lm = self._weighted_ce(
                    self._ensemble_proba(H + zeta - ej, rf_models,
                                         et_models, nc), y, nc)
                grad[:, j] = (lp - lm) / (2 * self.delta)
            gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
            zeta = self._l2_project(
                zeta + self.step_size * grad / gn, self.epsilon)
        return zeta

    def adversarial_train_layer(self, H, y, n_rf, n_et,
                                base_params, nc):
        """Adversarial training pipeline: clean → PGD → mixed retraining"""
        rf0 = [RandomForestClassifier(
            n_estimators=n_rf, **base_params).fit(H, y)]
        et0 = [ExtraTreesClassifier(
            n_estimators=n_et, **base_params).fit(H, y)]

        yp = np.argmax(self._ensemble_proba(H, rf0, et0, nc), axis=1)
        self.update_confusion_info(y, yp)

        zeta = self.generate_adversarial(H, y, rf0, et0, nc)
        Hc = np.vstack([H, H + zeta])
        yc = np.concatenate([y, y])

        rfs = [RandomForestClassifier(
            n_estimators=n_rf, **base_params).fit(Hc, yc)]
        ets = [ExtraTreesClassifier(
            n_estimators=n_et, **base_params).fit(Hc, yc)]
        return rfs, ets


# =====================================================================
# CascadeLayer
# =====================================================================

class CascadeLayer:
    """Cascade layer:
    Adversarial training → RF + ET → class probabilities + residual signal + feature importance
    """

    def __init__(self, layer_id, n_rf, n_et, base_params,
                 adv_trainer, n_classes, cv_folds=5):
        self.layer_id = layer_id
        self.n_rf = n_rf
        self.n_et = n_et
        self.base_params = base_params
        self.adv = adv_trainer
        self.nc = n_classes
        self.cv_folds = cv_folds

        self.rf_models_ = None
        self.et_models_ = None
        self.residual_rate_ = None
        self.accuracy_ = None
        self.feature_importances_ = None

    def fit_predict(self, H, y):
        """Training with K-fold probability estimation and feature importance extraction"""

        self.rf_models_, self.et_models_ = \
            self.adv.adversarial_train_layer(
                H, y, self.n_rf, self.n_et, self.base_params, self.nc)

        # Feature importance (averaged over RF and ET)
        imp = np.zeros(H.shape[1])
        for m in self.rf_models_ + self.et_models_:
            imp += m.feature_importances_
        self.feature_importances_ = imp / (
            len(self.rf_models_) + len(self.et_models_))

        # K-fold cross-validated class probabilities
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=42 + self.layer_id)
        proba_cv = np.zeros((len(y), self.nc))
        for tr, va in skf.split(H, y):
            rf_f, et_f = self.adv.adversarial_train_layer(
                H[tr], y[tr], self.n_rf, self.n_et,
                self.base_params, self.nc)
            p_rf = np.mean([m.predict_proba(H[va]) for m in rf_f], axis=0)
            p_et = np.mean([m.predict_proba(H[va]) for m in et_f], axis=0)
            proba_cv[va] = (p_rf + p_et) / 2.0

        yp = np.argmax(proba_cv, axis=1)
        self.residual_rate_ = np.mean(yp != y)
        self.accuracy_ = np.mean(yp == y)
        return proba_cv

    def predict_proba(self, H):
        p_rf = np.mean(
            [m.predict_proba(H) for m in self.rf_models_], axis=0)
        p_et = np.mean(
            [m.predict_proba(H) for m in self.et_models_], axis=0)
        return (p_rf + p_et) / 2.0


# =====================================================================
# AACF: Adversarial Adaptive Cascade Forest
# =====================================================================

class AACF:
    """
    Adversarial Adaptive Cascade Forest (AACF)

    Pipeline:
      X → normalization → SNR augmentation → structured scanning → A^(0)
        → layer-wise cascade (adversarial training + adaptive growth + residual propagation + importance feedback)
        → early stopping → multi-layer weighted fusion for prediction

    Parameters
    ----------
    cfg : dict
        Hyperparameter configuration (default: AACF_CONFIG)
    feature_groups : dict or None
        Feature group indices (default: DEFAULT_FEATURE_GROUPS)

    Attributes (after fitting)
    ----------
    scanner_ : MultiGranularityScanner
    layers_ : list[CascadeLayer]
    layer_metrics_ : list[dict]
        Per-layer evaluation metrics
    best_layer_idx_ : int
        Index of the optimal layer
    n_classes_ : int
    """

    def __init__(self, cfg: dict = None,
                 feature_groups: dict = None):
        self.cfg = cfg or AACF_CONFIG
        self.feature_groups = feature_groups
        self.scanner_ = None
        self.snr_aug_ = None
        self.layers_ = []
        self.layer_metrics_ = []
        self.layer_proba_ = []
        self.n_classes_ = None
        self.best_layer_idx_ = None
        self.scaler_ = StandardScaler()
        self.X_original_ = None

    def _log(self, msg):
        if self.cfg.get('verbose', True):
            print(msg)

    def _adaptive_tree_count(self, rho, layer):
        """M = M_init (L1) or M_min + μ·ρ·(M_max-M_min)"""
        c = self.cfg
        if layer == 1:
            return c['rf_init'], c['et_init']
        M1 = int(c['rf_min'] + c['rf_growth_rate'] * rho
                 * (c['rf_max'] - c['rf_min']))
        M2 = int(c['et_min'] + c['et_growth_rate'] * rho
                 * (c['et_max'] - c['et_min']))
        return max(M1, c['rf_min']), max(M2, c['et_min'])

    def _apply_importance_weighting(self, H, importances):
        """w = (1-λ)·1 + λ·norm(imp);  H_w = H ⊙ w"""
        lam = self.cfg['importance_smooth']
        imp_n = importances / (importances.max() + 1e-12)
        w = (1 - lam) * np.ones_like(imp_n) + lam * imp_n
        return H * w[np.newaxis, :]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AACF':
        """
        Fit the AACF model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix (from ARFE output)
        y : array-like of shape (n_samples,)
            Class labels (integer-encoded)
        """
        c = self.cfg
        self.n_classes_ = len(np.unique(y))
        n_samples, n_feat = X.shape

        self._log(f"\nAACF: {n_feat}-dim, {n_samples} samples, "
                  f"{self.n_classes_} classes")
        self._log(f"  max_layers={c['cascade_max_layers']}, "
                  f"patience={c['cascade_patience']}, "
                  f"ε={c['at_epsilon']}")

        # Normalization
        X_sc = self.scaler_.fit_transform(X)
        self.X_original_ = X_sc.copy()

        # SNR augmentation
        self.snr_aug_ = SNRAugmenter(cfg=c)
        X_aug, y_aug = self.snr_aug_.augment(X_sc, y)
        if c['snr_augment']:
            self._log(f"  SNR augment: {n_samples} -> {X_aug.shape[0]}")

        # Multi-granularity scanning
        self.scanner_ = MultiGranularityScanner(
            cfg=c, feature_groups=self.feature_groups)
        A0_aug = self.scanner_.fit_transform(X_aug, y_aug)
        A0 = A0_aug[:n_samples]

        # Cascade process
        adv = AdversarialTrainer(cfg=c)
        bp = {
            'max_depth': c['tree_max_depth'],
            'min_samples_leaf': c['tree_min_samples_leaf'],
            'max_features': c['tree_max_features'],
            'random_state': c['random_state'],
        }

        prev_rho, prev_proba, prev_imp = 1.0, None, None
        best_acc, no_improve = 0.0, 0
        self.layers_ = []
        self.layer_metrics_ = []
        self.layer_proba_ = []

        self._log(f"  {'L':<4s} {'M1':>5s} {'M2':>5s} {'ρ':>8s} "
                  f"{'Acc':>8s} {'Δ':>8s} {'Hdim':>6s} {'Status'}")
        self._log(f"  {'-'*60}")

        for l in range(1, c['cascade_max_layers'] + 1):
            M1, M2 = self._adaptive_tree_count(prev_rho, l)

            # Input concatenation
            parts = [A0]
            if c['cascade_residual']:
                parts.append(X_sc)
            if prev_proba is not None:
                parts.append(prev_proba)
            H = np.hstack(parts)

            # Importance feedback
            if (c['importance_feedback'] and prev_imp is not None
                    and prev_imp.shape[0] == H.shape[1]):
                H = self._apply_importance_weighting(H, prev_imp)

            # Training
            layer = CascadeLayer(l, M1, M2, bp, adv,
                                 self.n_classes_, c['cascade_cv_folds'])
            proba_l = layer.fit_predict(H, y)

            rho_l = layer.residual_rate_
            acc_l = layer.accuracy_
            d_acc = acc_l - best_acc

            if d_acc >= c['cascade_es_threshold']:
                best_acc = acc_l
                self.best_layer_idx_ = l - 1
                no_improve = 0
                status = "improved"
            else:
                no_improve += 1
                status = f"no_improve({no_improve}/{c['cascade_patience']})"

            self.layers_.append(layer)
            self.layer_metrics_.append({
                'layer': l, 'M1': M1, 'M2': M2,
                'rho': rho_l, 'accuracy': acc_l, 'delta_acc': d_acc,
                'H_dim': H.shape[1],
            })
            self.layer_proba_.append(proba_l)

            self._log(f"  {l:<4d} {M1:>5d} {M2:>5d} {rho_l:>8.4f} "
                      f"{acc_l:>8.4f} {d_acc:>+8.4f} {H.shape[1]:>6d} "
                      f"{status}")

            prev_rho = rho_l
            prev_proba = proba_l
            prev_imp = layer.feature_importances_

            if no_improve >= c['cascade_patience']:
                self._log(f"  Early stop L{l} "
                          f"(best=L{self.best_layer_idx_+1}, "
                          f"acc={best_acc:.4f})")
                break

        self._log(f"  Done: {len(self.layers_)} layers, "
                  f"best=L{self.best_layer_idx_+1} "
                  f"(acc={best_acc:.4f})\n")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        If ensemble_weighted=True, predictions are obtained via
        exponentially decayed weighted fusion across all layers;
        otherwise, only the best layer is used.
        """
        c = self.cfg
        Xs = self.scaler_.transform(X)
        A0 = self.scanner_.transform(Xs)

        all_proba, prev_proba = [], None
        for li, layer in enumerate(self.layers_):
            parts = [A0]
            if c['cascade_residual']:
                parts.append(Xs)
            if prev_proba is not None:
                parts.append(prev_proba)
            prev_proba = layer.predict_proba(np.hstack(parts))
            all_proba.append(prev_proba)

        if c['ensemble_weighted'] and len(all_proba) > 1:
            best = self.best_layer_idx_
            decay = c['ensemble_decay']
            w = np.array([decay ** abs(i - best)
                          for i in range(len(all_proba))])
            w[best + 1:] *= 0.3
            return sum(wi * pi for wi, pi in zip(w, all_proba)) / w.sum()
        return all_proba[self.best_layer_idx_]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return enabled components and improvements"""
        return accuracy_score(y, self.predict(X))

    # ------------------------------------------------------------------
    # Summary / Inspection
    # ------------------------------------------------------------------
    def get_active_improvements(self) -> List[str]:
        """Structured summary (JSON-serializable)"""
        c = self.cfg
        enabled = []
        if c.get('scan_use_groups'):     enabled.append('structured_scan')
        if c.get('cascade_residual'):    enabled.append('residual_input')
        if c.get('snr_augment'):         enabled.append('snr_augment')
        if c.get('at_confusion_focus'):  enabled.append('confusion_focus')
        if c.get('ensemble_weighted'):   enabled.append('weighted_ensemble')
        if c.get('importance_feedback'): enabled.append('importance_feedback')
        return enabled

    def summary(self) -> dict:
        """Print concise summary"""
        return {
            'n_layers': len(self.layers_),
            'best_layer': self.best_layer_idx_ + 1 if self.best_layer_idx_ is not None else None,
            'best_accuracy': (self.layer_metrics_[self.best_layer_idx_]['accuracy']
                              if self.best_layer_idx_ is not None else None),
            'layer_metrics': self.layer_metrics_,
            'active_improvements': self.get_active_improvements(),
            'config': {k: v for k, v in self.cfg.items()
                       if k != 'verbose'},
        }

    def print_summary(self):
        """打印精简摘要"""
        if not self.layers_:
            print("AACF not trained."); return

        print(f"\nAACF: {len(self.layers_)} layers, "
              f"best=L{self.best_layer_idx_+1}")
        print(f"  {'L':<4s} {'M1':>5s} {'M2':>5s} {'ρ':>8s} "
              f"{'Acc':>8s} {'Δ':>8s}")
        print(f"  {'-'*40}")
        for m in self.layer_metrics_:
            mk = " <" if m['layer'] == self.best_layer_idx_ + 1 else ""
            print(f"  {m['layer']:<4d} {m['M1']:>5d} {m['M2']:>5d} "
                  f"{m['rho']:>8.4f} {m['accuracy']:>8.4f} "
                  f"{m['delta_acc']:>+8.4f}{mk}")
        print(f"  Improvements: {self.get_active_improvements()}")