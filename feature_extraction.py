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

Pipeline: 52 features → Elastic Net + Domain Prior → 35 retained features
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
    计算信号的各阶混合矩 M_{pq} = E[x^{p-q} * (x*)^q]
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
    基于混合矩计算高阶累积量 (HOC), Leonov-Shiryaev公式展开到六阶
    """
    M = compute_moments(signal)

    C = {}
    # --- 二阶 ---
    C['C20'] = M['M20']
    C['C21'] = M['M21']

    # --- 四阶 ---
    C['C40'] = M['M40'] - 3.0 * M['M20'] ** 2
    C['C41'] = M['M41'] - 3.0 * M['M21'] * M['M20']
    C['C42'] = M['M42'] - np.abs(M['M20']) ** 2 - 2.0 * M['M21'] ** 2

    # --- 六阶 (Leonov-Shiryaev) ---
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
    """构建52维特征名称列表"""
    names = []
    for c in HOC_FEATURES:
        names.append(f"|{c}|")
    for (num, den) in NORMALIZED_FEATURES:
        names.append(f"{num}/{den}")
    for (num, den) in QUADRATIC_FEATURES:
        names.append(f"{num}/{den}^2")
    return names


def extract_52_features(signal: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """从单段复信号中提取52维HOC域特征向量"""
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
    """批量提取52维特征"""
    n_samples = signals.shape[0]
    feature_matrix = np.zeros((n_samples, 52), dtype=np.float64)
    for i in range(n_samples):
        feature_matrix[i] = extract_52_features(signals[i], eps=eps)
    return feature_matrix


# =============================================================================
# Section 3: Elastic Net Feature Screening (52 → 35)
#             Hybrid Filter-Wrapper with Domain Knowledge Prior
# =============================================================================

class ElasticNetScreener:
    """
    Elastic Net 特征初筛器 (混合准则版)

    采用 Hybrid Filter-Wrapper 框架, 融合数据驱动与领域知识两类评分:
      composite = w_en * EN_importance_normalized + w_prior * Prior_normalized

    数据驱动分量:
      Elastic Net OvR 系数绝对值 — 基于L1稀疏性识别与分类无关的冗余特征

    领域知识先验 (基于累积量理论):
      归一化比评分 = disc(num) × stab(den) + 配对修正
      二次比评分   = disc(num) × stab_sq(den) + 跨阶增益 + 同阶惩罚
                     + 奇q惩罚 + q匹配冗余惩罚

    核心理论依据:
    ┌──────────────────────────────────────────────────────────────────┐
    │  分子可区分度 (Numerator Discriminability)                       │
    │  · C40, C42: 峰度相关 → PSK/QAM 判别核心指标                    │
    │  · C62, C63: 高阶精细分类器, 对星座形状高度敏感                   │
    │  · C41, C61: 奇阶(q为奇数)对对称调制(PSK,QAM)渐近为零, 低鉴别力  │
    │                                                                  │
    │  分母稳定性 (Denominator Stability)                              │
    │  · C21 = E[|x|²]: 恒正(信号功率), 最稳定的归一化基准             │
    │  · C20 = E[x²]: 圆对称信号下可趋近零, 稳定性次之                │
    │  · C41, C61: 对称调制下趋近零, 做分母极不稳定; 平方后更甚         │
    │  · C42²: 四阶累积量平方后估计误差倍增, 作为二次比分母可靠性极低   │
    │                                                                  │
    │  阶次信息增益 (Cross-Order Information Gain)                     │
    │  · 跨阶比值(如6阶/2阶)比同阶比值(如4阶/4阶)提供更多互补信息      │
    │  · 四阶同阶比冗余严重(信息被HOC直接特征覆盖) → 强惩罚            │
    │  · 六阶同阶比可通过q值差异仍保留部分信息 → 轻惩罚                │
    │                                                                  │
    │  q值匹配冗余 (Even-q Redundancy)                                │
    │  · 分子分母偶q值相同(如C62/C42², 均q=2)时, 二次比与归一化比      │
    │    高度线性相关 → 信息冗余惩罚                                   │
    └──────────────────────────────────────────────────────────────────┘

    分组配额: HOC=9(全保留), Normalized=10/20, Quadratic=16/23
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
    # 累积量属性参数表
    # 参考: Swami & Sadler, IEEE Trans SP, 2000
    #       Nandi & Azzouz, IEEE Trans Commun, 1998
    # ------------------------------------------------------------------

    # 各累积量作为比值分子时对调制类型的区分能力
    _NUM_DISC = {
        'C20': 0.40, 'C21': 0.55,
        'C40': 0.88, 'C41': 0.22, 'C42': 0.78,
        'C60': 0.48, 'C61': 0.25, 'C62': 0.95, 'C63': 0.82,
    }

    # 各累积量作为归一化比 (Cpq/Crs) 分母的估计稳定性
    _DEN_STAB = {
        'C20': 0.52, 'C21': 0.95, 'C40': 0.85, 'C41': 0.15,
        'C42': 0.60, 'C60': 0.50,
    }

    # 各累积量作为二次比 (Cpq/Crs²) 分母的估计稳定性
    # 平方运算放大低阶估计误差: C42²的方差远高于C42本身
    _DEN_STAB_SQ = {
        'C20': 0.88, 'C21': 0.85, 'C40': 0.52, 'C41': 0.28,
        'C42': 0.05, 'C60': 0.48, 'C61': 0.12,
    }

    # 归一化比特定配对的修正项
    # 物理依据: 6阶累积量(C60,C61)在低SNR下估计方差显著增大,
    # 以低阶(C20,C21)归一化时比值波动剧烈; C42/C20不如C42/C21稳定
    _PAIR_ADJ_NORM = {
        ('C21', 'C20'): +0.20,   # 功率比 — 所有调制方案的通用基础归一化指标
        ('C41', 'C40'): -0.10,   # 同阶(4阶/4阶)冗余 + 奇q分子, 与|C41|直接特征高度重叠
        ('C42', 'C20'): -0.10,   # 冗余: C42/C21更可靠(C21恒正)
        ('C60', 'C20'): -0.18,   # 6阶/2阶: 阶次跨度过大导致有限样本方差估计失控
        ('C60', 'C21'): -0.25,   # 同上; 尽管C21更稳定, 6阶估计方差O(N^-3)仍主导
        ('C60', 'C40'): +0.10,   # 6阶/4阶: 中阶归一化有效抑制高阶估计方差
        ('C61', 'C20'): -0.18,   # C61低鉴别力 + 6阶/2阶高方差
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
            三组保留配额 (HOC, Normalized, Quadratic)
        w_en : float
            Elastic Net 数据驱动分量权重
        w_prior : float
            领域知识先验分量权重
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
        基于累积量理论计算领域知识先验评分

        归一化比 (Cpq/Crs):
            score = disc(num) × stab(den) + pair_adjustment

        二次比 (Cpq/Crs²):
            score = disc(num) × stab_sq(den)
                    + 跨阶信息增益
                    + 同阶冗余惩罚
                    + 奇q分母不稳定惩罚
                    + 偶q匹配冗余惩罚
        """
        scores = np.zeros(52)

        # --- Group A: HOC 全保留, 先验 = 1.0 ---
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

            # (a) 跨阶信息增益: 阶次差越大, 互补信息越多
            gap_bonus = 0.05 * abs(p_n - p_d) / 2.0

            # (b) 同阶冗余惩罚
            #     四阶/四阶: 信息完全被HOC直接特征覆盖 → 强惩罚
            #     六阶/六阶: q值差异仍可提供一定增量信息 → 轻惩罚
            same_order_pen = 0.0
            if p_n == p_d:
                same_order_pen = -0.45 if p_n == 4 else -0.15

            # (c) 奇q分母不稳定: C41(q=1), C61(q=1) 对对称调制趋零
            odd_q_pen = -0.08 if (q_d % 2 == 1 and q_d > 0) else 0.0

            # (d) 偶q匹配冗余: 分子分母偶q且相同(如C62/C42², 均q=2)
            #     二次比与归一化比高度线性相关 → 信息冗余
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
        """OvR Elastic Net 训练 → 52维 importance 向量"""
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
        """Min-Max 归一化到 [0, 1]"""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-15:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _group_quota_select(self, composite: np.ndarray) -> np.ndarray:
        """分组配额选择: 各组内按 composite 降序保留 quota 个"""
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
        训练Elastic Net + 计算先验 → 复合评分 → 分组配额筛选
        """
        n_features = X.shape[1]
        assert n_features == 52, f"Expected 52 features, got {n_features}"

        X_scaled = self.scaler_.fit_transform(X)

        # ① Elastic Net importance (数据驱动)
        en_imp = self._compute_en_importance(X_scaled, y)
        self.feature_importances_ = en_imp

        # ② Domain prior (理论驱动)
        prior = self._compute_domain_prior()
        self.prior_scores_ = prior

        # ③ 各组内独立归一化 → 加权融合
        composite = np.zeros(52)
        for gname, (start, end) in self.GROUP_RANGES.items():
            en_norm = self._normalize_scores(en_imp[start:end])
            pr_norm = self._normalize_scores(prior[start:end])
            composite[start:end] = self.w_en * en_norm + self.w_prior * pr_norm

        self.composite_scores_ = composite

        # ④ 分组配额选择
        self.selected_indices_ = self._group_quota_select(composite)

        # ------ 打印 ------
        self._print_report(en_imp, prior, composite)

        return self

    def _print_report(self, en_imp, prior, composite):
        """打印筛选报告"""
        groups = {
            'A_HOC':        np.arange(0, self.N_HOC),
            'B_Normalized': np.arange(self.N_HOC, self.N_HOC + self.N_NORM),
            'C_Quadratic':  np.arange(self.N_HOC + self.N_NORM,
                                      self.N_HOC + self.N_NORM + self.N_QUAD),
        }
        quotas = dict(zip(groups.keys(), self.group_quota))

        print(f"\n[Hybrid Screening] 52 → {self.n_select} features")
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
        """应用筛选: 52 → 35"""
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
    生成常见调制类型的复基带信号 (用于测试)
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

    # 归一化功率 + AWGN
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
    print("52-dim HOC Feature Extraction + Hybrid Screening (52 → 35)")
    print("=" * 70)

    mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '2FSK', '4FSK']
    n_per_class = 200
    signal_length = 1024
    snr_db = 10.0

    print(f"\n[1] Generating {len(mod_types)} modulation types × "
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