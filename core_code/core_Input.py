import pickle
import numpy as np
# ---- Fully silent integration of the three modules ----
from Elastic import extract_features_batch, ElasticNetScreener
from core_ARFE import ARFE
from core_AACF import AACF, AACF_CONFIG

from sklearn.model_selection import train_test_split
# ---- Dataset partitioning ----
with open('XXX.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
# XX.pkl is dataset

# ---- Conversion to (signals, labels) format ----
signals_list, labels_list = [], []
mod_types = sorted(set(k[0] for k in data.keys()))
mod_to_idx = {m: i for i, m in enumerate(mod_types)}

for (mod, snr), samples in data.items():
    if snr == 10:  # SNR selection
        iq = samples  # (N, 2, 128) — I/Q
        # Conversion to complex-valued signal: I + jQ
        complex_signals = iq[:, 0, :] + 1j * iq[:, 1, :]
        signals_list.append(complex_signals)
        labels_list.extend([mod_to_idx[mod]] * len(iq))

signals = np.vstack(signals_list)   # (N, 128) complex
y = np.array(labels_list)           # (N,) int

# ① feature extraction (input any signal)
X_52 = extract_features_batch(signals)

# ② Elastic Net pre-selection
screener = ElasticNetScreener(verbose=False)
X_35 = screener.fit_transform(X_52, y)

# ③ ARFE
arfe = ARFE(target_n_features=27, verbose=False)
X_27 = arfe.fit_transform(X_35, y, screener.get_selected_feature_names())

# ④ AACF
cfg = AACF_CONFIG.copy()
cfg['verbose'] = False       # change any hyperparameters based of your need
aacf = AACF(cfg=cfg)
aacf.fit(X_27, y)

# ---- Train/test split at the signal level (to prevent information leakage) ----
signals_train, signals_test, y_train, y_test = train_test_split(
    signals, y, test_size=0.3, stratify=y, random_state=42
)

# ---- Feature extraction (performed separately for training and testing sets) ----
X_train_52 = np.nan_to_num(extract_features_batch(signals_train))
X_test_52  = np.nan_to_num(extract_features_batch(signals_test))

# ---- Elastic Net (Fit on the training set only; apply transformation to the test set) ----
screener = ElasticNetScreener(verbose=False)
X_train_35 = screener.fit_transform(X_train_52, y_train)
X_test_35  = screener.transform(X_test_52)

# ---- ARFE (The same principle applies here) ----
arfe = ARFE(target_n_features=27, verbose=False)
X_train_27 = arfe.fit_transform(X_train_35, y_train,
                                 screener.get_selected_feature_names())
X_test_27  = arfe.transform(X_test_35)

# ---- AACF ----
cfg = AACF_CONFIG.copy()
cfg['verbose'] = True
aacf = AACF(cfg=cfg)
aacf.fit(X_train_27, y_train)

# ---- 评估 ----
train_acc = aacf.score(X_train_27, y_train)
test_acc  = aacf.score(X_test_27, y_test)
print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")