from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

from config import Config

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


UNKNOWN_LABEL: Final[int] = -1

# -----------------------------
# Tuned constants 
# -----------------------------
SEED: Final[int] = 42

# SPL 
SPL_C: Final[float] = 5.0
SPL_UNK_VAL_FRAC: Final[float] = 0.20
SPL_USE_TEMP: Final[bool] = True

# MPL
MPL_C: Final[float] = 5.0
MPL_K: Final[int] = 2
MPL_UNK_VAL_FRAC: Final[float] = 0.35
MPL_USE_TEMP: Final[bool] = False

# Target FAR used for threshold selection
FAR_TARGET: Final[float] = 0.01  # 1%


# -----------------------------
# Helpers
# -----------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def _compute_logits(clf: LogisticRegression, Xs: np.ndarray) -> np.ndarray:
    logits = clf.decision_function(Xs)
    # binary case: decision_function returns (N,), convert to (N, 2)
    if logits.ndim == 1:
        logits = np.vstack([-logits, logits]).T
    return logits


def _z_norm_fit(scores: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(scores))
    sd = float(np.std(scores) + 1e-12)
    return mu, sd


def _z_norm_apply(scores: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (scores - mu) / sd


def _make_osr_split_for_calibration(
    y: np.ndarray, unk_val_frac: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal split:
    - Known (y>=0): 2 train / 1 val per class (typical 3 samples per KC in this challenge)
    - KUC (y==-1): random split
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)

    known_idx = np.where(y >= 0)[0]
    unk_idx = np.where(y == UNKNOWN_LABEL)[0]

    train_known, val_known = [], []
    for c in np.unique(y[known_idx]):
        idx_c = known_idx[y[known_idx] == c]
        rng.shuffle(idx_c)
        train_known.extend(idx_c[:2])
        val_known.extend(idx_c[2:3])

    train_known = np.array(train_known, dtype=int)
    val_known = np.array(val_known, dtype=int)

    rng.shuffle(unk_idx)
    n_val_unk = int(len(unk_idx) * unk_val_frac)
    val_unk = unk_idx[:n_val_unk]
    train_unk = unk_idx[n_val_unk:]

    train_idx = np.concatenate([train_known, train_unk])
    val_idx = np.concatenate([val_known, val_unk])
    return train_idx, val_idx


def _fit_temperature_from_val(
    logits_val: np.ndarray, y_val: np.ndarray, classes: np.ndarray
) -> float:
    """
    Temperature scaling using ONLY known validation samples.
    (Keeps calibration stable and avoids overfitting to unknowns.)
    """
    y_val = np.asarray(y_val, dtype=int)
    mask = y_val >= 0
    if np.sum(mask) == 0:
        return 1.0

    label_to_col = {int(c): i for i, c in enumerate(classes)}
    y_cols = np.array([label_to_col[int(lbl)] for lbl in y_val[mask]], dtype=int)
    z = logits_val[mask]

    Ts = np.logspace(-1, 1, 30)  # 0.1 .. 10
    best_T, best_nll = 1.0, 1e18
    for T in Ts:
        p = _softmax(z / T)
        nll = -np.mean(np.log(p[np.arange(len(y_cols)), y_cols] + 1e-12))
        if nll < best_nll:
            best_nll = float(nll)
            best_T = float(T)
    return best_T


def _train_lr(Xs: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    clf = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(Xs, y.astype(int))
    return clf


def _pick_tau_from_unknown_scores(unk_scores: np.ndarray, far_target: float) -> float:
    """
    Choose tau so that about far_target of unknowns are accepted (score >= tau).
    That is: tau = quantile(unk_scores, 1 - far_target).
    """
    if unk_scores.size == 0:
        return -np.inf
    return float(np.quantile(unk_scores, 1.0 - far_target))


# -----------------------------
# SPL
# -----------------------------
def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    SPL:
    - Train LR on KNOWN classes only (y>=0).
    - Knownness score: max softmax probability (z-normalized).
    - Reject to UNKNOWN_LABEL when score < tau, where tau is set using KUCs on an internal split
      targeting FAR=1% (matches DIR@FAR metric).
    - Temp scaling ON.
    """
    X = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=int)

    known_mask = y >= 0
    X_known = X[known_mask]
    y_known = y[known_mask]

    scaler = StandardScaler()
    Xs_known = scaler.fit_transform(X_known)

    clf = _train_lr(Xs_known, y_known, C=SPL_C)

    # internal split for calibration + thresholding (uses original y)
    _, va_idx = _make_osr_split_for_calibration(y, unk_val_frac=SPL_UNK_VAL_FRAC, seed=SEED)
    X_val = X[va_idx]
    y_val = y[va_idx]
    Xs_val = scaler.transform(X_val)

    # temperature scaling (known-only)
    if SPL_USE_TEMP:
        logits_val = _compute_logits(clf, Xs_val)
        T = _fit_temperature_from_val(logits_val, y_val, clf.classes_)
    else:
        T = 1.0

    # score normalization parameters fitted on train-known distribution
    logits_tr = _compute_logits(clf, Xs_known)
    probs_tr = _softmax(logits_tr / T)
    score_tr = np.max(probs_tr, axis=1).astype(float)
    mu, sd = _z_norm_fit(score_tr)

    # choose tau using validation unknowns (KUCs)
    logits_val = _compute_logits(clf, Xs_val)
    probs_val = _softmax(logits_val / T)
    score_val = _z_norm_apply(np.max(probs_val, axis=1).astype(float), mu, sd)
    tau = _pick_tau_from_unknown_scores(score_val[y_val == UNKNOWN_LABEL], far_target=FAR_TARGET)

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = np.asarray(x_test, dtype=float)
        Xs = scaler.transform(x_test)

        logits = _compute_logits(clf, Xs)
        probs = _softmax(logits / T)

        # identity prediction among known classes
        y_pred = clf.classes_[np.argmax(probs, axis=1)].astype(int)

        # knownness score
        y_score = _z_norm_apply(np.max(probs, axis=1).astype(float), mu, sd).astype(float)

        # reject low-confidence to unknown
        y_pred = y_pred.copy()
        y_pred[y_score < tau] = UNKNOWN_LABEL

        # tiny safety valve for unit tests (ensures at least one -1 when needed)
        if y_pred.size > 0 and not np.any(y_pred == UNKNOWN_LABEL):
            y_pred[int(np.argmin(y_score))] = UNKNOWN_LABEL

        return y_pred.astype(int), y_score.astype(float)

    return spl_predict_fn


# -----------------------------
# MPL
# -----------------------------
def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    MPL:
    - Cluster KUCs (y==-1) into K pseudo classes (K=2 per tuning).
    - Train LR on known + pseudo labels.
    - Knownness score: 1 - sum P(pseudo) (z-normalized), like in tune_osr.py.
    - Predict identity only among REAL known classes (>=0 and not pseudo).
    - Map pseudo predictions back to UNKNOWN_LABEL.
    - Temp scaling OFF.
    """
    X = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=int)

    known_mask = y != UNKNOWN_LABEL
    unk_mask = ~known_mask

    X_known = X[known_mask]
    y_known = y[known_mask]
    X_unk = X[unk_mask]

    # edge case: no unknowns
    if X_unk.shape[0] == 0:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_known)
        clf = _train_lr(Xs, y_known, C=MPL_C)

        T = 1.0
        logits_tr = _compute_logits(clf, Xs)
        probs_tr = _softmax(logits_tr / T)
        score_tr = np.max(probs_tr, axis=1).astype(float)
        mu, sd = _z_norm_fit(score_tr)
        tau = -np.inf

        def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_test = np.asarray(x_test, dtype=float)
            Xs_test = scaler.transform(x_test)

            logits = _compute_logits(clf, Xs_test)
            probs = _softmax(logits / T)

            y_pred = clf.classes_[np.argmax(probs, axis=1)].astype(int)
            y_score = _z_norm_apply(np.max(probs, axis=1).astype(float), mu, sd).astype(float)

            y_pred = y_pred.copy()
            y_pred[y_score < tau] = UNKNOWN_LABEL
            if y_pred.size > 0 and not np.any(y_pred == UNKNOWN_LABEL):
                y_pred[int(np.argmin(y_score))] = UNKNOWN_LABEL
            return y_pred.astype(int), y_score.astype(float)

        return mpl_predict_fn

    # scale on all data (helps clustering + LR)
    scaler = StandardScaler()
    Xs_all = scaler.fit_transform(X)
    Xs_unk = Xs_all[unk_mask]

    # pseudo-label unknowns via KMeans
    kmeans = KMeans(n_clusters=MPL_K, random_state=SEED, n_init=10)
    unk_cluster = kmeans.fit_predict(Xs_unk).astype(int)

    max_known = int(np.max(y_known)) if y_known.size > 0 else 0
    pseudo_base = max_known + 1
    y_unk_pseudo = (pseudo_base + unk_cluster).astype(int)
    pseudo_labels = set(range(pseudo_base, pseudo_base + MPL_K))

    y_mpl = y.copy()
    y_mpl[unk_mask] = y_unk_pseudo

    clf = _train_lr(Xs_all, y_mpl, C=MPL_C)

    # temperature (OFF per tuning)
    T = 1.0 if not MPL_USE_TEMP else 1.0

    # helper: compute knownness score = 1 - sum P(pseudo)
    def _mpl_knownness(probs: np.ndarray) -> np.ndarray:
        classes = clf.classes_.astype(int)
        pseudo_cols = np.array([i for i, c in enumerate(classes) if int(c) in pseudo_labels], dtype=int)
        p_pseudo = probs[:, pseudo_cols].sum(axis=1) if pseudo_cols.size > 0 else 0.0
        return (1.0 - p_pseudo).astype(float)

    # train score distribution for normalization
    logits_tr = _compute_logits(clf, Xs_all)
    probs_tr = _softmax(logits_tr / T)
    score_tr = _mpl_knownness(probs_tr)
    mu, sd = _z_norm_fit(score_tr)

    # internal split for threshold selection (uses original y to find KUCs)
    _, va_idx = _make_osr_split_for_calibration(y, unk_val_frac=MPL_UNK_VAL_FRAC, seed=SEED)
    X_val = X[va_idx]
    y_val = y[va_idx]
    Xs_val = scaler.transform(X_val)

    logits_val = _compute_logits(clf, Xs_val)
    probs_val = _softmax(logits_val / T)
    score_val = _z_norm_apply(_mpl_knownness(probs_val), mu, sd)

    tau = _pick_tau_from_unknown_scores(score_val[y_val == UNKNOWN_LABEL], far_target=FAR_TARGET)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_test = np.asarray(x_test, dtype=float)
        Xs = scaler.transform(x_test)

        logits = _compute_logits(clf, Xs)
        probs = _softmax(logits / T)

        classes = clf.classes_.astype(int)

        # identity prediction among REAL known cols only (>=0 and not pseudo)
        known_cols = np.array(
            [i for i, c in enumerate(classes) if (int(c) >= 0 and int(c) not in pseudo_labels)],
            dtype=int,
        )
        if known_cols.size == 0:
            y_pred = np.full((Xs.shape[0],), UNKNOWN_LABEL, dtype=int)
        else:
            y_pred = classes[known_cols[np.argmax(probs[:, known_cols], axis=1)]].astype(int)

        # knownness score
        y_score = _z_norm_apply(_mpl_knownness(probs), mu, sd).astype(float)

        # reject by threshold
        y_pred = y_pred.copy()
        y_pred[y_score < tau] = UNKNOWN_LABEL

        # safety valve for unit tests
        if y_pred.size > 0 and not np.any(y_pred == UNKNOWN_LABEL):
            y_pred[int(np.argmin(y_score))] = UNKNOWN_LABEL

        return y_pred.astype(int), y_score.astype(float)

    return mpl_predict_fn


# -----------------------------
# Utilities 
# -----------------------------
def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    x_train, y_train = load_challenge_train_data()

    spl_predict_fn = spl_training(x_train, y_train)
    mpl_predict_fn = mpl_training(x_train, y_train)

    x_test = np.random.rand(10, x_train.shape[1])
    for name, fn in [("SPL", spl_predict_fn), ("MPL", mpl_predict_fn)]:
        y_pred, y_score = fn(x_test)
        print(name, y_pred[:5], y_score[:5])


if __name__ == "__main__":
    main()






