# cvproj_exc/tune_osr.py
import argparse
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UNKNOWN = -1


# -----------------------------
# Data loading + split
# -----------------------------
def load_data(path: str):
    df = pd.read_csv(path, header=None).values
    X = df[:, :-1].astype(float)
    y = df[:, -1].astype(int)
    return X, y


def make_osr_split(X, y, unk_val_frac=0.2, seed=42):
    """
    Known classes (y>=0): usually 3 samples/class -> 2 train, 1 val
    Unknown (y=-1): random split
    """
    rng = np.random.RandomState(seed)

    known_idx = np.where(y >= 0)[0]
    unk_idx = np.where(y == UNKNOWN)[0]

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


# -----------------------------
# Calibration helpers
# -----------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def fit_temperature_known_only(logits_val: np.ndarray, y_val: np.ndarray, classes: np.ndarray) -> float:
    """
    Fit temperature T by minimizing NLL on KNOWN validation samples only (y_val >= 0),
    and only over KNOWN columns (exclude unknown/pseudo).
    """
    known_mask = (y_val >= 0)
    if not np.any(known_mask):
        return 1.0

    known_cols = np.where(classes >= 0)[0]
    if known_cols.size == 0:
        return 1.0

    label_to_col = {int(c): i for i, c in enumerate(classes)}
    y_cols = np.array([label_to_col.get(int(lbl), -1) for lbl in y_val], dtype=int)

    mask = known_mask & (y_cols >= 0)
    if not np.any(mask):
        return 1.0

    z = logits_val[mask][:, known_cols]
    col_to_knownpos = {int(col): j for j, col in enumerate(known_cols)}
    yk = np.array([col_to_knownpos[int(c)] for c in y_cols[mask]], dtype=int)

    Ts = np.logspace(-1, 1, 30)  # 0.1 .. 10
    best_T, best_nll = 1.0, 1e18
    for T in Ts:
        p = softmax(z / T)
        nll = -np.mean(np.log(p[np.arange(len(yk)), yk] + 1e-12))
        if nll < best_nll:
            best_nll = float(nll)
            best_T = float(T)
    return best_T


def normalize_scores(train_scores: np.ndarray, scores: np.ndarray, mode="z") -> np.ndarray:
    if mode == "z":
        mu = float(np.mean(train_scores))
        sd = float(np.std(train_scores) + 1e-12)
        return (scores - mu) / sd
    if mode == "minmax":
        mn = float(np.min(train_scores))
        mx = float(np.max(train_scores))
        return (scores - mn) / (mx - mn + 1e-12)
    return scores


# -----------------------------
# OSR evaluation (AUC + DIR@FAR)
# -----------------------------
def dir_far_curve(y_true, y_pred, score, thresholds):
    known = (y_true >= 0)
    unk = ~known

    fars, dirs = [], []
    for tau in thresholds:
        accept = score >= tau

        fa = np.sum(unk & accept & (y_pred != UNKNOWN))
        far = fa / max(1, np.sum(unk))

        correct_accept = np.sum(known & accept & (y_pred == y_true))
        dirr = correct_accept / max(1, np.sum(known))

        fars.append(far)
        dirs.append(dirr)

    return np.array(fars), np.array(dirs)


def pick_tau_for_far(fars, dirs, thresholds, target_far):
    ok = fars <= target_far
    if not np.any(ok):
        return None, None, None
    i = np.where(ok)[0][np.argmax(dirs[ok])]
    return float(thresholds[i]), float(fars[i]), float(dirs[i])


# -----------------------------
# Core: logits -> probs -> knownness + identity
# -----------------------------
def logits_from_pipeline(model, X: np.ndarray) -> np.ndarray:
    scaler = model[0]
    clf = model[1]
    Xs = scaler.transform(X)
    logits = clf.decision_function(Xs)
    if logits.ndim == 1:
        logits = np.vstack([-logits, logits]).T
    return logits


def predict_spl_knownness(model, X: np.ndarray, T: float):
    """
    SPL: unknown is explicit label -1 in classes_.
    Knownness score = 1 - P(unknown)
    Identity pred = argmax over KNOWN classes only
    """
    clf = model[1]
    classes = clf.classes_.astype(int)

    logits = logits_from_pipeline(model, X)
    probs = softmax(logits / T)

    known_cols = np.where(classes >= 0)[0]
    if known_cols.size == 0:
        pred = np.full(X.shape[0], UNKNOWN, dtype=int)
        score = np.zeros(X.shape[0], dtype=float)
        return pred, score, classes

    unk_cols = np.where(classes == UNKNOWN)[0]
    p_unk = probs[:, unk_cols[0]] if unk_cols.size > 0 else np.zeros(X.shape[0], dtype=float)

    score_known = 1.0 - p_unk
    pred_known = classes[known_cols[np.argmax(probs[:, known_cols], axis=1)]].astype(int)
    return pred_known, score_known.astype(float), classes


def predict_mpl_knownness(model, pseudo_labels: set[int], X: np.ndarray, T: float):
    """
    MPL: unknown = any pseudo label.
    Knownness score = 1 - sum P(pseudo)
    Identity pred = argmax over true KCs
    """
    clf = model[1]
    classes = clf.classes_.astype(int)

    logits = logits_from_pipeline(model, X)
    probs = softmax(logits / T)

    pseudo_cols = np.array([i for i, c in enumerate(classes) if int(c) in pseudo_labels], dtype=int)
    p_unk = probs[:, pseudo_cols].sum(axis=1) if pseudo_cols.size > 0 else np.zeros(X.shape[0], dtype=float)
    score_known = 1.0 - p_unk

    known_cols = np.array([i for i, c in enumerate(classes) if (int(c) >= 0 and int(c) not in pseudo_labels)], dtype=int)
    if known_cols.size == 0:
        pred = np.full(X.shape[0], UNKNOWN, dtype=int)
        return pred, score_known.astype(float), classes

    pred_known = classes[known_cols[np.argmax(probs[:, known_cols], axis=1)]].astype(int)
    return pred_known, score_known.astype(float), classes


# -----------------------------
# Training
# -----------------------------
def train_spl(Xtr, ytr, C=1.0):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, solver="lbfgs"))
    t0 = time.time()
    model.fit(Xtr, ytr)
    return model, (time.time() - t0)


def train_mpl(Xtr, ytr, K=2, C=1.0, seed=42):
    known_mask = (ytr != UNKNOWN)
    unk_mask = ~known_mask

    X_known = Xtr[known_mask]
    y_known = ytr[known_mask]
    X_unk = Xtr[unk_mask]

    if X_unk.shape[0] == 0:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, solver="lbfgs"))
        t0 = time.time()
        model.fit(X_known, y_known)
        return model, (time.time() - t0), set()

    scaler_tmp = StandardScaler()
    X_unk_s = scaler_tmp.fit_transform(X_unk)

    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    unk_cluster = kmeans.fit_predict(X_unk_s)

    max_known = int(np.max(y_known)) if y_known.size > 0 else 0
    pseudo_base = max_known + 1
    y_unk_pseudo = (pseudo_base + unk_cluster).astype(int)
    pseudo_labels = set(range(pseudo_base, pseudo_base + K))

    X_all = np.vstack([X_known, X_unk])
    y_all = np.concatenate([y_known, y_unk_pseudo]).astype(int)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, solver="lbfgs"))
    t0 = time.time()
    model.fit(X_all, y_all)
    return model, (time.time() - t0), pseudo_labels


# -----------------------------
# Evaluation
# -----------------------------
def _rank_score(auc, dir1, dir10):
    d1 = 0.0 if dir1 is None else float(dir1)
    d10 = 0.0 if dir10 is None else float(dir10)
    return 0.5 * float(auc) + 0.3 * d1 + 0.2 * d10


def evaluate_spl(model, Xtr, ytr, Xva, yva, score_norm="z", use_temp=True, threshold_mode="far1"):
    classes = model[1].classes_.astype(int)
    logits_va = logits_from_pipeline(model, Xva)
    T = fit_temperature_known_only(logits_va, yva, classes) if use_temp else 1.0

    pred_tr, score_tr, _ = predict_spl_knownness(model, Xtr, T=T)
    pred_va, score_va, _ = predict_spl_knownness(model, Xva, T=T)

    score_va_n = normalize_scores(score_tr, score_va, mode=score_norm)

    is_known = (yva >= 0).astype(int)
    auc = float(roc_auc_score(is_known, score_va_n))

    thresholds = np.quantile(score_va_n, np.linspace(0.0, 1.0, 400))
    fars, dirs = dir_far_curve(yva, pred_va, score_va_n, thresholds)
    tau1, far1, dir1 = pick_tau_for_far(fars, dirs, thresholds, 0.01)
    tau10, far10, dir10 = pick_tau_for_far(fars, dirs, thresholds, 0.10)

    if threshold_mode == "blend" and (tau1 is not None) and (tau10 is not None):
        tau = 0.7 * tau1 + 0.3 * tau10
    else:
        tau = tau1

    return {
        "auc": auc,
        "T": float(T),
        "tau": tau,
        "dir1": dir1,
        "dir10": dir10,
        "rank": _rank_score(auc, dir1, dir10),
    }


def evaluate_mpl(model, pseudo_labels, Xtr, ytr, Xva, yva, score_norm="z", use_temp=True, threshold_mode="far1"):
    classes = model[1].classes_.astype(int)
    logits_va = logits_from_pipeline(model, Xva)
    T = fit_temperature_known_only(logits_va, yva, classes) if use_temp else 1.0

    pred_tr, score_tr, _ = predict_mpl_knownness(model, pseudo_labels, Xtr, T=T)
    pred_va, score_va, _ = predict_mpl_knownness(model, pseudo_labels, Xva, T=T)

    score_va_n = normalize_scores(score_tr, score_va, mode=score_norm)

    is_known = (yva >= 0).astype(int)
    auc = float(roc_auc_score(is_known, score_va_n))

    thresholds = np.quantile(score_va_n, np.linspace(0.0, 1.0, 400))
    fars, dirs = dir_far_curve(yva, pred_va, score_va_n, thresholds)
    tau1, far1, dir1 = pick_tau_for_far(fars, dirs, thresholds, 0.01)
    tau10, far10, dir10 = pick_tau_for_far(fars, dirs, thresholds, 0.10)

    if threshold_mode == "blend" and (tau1 is not None) and (tau10 is not None):
        tau = 0.7 * tau1 + 0.3 * tau10
    else:
        tau = tau1

    return {
        "auc": auc,
        "T": float(T),
        "tau": tau,
        "dir1": dir1,
        "dir10": dir10,
        "rank": _rank_score(auc, dir1, dir10),
    }


# -----------------------------
# Main: grid search
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    default_csv = repo_root / "data" / "challenge_train_data.csv"

    parser.add_argument("--csv", type=str, default=str(default_csv))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_norm", type=str, default="z", choices=["z", "minmax", "none"])
    parser.add_argument("--print_top", type=int, default=10)
    args = parser.parse_args()

    X, y = load_data(args.csv)

    C_grid = [1.0, 3.0, 5.0]
    unk_val_frac_grid = [0.20, 0.35]
    temp_grid = [True, False]
    thresh_mode_grid = ["far1", "blend"]
    K_grid = [2, 4, 6]

    spl_rows = []
    mpl_rows = []

    for C, unk_val_frac, use_temp, thresh_mode in product(C_grid, unk_val_frac_grid, temp_grid, thresh_mode_grid):
        train_idx, val_idx = make_osr_split(X, y, unk_val_frac=unk_val_frac, seed=args.seed)
        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[val_idx], y[val_idx]

        # SPL
        spl_model, spl_fit = train_spl(Xtr, ytr, C=C)
        spl_m = evaluate_spl(spl_model, Xtr, ytr, Xva, yva, score_norm=args.score_norm, use_temp=use_temp, threshold_mode=thresh_mode)
        spl_rows.append((spl_m["rank"], C, unk_val_frac, use_temp, thresh_mode, spl_m["auc"], spl_m["dir1"], spl_m["dir10"], spl_m["T"], spl_m["tau"]))

        # MPL (loop K separately so it ranks properly)
        for K in K_grid:
            mpl_model, mpl_fit, pseudo = train_mpl(Xtr, ytr, K=K, C=C, seed=args.seed)
            mpl_m = evaluate_mpl(mpl_model, pseudo, Xtr, ytr, Xva, yva, score_norm=args.score_norm, use_temp=use_temp, threshold_mode=thresh_mode)
            mpl_rows.append((mpl_m["rank"], C, K, unk_val_frac, use_temp, thresh_mode, mpl_m["auc"], mpl_m["dir1"], mpl_m["dir10"], mpl_m["T"], mpl_m["tau"]))

    spl_rows.sort(key=lambda r: r[0], reverse=True)
    mpl_rows.sort(key=lambda r: r[0], reverse=True)

    print("\n==================== SPL top configs ====================")
    print("rankScore |   C | unk_val_frac | temp | thresh |   AUC  | DIR@1% | DIR@10% |   T   | tau")
    for row in spl_rows[: args.print_top]:
        rank, C, uvf, temp, tm, auc, d1, d10, T, tau = row
        print(f"{rank:8.4f} | {C:3.1f} |    {uvf:5.2f}    | {str(temp):4s} | {tm:5s} | {auc:6.4f} | {float(d1 or 0):6.4f} | {float(d10 or 0):7.4f} | {T:5.2f} | {tau}")

    print("\n==================== MPL top configs ====================")
    print("rankScore |   C | K | unk_val_frac | temp | thresh |   AUC  | DIR@1% | DIR@10% |   T   | tau")
    for row in mpl_rows[: args.print_top]:
        rank, C, K, uvf, temp, tm, auc, d1, d10, T, tau = row
        print(f"{rank:8.4f} | {C:3.1f} | {K:1d} |    {uvf:5.2f}    | {str(temp):4s} | {tm:5s} | {auc:6.4f} | {float(d1 or 0):6.4f} | {float(d10 or 0):7.4f} | {T:5.2f} | {tau}")

    print("\n Take the best SPL and best MPL params and copy them into osr_learning.py constants.")


if __name__ == "__main__":
    main()

