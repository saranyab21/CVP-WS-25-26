import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from osr_learning import spl_training, mpl_training, UNKNOWN_LABEL
from config import Config


def make_open_set_split(x, y, seed=42, holdout_classes=2, known_test_frac=0.2, unk_test_frac=0.2):
    """
    Robust open-set split:
      - Hold out `holdout_classes` known classes completely from training (become unknown in test)
      - From remaining known classes: put some samples into test (known_test_frac)
      - From unknown (-1): put some into test (unk_test_frac)
    Guarantees test has BOTH known and unknown.
    """
    rng = np.random.RandomState(seed)

    y = np.asarray(y, dtype=int)
    known_classes = np.unique(y[y >= 0])
    rng.shuffle(known_classes)

    # 1) Choose held-out known classes (become unknown at test)
    holdout = set(known_classes[:holdout_classes])
    remaining = set(known_classes[holdout_classes:])

    # indices
    idx_holdout = np.where(np.isin(y, list(holdout)))[0]
    idx_remaining_known = np.where(np.isin(y, list(remaining)))[0]
    idx_unknown = np.where(y == UNKNOWN_LABEL)[0]

    # 2) From remaining known classes, sample some for test
    rng.shuffle(idx_remaining_known)
    n_known_test = max(1, int(len(idx_remaining_known) * known_test_frac))
    idx_known_test = idx_remaining_known[:n_known_test]
    idx_known_train = idx_remaining_known[n_known_test:]

    # 3) From unknown, sample some for test
    rng.shuffle(idx_unknown)
    n_unk_test = max(1, int(len(idx_unknown) * unk_test_frac))
    idx_unk_test = idx_unknown[:n_unk_test]
    idx_unk_train = idx_unknown[n_unk_test:]

    # Training = remaining known train + unknown train
    train_idx = np.concatenate([idx_known_train, idx_unk_train])

    # Test = remaining known test + held-out known (mapped to unknown) + unknown test
    test_idx = np.concatenate([idx_known_test, idx_holdout, idx_unk_test])

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx].copy()

    # Map held-out known classes to UNKNOWN_LABEL in y_test
    y_test[np.isin(y_test, list(holdout))] = UNKNOWN_LABEL

    return x_train, y_train, x_test, y_test, holdout


def eval_osr(predict_fn, x_test, y_test):
    """
    AUC: known vs unknown using score
    DIR@FAR: accept known samples at FAR computed from unknown scores
    """
    y_pred, score = predict_fn(x_test)
    y_test = np.asarray(y_test, dtype=int)
    score = np.asarray(score, dtype=float)

    is_known = (y_test != UNKNOWN_LABEL).astype(int)

    # must contain both 0 and 1
    if len(np.unique(is_known)) < 2:
        return np.nan, {0.01: np.nan, 0.10: np.nan}

    auc = roc_auc_score(is_known, score)

    dir_results = {}
    for far in (0.01, 0.10):
        unk_scores = score[is_known == 0]
        if len(unk_scores) == 0:
            dir_results[far] = np.nan
            continue

        thresh = np.quantile(unk_scores, 1 - far)
        accept = score >= thresh
        dir_val = float(np.mean(accept[is_known == 1]))  # DIR = fraction of known accepted
        dir_results[far] = dir_val

    return float(auc), dir_results


def main():
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1].astype(float)
    y = df[:, -1].astype(int)

    x_tr, y_tr, x_te, y_te, holdout = make_open_set_split(
        x, y,
        seed=42,
        holdout_classes=2,
        known_test_frac=0.2,
        unk_test_frac=0.2,
    )

    print(f"Held-out classes (treated as unknown in test): {sorted(list(holdout))}")
    print(f"Train size: {len(x_tr)} | Test size: {len(x_te)}")
    print(f"Test known: {np.sum(y_te != UNKNOWN_LABEL)} | Test unknown: {np.sum(y_te == UNKNOWN_LABEL)}")

    spl_fn = spl_training(x_tr, y_tr)
    mpl_fn = mpl_training(x_tr, y_tr)

    for name, fn in [("SPL", spl_fn), ("MPL", mpl_fn)]:
        auc, dir_vals = eval_osr(fn, x_te, y_te)
        print(f"\n{name}")
        print(f"AUC: {auc:.4f}")
        print(f"DIR@FAR=1% : {dir_vals[0.01]:.4f}")
        print(f"DIR@FAR=10%: {dir_vals[0.10]:.4f}")


if __name__ == "__main__":
    main()