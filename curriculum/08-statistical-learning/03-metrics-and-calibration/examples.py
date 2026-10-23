"""08.03 Metrics and calibration — confusion matrices, ROC/PR curves, thresholds, proper scores, calibration.

Run:  python curriculum/08-statistical-learning/03-metrics-and-calibration/examples.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)

# An imbalanced problem: 3% positives (think fraud or a rare disease).
X, y = make_classification(n_samples=20_000, n_features=20, n_informative=6, weights=[0.97, 0.03],
                           class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
print(f"Test set: {len(y_te):,} cases, {y_te.sum()} positives ({y_te.mean():.1%})")

logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
p_logit = logit.predict_proba(X_te)[:, 1]

print("\nWorked example 1: accuracy is misleading on imbalanced data")
print(f"  always predict 'negative': accuracy {np.mean(y_te == 0):.4f}, recall 0")
pred = (p_logit >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"  logistic regression at threshold 0.5: TN {tn}, FP {fp}, FN {fn}, TP {tp}")
print(f"  accuracy {(tp + tn) / len(y_te):.4f}  precision {precision_score(y_te, pred):.3f}  "
      f"recall {recall_score(y_te, pred):.3f}  F1 {f1_score(y_te, pred):.3f}")

print("\nWorked example 2: the threshold is a decision, not part of the model")
for thr in [0.5, 0.2, 0.1, 0.05]:
    pr = (p_logit >= thr).astype(int)
    print(f"  threshold {thr:<4}: precision {precision_score(y_te, pr, zero_division=0):.3f}  recall {recall_score(y_te, pr):.3f}"
          f"  flagged {pr.sum():>4} cases")
cost_fn, cost_fp = 50, 1          # missing a case costs 50 times more than a false alarm
thresholds = np.linspace(0.01, 0.9, 90)
costs = [cost_fn * np.sum((p_logit < t) & (y_te == 1)) + cost_fp * np.sum((p_logit >= t) & (y_te == 0)) for t in thresholds]
best_t = thresholds[int(np.argmin(costs))]
print(f"  with costs FN = {cost_fn}, FP = {cost_fp}: best threshold {best_t:.2f} (theory: cost_fp/(cost_fp + cost_fn) = {cost_fp / (cost_fp + cost_fn):.3f})")

print("\nWorked example 3: ROC-AUC vs PR-AUC (average precision)")
nb = GaussianNB().fit(X_tr, y_tr)
rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
probs = {"logistic": p_logit, "naive Bayes": nb.predict_proba(X_te)[:, 1], "random forest": rf.predict_proba(X_te)[:, 1]}
for name, p in probs.items():
    print(f"  {name:<14} ROC-AUC {roc_auc_score(y_te, p):.3f}   PR-AUC {average_precision_score(y_te, p):.3f}"
          f"   (random classifier: ROC 0.5, PR {y_te.mean():.3f})")
auc_rank = np.mean(p_logit[y_te == 1][:, None] > p_logit[y_te == 0][None, :])
print(f"  ROC-AUC = P(random positive scores above random negative) = {auc_rank:.3f}")

print("\nWorked example 4: proper scoring rules and calibration")


def ece(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return total


cal_iso = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(X_tr, y_tr)
cal_sig = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5).fit(X_tr, y_tr)
probs["naive Bayes + isotonic"] = cal_iso.predict_proba(X_te)[:, 1]
probs["naive Bayes + Platt"] = cal_sig.predict_proba(X_te)[:, 1]
for name, p in probs.items():
    print(f"  {name:<24} log loss {log_loss(y_te, p):.4f}  Brier {brier_score_loss(y_te, p):.4f}  ECE {ece(y_te, p):.4f}"
          f"  ROC-AUC {roc_auc_score(y_te, p):.3f}")
print("  -> calibration fixes the probabilities (log loss, Brier, ECE) without changing the ranking much (AUC)")

print("\nWorked example 5: temperature scaling — one parameter for an overconfident model")
logits = logit.decision_function(X_te) * 3          # pretend the model is overconfident: logits 3x too large
logits_val, logits_test = logits[: len(logits) // 2], logits[len(logits) // 2:]
y_val, y_test = y_te[: len(y_te) // 2], y_te[len(y_te) // 2:]
temps = np.linspace(0.5, 6, 111)
nll = [log_loss(y_val, 1 / (1 + np.exp(-logits_val / T))) for T in temps]
T_best = temps[int(np.argmin(nll))]
before = 1 / (1 + np.exp(-logits_test))
after = 1 / (1 + np.exp(-logits_test / T_best))
print(f"  fitted temperature T = {T_best:.2f} (the true over-scaling was 3)")
print(f"  test ECE before {ece(y_test, before):.4f} -> after {ece(y_test, after):.4f}; "
      f"AUC unchanged {roc_auc_score(y_test, before):.3f} -> {roc_auc_score(y_test, after):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name in ["logistic", "naive Bayes", "random forest"]:
    fpr, tpr, _ = roc_curve(y_te, probs[name])
    axes[0].plot(fpr, tpr, label=name)
    prec, rec, _ = precision_recall_curve(y_te, probs[name])
    axes[1].plot(rec, prec, label=name)
axes[0].plot([0, 1], [0, 1], color="grey", ls="--")
axes[0].set_title("ROC curves: all look good")
axes[0].set_xlabel("false positive rate")
axes[0].set_ylabel("true positive rate (recall)")
axes[1].axhline(y_te.mean(), color="grey", ls="--")
axes[1].set_title("Precision–recall: the real difficulty shows")
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
for name in ["logistic", "naive Bayes", "naive Bayes + isotonic"]:
    frac, mean_pred = calibration_curve(y_te, probs[name], n_bins=10, strategy="quantile")
    axes[2].plot(mean_pred, frac, marker="o", label=name)
axes[2].plot([0, 1], [0, 1], color="grey", ls="--")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_title("Reliability diagram (diagonal = calibrated)")
axes[2].set_xlabel("mean predicted probability")
axes[2].set_ylabel("observed frequency")
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "roc_pr_calibration.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "roc_pr_calibration.png")
