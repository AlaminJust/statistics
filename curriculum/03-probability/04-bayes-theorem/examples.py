"""03.04 Bayes' theorem — worked examples, natural frequencies and a tiny Naive Bayes spam filter.

Run:  python curriculum/03-probability/04-bayes-theorem/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(4)


def posterior(prior, sensitivity, specificity):
    """P(condition | positive test)."""
    true_pos = sensitivity * prior
    false_pos = (1 - specificity) * (1 - prior)
    return true_pos / (true_pos + false_pos)


print("Worked example 1: medical test (sensitivity 0.99, specificity 0.95, prevalence 0.01)")
p = posterior(0.01, 0.99, 0.95)
print(f"  P(positive) = 0.99*0.01 + 0.05*0.99 = {0.99 * 0.01 + 0.05 * 0.99:.4f}")
print(f"  P(disease | positive) = 0.0099 / 0.0594 = {p:.4f}")

print("\nNatural frequencies: imagine 10,000 people")
n = 10_000
sick = round(n * 0.01)
healthy = n - sick
tp = round(sick * 0.99)
fp = round(healthy * 0.05)
print(f"  {sick} sick -> {tp} test positive;  {healthy} healthy -> {fp} test positive (false alarms)")
print(f"  of {tp + fp} positives, {tp} are sick: {tp / (tp + fp):.3f}")

print("  simulation of 1,000,000 people:")
disease = rng.random(1_000_000) < 0.01
positive = np.where(disease, rng.random(1_000_000) < 0.99, rng.random(1_000_000) < 0.05)
print(f"  P(disease | positive) ≈ {disease[positive].mean():.4f}")

print("\nWorked example 2: a second, independent positive test")
p2 = posterior(p, 0.99, 0.95)
print(f"  prior is now {p:.4f} -> posterior {p2:.4f}")
print(f"  a third positive -> {posterior(p2, 0.99, 0.95):.4f}")

print("\nWorked example 3: the odds form")
prior_odds = 0.01 / 0.99
lr = 0.99 / 0.05
post_odds = prior_odds * lr
print(f"  prior odds {prior_odds:.4f} x likelihood ratio {lr:.1f} = posterior odds {post_odds:.4f}"
      f" -> probability {post_odds / (1 + post_odds):.4f}")

print("\nHow the posterior depends on prevalence")
for prev in [0.001, 0.01, 0.1, 0.3, 0.5]:
    print(f"  prevalence {prev:<6} -> P(disease | positive) = {posterior(prev, 0.99, 0.95):.3f}")
prevalences = np.linspace(0.001, 0.5, 300)
fig, ax = plt.subplots(figsize=(6.5, 4))
for spec in [0.90, 0.95, 0.99]:
    ax.plot(prevalences, posterior(prevalences, 0.99, spec), label=f"specificity {spec}")
ax.set_xscale("log")
ax.set_title("A positive test means little when the condition is rare")
ax.set_xlabel("prevalence (prior), log scale")
ax.set_ylabel("P(condition | positive)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "posterior_vs_prevalence.png", dpi=120)
plt.close(fig)

print("\nWorked example 4: reversing the machines example from 03.03")
p_defect = 0.6 * 0.02 + 0.4 * 0.05
print(f"  P(M2 | defect) = 0.4*0.05 / {p_defect:.3f} = {0.4 * 0.05 / p_defect:.3f}")
print(f"  P(M1 | defect) = 0.6*0.02 / {p_defect:.3f} = {0.6 * 0.02 / p_defect:.3f}")

print("\nWorked example 5: a two-word Naive Bayes spam filter")
prior = {"spam": 0.4, "ham": 0.6}
p_word = {"spam": {"free": 0.50, "meeting": 0.05}, "ham": {"free": 0.05, "meeting": 0.30}}
email = ["free", "meeting"]
scores = {c: prior[c] * math.prod(p_word[c][w] for w in email) for c in prior}
total = sum(scores.values())
for c, s in scores.items():
    print(f"  {c}: prior x P(free|{c}) x P(meeting|{c}) = {s:.4f} -> posterior {s / total:.4f}")
log_scores = {c: math.log(prior[c]) + sum(math.log(p_word[c][w]) for w in email) for c in prior}
print("  the same in log space:", {c: round(v, 4) for c, v in log_scores.items()},
      "-> pick the larger:", max(log_scores, key=log_scores.get))

print("\nWorked example 6: Bertrand's box (03.01 exercise 11) by Bayes")
prior_box = {"GG": 1 / 3, "SS": 1 / 3, "GS": 1 / 3}
p_gold = {"GG": 1.0, "SS": 0.0, "GS": 0.5}
evidence = sum(prior_box[b] * p_gold[b] for b in prior_box)
print(f"  P(GG | drew gold) = (1/3 * 1) / {evidence:.3f} = {prior_box['GG'] * p_gold['GG'] / evidence:.4f}")
print("\nFigure saved:", FIGURES / "posterior_vs_prevalence.png")
