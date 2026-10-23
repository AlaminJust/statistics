"""04.06 Entropy, cross-entropy and KL divergence — worked examples.

Run:  python curriculum/04-distributions/06-information-theory/examples.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(46)


def entropy(p, base=2):
    return -sum(pi * math.log(pi, base) for pi in p if pi > 0)


def cross_entropy(p, q, base=2):
    return -sum(pi * math.log(qi, base) for pi, qi in zip(p, q) if pi > 0)


def kl(p, q, base=2):
    return sum(pi * math.log(pi / qi, base) for pi, qi in zip(p, q) if pi > 0)


print("Worked example 1: entropy in bits")
for name, p in [("fair coin", [0.5, 0.5]), ("coin with P(H) = 0.9", [0.9, 0.1]), ("certain outcome", [1.0, 0.0]),
                ("fair die", [1 / 6] * 6), ("fair 8-sided die", [1 / 8] * 8), ("loaded die", [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])]:
    print(f"  {name:<22} H = {entropy(p) + 0.0:.3f} bits")   # + 0.0 turns -0.0 into 0.0
print("  in nats (natural log), fair coin:", round(entropy([0.5, 0.5], base=math.e), 4), "= ln 2")

print("\nWorked example 2: cross-entropy and KL — weather in a city")
p = [0.6, 0.3, 0.1]     # true: sunny, cloudy, rainy
q = [1 / 3, 1 / 3, 1 / 3]  # a model that knows nothing
r = [0.5, 0.3, 0.2]     # a better model
for name, model in [("uniform model q", q), ("better model r", r), ("the truth p itself", p)]:
    print(f"  {name:<20} H(p, model) = {cross_entropy(p, model):.4f}   KL(p || model) = {kl(p, model):.4f}")
print(f"  H(p) = {entropy(p):.4f}  -> cross-entropy = entropy + KL")

print("\nKL is not symmetric")
print(f"  KL(p || r) = {kl(p, r):.4f}   KL(r || p) = {kl(r, p):.4f}")

print("\nWorked example 3: binary cross-entropy loss = negative Bernoulli log-likelihood")
y = np.array([1, 0, 1, 1, 0])
p_hat = np.array([0.9, 0.2, 0.6, 0.95, 0.4])
bce = -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
nll = -np.mean(np.log(np.where(y == 1, p_hat, 1 - p_hat)))
print(f"  BCE = {bce:.4f}   mean negative log-likelihood = {nll:.4f}")
print("  per-example losses:", np.round(-np.log(np.where(y == 1, p_hat, 1 - p_hat)), 4))
print(f"  a confident wrong answer (p_hat = 0.01 when y = 1) costs {-math.log(0.01):.2f} nats")

print("\nWorked example 4: KL between two normals")


def kl_normal(m1, s1, m2, s2):
    return math.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


print(f"  KL(N(0,1) || N(1,1)) = {kl_normal(0, 1, 1, 1):.4f} nats (= (Δμ)²/2)")
print(f"  KL(N(0,1) || N(0,2²)) = {kl_normal(0, 1, 0, 2):.4f}   KL(N(0,2²) || N(0,1)) = {kl_normal(0, 2, 0, 1):.4f}")
x = rng.normal(0, 1, 500_000)
mc = np.mean(stats.norm(0, 1).logpdf(x) - stats.norm(1, 1).logpdf(x))
print(f"  Monte Carlo estimate of KL(N(0,1) || N(1,1)) = {mc:.4f}")

print("\nWorked example 5: mutual information of the joint table from Lesson 03.06")
joint = np.array([[0.30, 0.10], [0.20, 0.40]])
px, py = joint.sum(axis=1), joint.sum(axis=0)
mi = sum(joint[i, j] * math.log2(joint[i, j] / (px[i] * py[j])) for i in range(2) for j in range(2))
print(f"  I(X; Y) = {mi:.4f} bits;  H(Y) = {entropy(py):.4f};  H(Y | X) = H(Y) - I = {entropy(py) - mi:.4f}")
indep = np.outer(px, py)
print("  for the independent table with the same marginals, I =",
      round(sum(indep[i, j] * math.log2(indep[i, j] / (px[i] * py[j])) for i in range(2) for j in range(2)), 6))

print("\nWorked example 6: forward vs reverse KL when fitting one normal to two bumps")
grid = np.linspace(-8, 8, 4001)
dx = grid[1] - grid[0]
target = 0.5 * stats.norm(-3, 1).pdf(grid) + 0.5 * stats.norm(3, 1).pdf(grid)
best = {}
for direction in ["forward KL(p||q)", "reverse KL(q||p)"]:
    results = []
    for m in np.linspace(-4, 4, 81):
        for s in np.linspace(0.5, 4.5, 81):
            q_ = stats.norm(m, s).pdf(grid) + 1e-300
            if direction.startswith("forward"):
                val = np.sum(target * np.log((target + 1e-300) / q_)) * dx
            else:
                val = np.sum(q_ * np.log(q_ / (target + 1e-300))) * dx
            results.append((val, m, s))
    best[direction] = min(results)
    print(f"  {direction}: best normal has mean {best[direction][1]:+.1f}, sd {best[direction][2]:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ps = np.linspace(0.001, 0.999, 300)
axes[0].plot(ps, [entropy([p_, 1 - p_]) for p_ in ps])
axes[0].set_title("Entropy of a coin is largest when fair")
axes[0].set_xlabel("P(heads)")
axes[0].set_ylabel("entropy (bits)")
axes[1].plot(grid, target, color="black", label="target: two bumps")
for direction, style in [("forward KL(p||q)", "--"), ("reverse KL(q||p)", ":")]:
    _, m, s = best[direction]
    axes[1].plot(grid, stats.norm(m, s).pdf(grid), ls=style, lw=2, label=f"{direction.split()[0]} fit")
axes[1].set_title("Forward KL covers both bumps; reverse KL picks one")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "entropy_and_kl.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "entropy_and_kl.png")
