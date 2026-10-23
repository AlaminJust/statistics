"""07.05 Latent variables, EM and variational inference — mixtures, the ELBO, ADVI, and Gaussian processes.

Run:  python curriculum/07-bayesian/05-latent-variables-em-vi/examples.py   (about 1 minute)
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.mixture import GaussianMixture

logging.getLogger("pymc").setLevel(logging.ERROR)
import pymc as pm  # noqa: E402

FIGURES = Path(__file__).parent / "figures" / "generated"
FIGURES.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(75)

print("Worked example 1: a two-group mixture with hidden labels — weekly study hours of two student types")
z_true = rng.random(500) < 0.35                     # 35% are 'intensive' students
x = np.where(z_true, rng.normal(18, 3, 500), rng.normal(8, 2, 500))
print(f"  500 students; the group labels are hidden. Overall mean {x.mean():.2f}, sd {x.std():.2f}")


def em_two_gaussians(x, weights, means, sds, iterations=200, tol=1e-10):
    history = []
    for _ in range(iterations):
        # E-step: responsibility of each component for each point (posterior P(z = k | x))
        dens = np.vstack([w * stats.norm.pdf(x, m, s) for w, m, s in zip(weights, means, sds)])
        resp = dens / dens.sum(axis=0)
        loglik = np.log(dens.sum(axis=0)).sum()
        history.append(loglik)
        # M-step: weighted MLEs
        nk = resp.sum(axis=1)
        weights = nk / len(x)
        means = (resp @ x) / nk
        sds = np.sqrt((resp * (x - means[:, None]) ** 2).sum(axis=1) / nk)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
    return weights, means, sds, resp, history


w, m, s, resp, hist = em_two_gaussians(x, np.array([0.5, 0.5]), np.array([5.0, 25.0]), np.array([5.0, 5.0]))
print(f"  EM converged in {len(hist)} iterations; log-likelihood {hist[0]:.1f} -> {hist[-1]:.1f} (never decreases)")
print(f"  weights {np.round(w, 3)}, means {np.round(m, 2)}, sds {np.round(s, 2)}  (truth: 0.65/0.35, 8/18, 2/3)")
assign = resp.argmax(axis=0) == np.argmax(m)
print(f"  share of students assigned to the correct group: {(assign == z_true).mean():.3f}")
print(f"  a student who studies 13 h: P(intensive) = {resp[:, np.argmin(np.abs(x - 13))][np.argmax(m)]:.2f}")
gm = GaussianMixture(2, random_state=0).fit(x.reshape(-1, 1))
print(f"  sklearn GaussianMixture: weights {np.round(np.sort(gm.weights_), 3)}, means {np.round(np.sort(gm.means_.ravel()), 2)}")

print("\nWorked example 2: EM can get stuck — a bad start")
w2, m2, s2, _, hist2 = em_two_gaussians(x, np.array([0.5, 0.5]), np.array([x.mean(), x.mean()]), np.array([5.0, 5.0]))
print(f"  start with two IDENTICAL means: final means {np.round(m2, 2)}, log-likelihood {hist2[-1]:.1f}"
      f" (good start: {hist[-1]:.1f})")
print("  -> run EM from several starts and keep the best log-likelihood")

print("\nWorked example 3: the ELBO for a tiny model, computed exactly")
# Model: θ ~ N(0, 1), y_i | θ ~ N(θ, 1), 10 observations. True posterior N(Σy/(n+1), 1/(n+1)).
y = rng.normal(1.5, 1, 10)
n = len(y)
post_mean, post_var = y.sum() / (n + 1), 1 / (n + 1)
log_evidence = stats.multivariate_normal(np.zeros(n), np.eye(n) + np.ones((n, n))).logpdf(y)


def elbo(q_mean, q_sd, draws=200_000):
    th = rng.normal(q_mean, q_sd, draws)
    log_joint = stats.norm.logpdf(th, 0, 1) + stats.norm.logpdf(y[:, None], th, 1).sum(axis=0)
    return np.mean(log_joint - stats.norm.logpdf(th, q_mean, q_sd))


for label, qm, qs in [("exact posterior", post_mean, np.sqrt(post_var)), ("wrong mean", post_mean + 0.5, np.sqrt(post_var)),
                      ("too wide", post_mean, 1.0)]:
    e = elbo(qm, qs)
    print(f"  q = {label:<16}: ELBO {e:8.3f}   log evidence {log_evidence:8.3f}   gap = KL(q || posterior) {log_evidence - e:6.3f}")

print("\nWorked example 4: variational inference (ADVI) vs NUTS on logistic regression")
hours = rng.gamma(4, 2.5, 300)
p_true = 1 / (1 + np.exp(-(-5 + 0.4 * hours)))
passed = (rng.random(300) < p_true).astype(int)
with pm.Model() as model:
    b0 = pm.Normal("b0", 0, 5)
    b1 = pm.Normal("b1", 0, 1)
    pm.Bernoulli("passed", logit_p=b0 + b1 * hours, observed=passed)
    nuts = pm.sample(1000, tune=1000, chains=4, random_seed=1, progressbar=False)
    approx = pm.fit(30_000, method="advi", random_seed=1, progressbar=False)
    advi = approx.sample(4000, random_seed=2)
for name, trace in [("NUTS", nuts), ("ADVI", advi)]:
    b1_draws = trace.posterior["b1"].to_numpy().ravel()
    b0_draws = trace.posterior["b0"].to_numpy().ravel()
    print(f"  {name}: b1 mean {b1_draws.mean():.4f}, sd {b1_draws.std():.4f};  corr(b0, b1) {np.corrcoef(b0_draws, b1_draws)[0, 1]:+.3f}")
print("  -> mean-field ADVI treats b0 and b1 as independent: it misses their strong correlation, shifts the mean,"
      " and badly understates the uncertainty")

print("\nWorked example 5: a Gaussian process — a prior over functions, with uncertainty")
xs = np.sort(rng.uniform(0, 10, 25))
ys = np.sin(xs) + 0.3 * xs + rng.normal(0, 0.25, 25)
gp = GaussianProcessRegressor(kernel=RBF(1.0) + WhiteKernel(0.1), normalize_y=True, random_state=0).fit(xs.reshape(-1, 1), ys)
grid = np.linspace(-1, 13, 300).reshape(-1, 1)
mean, sd = gp.predict(grid, return_std=True)
print(f"  learned kernel: {gp.kernel_}")
for xv in [5.0, 12.0]:
    i = np.argmin(np.abs(grid.ravel() - xv))
    print(f"  at x = {xv:>4}: prediction {mean[i]:.2f} ± {1.96 * sd[i]:.2f}  ({'inside' if xv < 10 else 'outside'} the data range)")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].hist(x, bins=40, density=True, alpha=0.4, color="grey")
grid_x = np.linspace(0, 30, 300)
for k in range(2):
    axes[0].plot(grid_x, w[k] * stats.norm.pdf(grid_x, m[k], s[k]), label=f"component {k + 1}")
axes[0].plot(grid_x, sum(w[k] * stats.norm.pdf(grid_x, m[k], s[k]) for k in range(2)), color="black", label="mixture")
axes[0].set_title("EM fit of a two-component mixture")
axes[0].set_xlabel("study hours")
axes[0].legend(fontsize=8)
axes[1].plot(hist, marker="o", ms=3)
axes[1].set_title("EM: the log-likelihood never decreases")
axes[1].set_xlabel("iteration")
axes[2].scatter(xs, ys, color="black", s=12, label="data")
axes[2].plot(grid, mean, label="GP mean")
axes[2].fill_between(grid.ravel(), mean - 1.96 * sd, mean + 1.96 * sd, alpha=0.25, label="95% band")
axes[2].set_title("Gaussian process: uncertainty grows away from data")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "em_and_gp.png", dpi=120)
plt.close(fig)
print("\nFigure saved:", FIGURES / "em_and_gp.png")
