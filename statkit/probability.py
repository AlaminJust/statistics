"""Module 03 — probability, implemented from scratch (plain Python; `rng` arguments are NumPy Generators).

Run: make test M=probability
"""


# ---- 03.01 Sample spaces and axioms ------------------------------------------------------------

def simulate_probability(event, trials, rng):
    """Call event(rng) `trials` times (it runs one experiment and returns True/False); return the share of True."""
    raise NotImplementedError


# ---- 03.02 Counting ----------------------------------------------------------------------------

def permutations(n, k):
    """Ordered selections of k from n: n! / (n - k)!. Return 0 if k > n."""
    raise NotImplementedError


def n_choose_k(n, k):
    """Unordered selections of k from n, without math.comb. Return 0 if k < 0 or k > n."""
    raise NotImplementedError


# ---- 03.03 Conditional probability -------------------------------------------------------------

def total_probability(conditionals, priors):
    """Law of total probability: Σ P(A | B_j) P(B_j). Raise ValueError if the priors don't sum to 1."""
    raise NotImplementedError


# ---- 03.04 Bayes' theorem ----------------------------------------------------------------------

def bayes_posterior(prior, sensitivity, specificity):
    """P(condition | positive test) from prevalence, sensitivity P(+|D) and specificity P(-|not D)."""
    raise NotImplementedError


# ---- 03.05 Random variables and expectation ----------------------------------------------------

def expectation(values, probs):
    """Σ x p(x). Raise ValueError if probs don't sum to 1 (tolerance 1e-9)."""
    raise NotImplementedError


def variance_rv(values, probs):
    """Σ (x - μ)² p(x)."""
    raise NotImplementedError


# ---- 03.06 Joint distributions -----------------------------------------------------------------

def joint_to_marginals(table):
    """table = list of rows of a joint PMF. Return (row marginals, column marginals) as lists."""
    raise NotImplementedError


def covariance_rv(joint, x_values, y_values):
    """E[XY] - E[X]E[Y] for a joint PMF whose rows are indexed by x_values and columns by y_values."""
    raise NotImplementedError
