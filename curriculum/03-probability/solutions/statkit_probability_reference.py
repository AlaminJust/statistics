"""Reference implementation of statkit/probability.py. Compare with yours AFTER your tests pass."""


def simulate_probability(event, trials, rng):
    hits = 0
    for _ in range(trials):
        if event(rng):
            hits += 1
    return hits / trials


def permutations(n, k):
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    return result


def n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def total_probability(conditionals, priors):
    if abs(sum(priors) - 1) > 1e-9:
        raise ValueError("priors must sum to 1")
    return sum(c * p for c, p in zip(conditionals, priors))


def bayes_posterior(prior, sensitivity, specificity):
    true_positive = sensitivity * prior
    false_positive = (1 - specificity) * (1 - prior)
    return true_positive / (true_positive + false_positive)


def expectation(values, probs):
    if abs(sum(probs) - 1) > 1e-9:
        raise ValueError("probabilities must sum to 1")
    return sum(v * p for v, p in zip(values, probs))


def variance_rv(values, probs):
    mu = expectation(values, probs)
    return sum((v - mu) ** 2 * p for v, p in zip(values, probs))


def joint_to_marginals(table):
    rows = [sum(row) for row in table]
    cols = [sum(column) for column in zip(*table)]
    return rows, cols


def covariance_rv(joint, x_values, y_values):
    px, py = joint_to_marginals(joint)
    ex = sum(x * p for x, p in zip(x_values, px))
    ey = sum(y * p for y, p in zip(y_values, py))
    exy = sum(x * y * joint[i][j] for i, x in enumerate(x_values) for j, y in enumerate(y_values))
    return exy - ex * ey
