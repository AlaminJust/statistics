"""Module 01 — math toolkit, implemented from scratch with plain Python and the `math` module.

Fill these in during Module 01 (the lesson that introduces each function is noted).
Run: make test M=mathkit
"""

import math  # noqa: F401  (you will need it)


# ---- 01.01 Notation and summation --------------------------------------------------------------

def summation(f, start, stop):
    """Σ from i = start to stop (inclusive) of f(i). summation(lambda i: 2*i + 1, 1, 4) -> 24."""
    raise NotImplementedError


def product(values):
    """Π of values. product([1, 2, 3, 4]) -> 24. product([]) -> 1."""
    raise NotImplementedError


# ---- 01.02 Functions, exponents and logarithms -------------------------------------------------

def sigmoid(z):
    """1 / (1 + e^(-z)) for a single number. Must not overflow for z = -1000 or z = 1000."""
    raise NotImplementedError


def log_sum_exp(values):
    """log(Σ e^v) computed stably by subtracting max(values) first. log_sum_exp([1000, 1001, 1002]) ≈ 1002.4076."""
    raise NotImplementedError


# ---- 01.03 Derivatives and optimisation --------------------------------------------------------

def numerical_derivative(f, x, h=1e-5):
    """Central-difference estimate of f'(x): (f(x + h) - f(x - h)) / (2h)."""
    raise NotImplementedError


def gradient_descent(grad, x0, learning_rate, steps):
    """Repeat x <- x - learning_rate * grad(x) `steps` times from x0; return the final x."""
    raise NotImplementedError


# ---- 01.04 Integrals and area ------------------------------------------------------------------

def integrate(f, a, b, n=1000):
    """Trapezoid-rule estimate of ∫_a^b f(x) dx using n strips."""
    raise NotImplementedError


# ---- 01.05 Vectors and matrices (vectors = lists, matrices = lists of rows) --------------------

def dot(u, v):
    """Σ u_i v_i. Raise ValueError if the lengths differ."""
    raise NotImplementedError


def norm(u):
    """Euclidean length √(u · u)."""
    raise NotImplementedError


def cosine_similarity(u, v):
    """(u · v) / (‖u‖ ‖v‖). Raise ValueError if either vector has length 0."""
    raise NotImplementedError


def transpose(A):
    """Swap rows and columns. transpose([[1, 2, 3], [4, 5, 6]]) -> [[1, 4], [2, 5], [3, 6]]."""
    raise NotImplementedError


def matmul(A, B):
    """Matrix product A·B as a list of rows. Raise ValueError if columns of A != rows of B."""
    raise NotImplementedError
