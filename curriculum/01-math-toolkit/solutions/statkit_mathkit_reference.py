"""Reference implementation of statkit/mathkit.py. Compare with yours AFTER your tests pass."""

import math


def summation(f, start, stop):
    total = 0
    for i in range(start, stop + 1):
        total += f(i)
    return total


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def sigmoid(z):
    # Two algebraically equal forms; each one is safe on one side of zero.
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def log_sum_exp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_descent(grad, x0, learning_rate, steps):
    x = x0
    for _ in range(steps):
        x = x - learning_rate * grad(x)
    return x


def integrate(f, a, b, n=1000):
    dx = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * dx)
    return total * dx


def dot(u, v):
    if len(u) != len(v):
        raise ValueError(f"vectors have different lengths: {len(u)} and {len(v)}")
    return sum(a * b for a, b in zip(u, v))


def norm(u):
    return math.sqrt(dot(u, u))


def cosine_similarity(u, v):
    nu, nv = norm(u), norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot(u, v) / (nu * nv)


def transpose(A):
    return [list(column) for column in zip(*A)]


def matmul(A, B):
    if len(A[0]) != len(B):
        raise ValueError(f"shapes don't match: A has {len(A[0])} columns, B has {len(B)} rows")
    columns = transpose(B)
    return [[dot(row, column) for column in columns] for row in A]
