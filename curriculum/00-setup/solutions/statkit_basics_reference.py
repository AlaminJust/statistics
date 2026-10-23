"""Reference implementation of statkit/basics.py. Compare with yours AFTER your tests pass."""


def count(values):
    n = 0
    for _ in values:
        n += 1
    return n


def total(values):
    running = 0
    for value in values:
        running += value
    return running


def minimum(values):
    iterator = iter(values)
    try:
        smallest = next(iterator)
    except StopIteration:
        raise ValueError("minimum() of empty data") from None
    for value in iterator:
        if value < smallest:
            smallest = value
    return smallest


def maximum(values):
    iterator = iter(values)
    try:
        largest = next(iterator)
    except StopIteration:
        raise ValueError("maximum() of empty data") from None
    for value in iterator:
        if value > largest:
            largest = value
    return largest
