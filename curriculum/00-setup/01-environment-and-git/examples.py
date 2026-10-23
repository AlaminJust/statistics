"""00.01 Environment and git — runnable examples.

Run:  python curriculum/00-setup/01-environment-and-git/examples.py
"""

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np

print("=" * 60)
print("1. Which Python am I using?")
print("=" * 60)
print("Python version :", platform.python_version())
print("Python path    :", sys.executable)
in_venv = sys.prefix != sys.base_prefix
print("Inside a venv? :", in_venv, "" if in_venv else "  <- run: source .venv/bin/activate")

print()
print("=" * 60)
print("2. Installed package versions (compare with requirements.txt)")
print("=" * 60)
for package in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn", "statsmodels", "pytest"]:
    try:
        print(f"{package:<14} {version(package)}")
    except PackageNotFoundError:
        print(f"{package:<14} NOT INSTALLED  <- run: make setup")

print()
print("=" * 60)
print("3. Reproducible randomness")
print("=" * 60)
rng = np.random.default_rng(seed=7)
print("Seeded dice rolls (same every run)    :", rng.integers(1, 7, size=5))
print("Unseeded dice rolls (change every run):", np.random.default_rng().integers(1, 7, size=5))

# Two generators with the same seed produce identical streams.
a = np.random.default_rng(123).normal(size=3)
b = np.random.default_rng(123).normal(size=3)
print("Same seed gives identical numbers?    :", np.array_equal(a, b))

print()
print("=" * 60)
print("4. Your git history so far")
print("=" * 60)
try:
    log = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True, check=True)
    print(log.stdout or "(no commits yet)")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Not inside a git repository, or git is not installed.")
