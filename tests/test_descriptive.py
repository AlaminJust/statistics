"""Tests for statkit/descriptive.py (Module 02). Run: make test M=descriptive

Each test is named after the lesson that introduces the function, so you can run one lesson at a time:
    .venv/bin/python -m pytest -q tests/test_descriptive.py -k l0203
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statkit import descriptive as ds

ROOT = Path(__file__).resolve().parent.parent
STUDENTS = pd.read_csv(ROOT / "data" / "students.csv")
SCORES = STUDENTS["exam_score"].tolist()
HOURS = STUDENTS["study_hours"].tolist()
D = [2, 4, 4, 5, 7, 9, 10, 12, 15, 30]          # the example data from Lesson 02.04
SALARIES = [32, 35, 38, 40, 41, 45, 250]         # the example data from Lesson 02.02


# ---- 02.01 -------------------------------------------------------------------------------------
def test_l0201_frequency_table():
    assert ds.frequency_table(["a", "b", "a"]) == {"a": 2, "b": 1}
    assert ds.frequency_table(STUDENTS["program"].tolist()) == {"Commerce": 49, "Arts": 72, "Science": 79}


def test_l0201_relative_frequencies():
    rel = ds.relative_frequencies(STUDENTS["program"].tolist())
    assert rel["Science"] == pytest.approx(0.395)
    assert sum(rel.values()) == pytest.approx(1.0)


# ---- 02.02 -------------------------------------------------------------------------------------
def test_l0202_mean_median():
    assert ds.mean(SALARIES) == pytest.approx(68.71428571)
    assert ds.median(SALARIES) == 40
    assert ds.median([12, 7, 3, 14, 9, 10]) == 9.5        # even n, unsorted input
    assert ds.mean(SCORES) == pytest.approx(np.mean(SCORES))
    assert ds.median(HOURS) == pytest.approx(np.median(HOURS))
    with pytest.raises(ValueError):
        ds.mean([])


def test_l0202_mode():
    assert ds.mode([2, 3, 3, 5, 7, 7, 7, 9]) == [7]
    assert ds.mode([1, 2, 2, 3, 3]) == [2, 3]
    assert ds.mode(["Science", "Arts", "Science"]) == ["Science"]


def test_l0202_weighted_trimmed_geometric():
    assert ds.weighted_mean([80, 70, 95], [3, 4, 1]) == pytest.approx(76.875)
    assert ds.trimmed_mean(SALARIES, 0.2) == pytest.approx(39.8)
    assert ds.trimmed_mean(HOURS, 0.1) == pytest.approx(stats.trim_mean(HOURS, 0.1))
    assert ds.geometric_mean([1.5, 0.5]) == pytest.approx(0.8660254)
    with pytest.raises(ValueError):
        ds.geometric_mean([1, 0, 2])


# ---- 02.03 -------------------------------------------------------------------------------------
def test_l0203_variance_std():
    x = [4, 7, 1, 8]
    assert ds.variance(x) == pytest.approx(10)
    assert ds.variance(x, ddof=0) == pytest.approx(7.5)
    assert ds.std([70, 72, 75, 78, 80]) == pytest.approx(4.1231056)
    assert ds.std(SCORES) == pytest.approx(np.std(SCORES, ddof=1))
    with pytest.raises(ValueError):
        ds.variance([5])


def test_l0203_range_mad_cv():
    assert ds.data_range([70, 72, 75, 78, 80]) == 10
    assert ds.median_absolute_deviation([4, 7, 1, 8]) == pytest.approx(2.0)
    assert ds.median_absolute_deviation(HOURS) == pytest.approx(stats.median_abs_deviation(HOURS))
    assert ds.coefficient_of_variation(HOURS) == pytest.approx(np.std(HOURS, ddof=1) / np.mean(HOURS))


# ---- 02.04 -------------------------------------------------------------------------------------
def test_l0204_percentiles():
    assert ds.percentile(D, 25) == pytest.approx(4.25)
    assert ds.percentile(D, 50) == pytest.approx(8)
    assert ds.percentile(D, 90) == pytest.approx(16.5)
    assert ds.percentile(D, 0) == 2 and ds.percentile(D, 100) == 30
    for p in [5, 33, 67, 99]:
        assert ds.percentile(HOURS, p) == pytest.approx(np.percentile(HOURS, p))


def test_l0204_summaries():
    assert ds.quartiles(D) == pytest.approx((4.25, 8, 11.5))
    assert ds.iqr(D) == pytest.approx(7.25)
    assert ds.five_number_summary(D) == pytest.approx((2, 4.25, 8, 11.5, 30))
    assert ds.iqr_outliers(D) == [30]
    commute = STUDENTS["commute_minutes"].tolist()
    assert len(ds.iqr_outliers(commute)) == 11


def test_l0204_zscores_skewness():
    z = ds.zscores(D)
    assert np.mean(z) == pytest.approx(0, abs=1e-12)
    assert np.std(z, ddof=1) == pytest.approx(1)
    assert z[-1] == pytest.approx(2.4744, abs=1e-3)
    assert ds.skewness(D) == pytest.approx(stats.skew(D))
    assert ds.skewness(HOURS) == pytest.approx(stats.skew(HOURS))
    assert ds.skewness([1, 2, 3]) == pytest.approx(0)


# ---- 02.05 -------------------------------------------------------------------------------------
def test_l0205_ecdf():
    xs, ps = ds.ecdf([3, 1, 4, 1, 5])
    assert list(xs) == [1, 1, 3, 4, 5]
    assert list(ps) == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_l0205_histogram_counts():
    counts, _ = np.histogram(HOURS, bins=10, range=(0, 30))
    assert list(ds.histogram_counts(HOURS, 10, 0, 30)) == counts.tolist()
    assert list(ds.histogram_counts([0, 1, 2, 3, 4], 2, 0, 4)) == [2, 3]      # 4 lands in the last bin


# ---- 02.06 -------------------------------------------------------------------------------------
def test_l0206_covariance_correlation():
    x, y = [2, 4, 6, 8, 10], [50, 60, 65, 80, 85]
    assert ds.covariance(x, y) == pytest.approx(45)
    assert ds.correlation(x, y) == pytest.approx(0.98787834)
    assert ds.correlation(HOURS, SCORES) == pytest.approx(np.corrcoef(HOURS, SCORES)[0, 1])
    with pytest.raises(ValueError):
        ds.covariance([1, 2], [1, 2, 3])


def test_l0206_ranks_spearman():
    assert list(ds.ranks([10, 20, 20, 40])) == pytest.approx([1, 2.5, 2.5, 4])
    assert list(ds.ranks([3, 1, 2])) == pytest.approx([3, 1, 2])
    assert ds.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 100]) == pytest.approx(1.0)
    assert ds.spearman(HOURS, SCORES) == pytest.approx(stats.spearmanr(HOURS, SCORES).statistic)
