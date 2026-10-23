"""Tests for statkit/inference.py (Module 05). Run: make test M=inference"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from statkit import inference as inf

ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_csv(ROOT / "data" / "students.csv")
SCORES = DF["exam_score"].tolist()
YES = DF.loc[DF["tutoring"] == "yes", "exam_score"].tolist()
NO = DF.loc[DF["tutoring"] == "no", "exam_score"].tolist()


def test_l0501_standard_error():
    assert inf.standard_error(SCORES) == pytest.approx(np.std(SCORES, ddof=1) / np.sqrt(200))


def test_l0502_mles():
    assert inf.mle_bernoulli([1, 1, 0, 1, 0, 1, 1, 0, 1, 1]) == pytest.approx(0.7)
    assert inf.mle_poisson([2, 3, 1, 4]) == pytest.approx(2.5)
    mu, s2 = inf.mle_normal(SCORES)
    assert mu == pytest.approx(np.mean(SCORES))
    assert s2 == pytest.approx(np.var(SCORES, ddof=0))


def test_l0503_intervals():
    lo, hi = inf.t_interval(SCORES)
    assert (lo, hi) == pytest.approx((62.0004906, 65.8595094), abs=1e-5)
    lo90, hi90 = inf.t_interval(SCORES, level=0.90)
    assert hi90 - lo90 < hi - lo
    assert inf.wilson_interval(2, 20) == pytest.approx(proportion_confint(2, 20, method="wilson"))
    assert inf.wilson_interval(45, 50) == pytest.approx(proportion_confint(45, 50, method="wilson"))
    assert inf.wilson_interval(0, 20)[0] == pytest.approx(0, abs=1e-12)


def test_l0504_bootstrap_ci():
    lo, hi = inf.bootstrap_ci(SCORES, np.mean, n_boot=4000, rng=np.random.default_rng(0))
    assert lo == pytest.approx(62.0, abs=0.35)
    assert hi == pytest.approx(65.86, abs=0.35)
    assert lo < np.mean(SCORES) < hi


def test_l0505_permutation_test():
    p = inf.permutation_test(YES, NO, n_perm=5000, rng=np.random.default_rng(0))
    assert p < 0.002
    p_null = inf.permutation_test(SCORES[:100], SCORES[100:], n_perm=3000, rng=np.random.default_rng(0))
    assert p_null > 0.05
    assert p_null == pytest.approx(stats.ttest_ind(SCORES[:100], SCORES[100:]).pvalue, abs=0.05)


def test_l0506_welch():
    t, df, p = inf.welch_t_test(YES, NO)
    ref = stats.ttest_ind(YES, NO, equal_var=False)
    assert t == pytest.approx(ref.statistic)
    assert df == pytest.approx(ref.df)
    assert p == pytest.approx(ref.pvalue)


def test_l0507_power_and_effect():
    assert inf.cohens_d(YES, NO) == pytest.approx(0.6159, abs=1e-3)
    assert inf.sample_size_two_means(0.5, 1.0) == 63
    assert inf.sample_size_two_means(5, 10) == 63
    assert inf.sample_size_two_means(0.2, 1.0) == 393


def test_l0508_multiple_testing():
    p = [0.212, 0.001, 0.041, 0.008, 0.906, 0.012, 0.074, 0.039, 0.205, 0.060]   # deliberately unsorted
    assert inf.bonferroni(p) == multipletests(p, 0.05, "bonferroni")[0].tolist()
    assert inf.benjamini_hochberg(p) == multipletests(p, 0.05, "fdr_bh")[0].tolist()
    assert sum(inf.benjamini_hochberg(p)) == 3
    rng = np.random.default_rng(1)
    random_p = rng.uniform(size=200).tolist()
    assert inf.benjamini_hochberg(random_p, 0.1) == multipletests(random_p, 0.1, "fdr_bh")[0].tolist()
