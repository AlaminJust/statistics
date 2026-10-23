"""Tests for statkit/regression.py (Module 06). Run: make test M=regression"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statkit import regression as reg

ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_csv(ROOT / "data" / "students.csv")
X = np.column_stack([np.ones(len(DF)), DF["study_hours"], DF["sleep_hours"], (DF["tutoring"] == "yes").astype(float)])
Y = DF["exam_score"].to_numpy(dtype=float)
REF = sm.OLS(Y, X).fit()


def test_l0601_five_points():
    Xs = np.column_stack([np.ones(5), [2, 4, 6, 8, 10]])
    ys = np.array([50, 60, 65, 80, 85.0])
    beta = reg.fit_ols(Xs, ys)
    assert beta == pytest.approx([41.0, 4.5])
    assert reg.predict(Xs, beta) == pytest.approx([50, 59, 68, 77, 86])
    assert reg.residuals(Xs, ys, beta) == pytest.approx([0, 1, -3, 3, -1], abs=1e-9)
    assert reg.r_squared(Xs, ys, beta) == pytest.approx(1 - 20 / 830)


def test_l0602_multiple_regression():
    beta = reg.fit_ols(X, Y)
    assert beta == pytest.approx(REF.params)
    assert reg.r_squared(X, Y, beta) == pytest.approx(REF.rsquared)
    assert reg.ols_standard_errors(X, Y) == pytest.approx(REF.bse)


def test_l0603_vif():
    d = load_diabetes(scaled=False, as_frame=True).frame.drop(columns="target")
    ours = reg.vif(d.to_numpy())
    Xc = sm.add_constant(d).to_numpy()
    theirs = [variance_inflation_factor(Xc, i) for i in range(1, Xc.shape[1])]
    assert ours == pytest.approx(theirs, rel=1e-6)


def test_l0604_logistic_gd():
    high = (DF["exam_score"] >= 70).astype(float).to_numpy()
    z = (DF["study_hours"] - DF["study_hours"].mean()) / DF["study_hours"].std()
    Xl = np.column_stack([np.ones(len(DF)), z])
    beta = reg.fit_logistic_gd(Xl, high, learning_rate=0.5, steps=20_000)
    ref = sm.Logit(high, Xl).fit(disp=0).params
    assert beta == pytest.approx(ref, abs=1e-3)


def test_l0605_ridge():
    Xd, yd = load_diabetes(return_X_y=True, scaled=False)
    Xs = StandardScaler().fit_transform(Xd)
    Xi = np.column_stack([np.ones(len(yd)), Xs])
    for lam in [1.0, 25.0, 300.0]:
        ours = reg.fit_ridge(Xi, yd, lam)
        sk = Ridge(alpha=lam).fit(Xs, yd)
        assert ours[0] == pytest.approx(sk.intercept_)
        assert ours[1:] == pytest.approx(sk.coef_)
    assert reg.fit_ridge(Xi, yd, 0.0) == pytest.approx(reg.fit_ols(Xi, yd))
