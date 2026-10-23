"""statkit — my statistics library, built from scratch while working through the course.

Each module is filled in during the matching lessons:

    basics.py         Module 00   count, total, minimum, maximum
    mathkit.py        Module 01   summation, logs, derivatives, integrals, vectors and matrices
    descriptive.py    Module 02   centre, spread, position, shape, correlation
    probability.py    Module 03   counting, Bayes, expectation, joint distributions
    distributions.py  Module 04   PMFs, PDFs, CDFs, quantiles, entropy and KL
    inference.py      Module 05   estimators, intervals, bootstrap, tests, power, multiple testing
    regression.py     Module 06   OLS, standard errors, VIF, logistic regression, ridge
    bayes.py          Module 07   conjugate updates, Metropolis, ESS, R-hat, EM
    learning.py       Module 08   splits, metrics, calibration, conformal prediction, PCA
    causal.py         Module 09   experiment design, CUPED, IPW, DiD, reporting helpers

Rules:
- Each module's docstring says which libraries are allowed (early modules: plain Python only).
- A function is done when its tests pass:  make test M=<module>
- Reference solutions live in curriculum/<module>/solutions/ — open them only after your tests pass.
"""
