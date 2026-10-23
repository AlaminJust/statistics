# Mini-project m01 — Fit a line with gradient descent

**Module 01 · week 2 · about 4 hours** · Branch: `git switch -c project/m01-gradient-descent`

## Goal

Fit the model `exam_score ≈ w · study_hours + b` to `data/students.csv` using **only** what you built in Module 01:
summation, derivatives and gradient descent. No `np.polyfit`, no scikit-learn until the final check.

The data were simulated with a true slope of about 2.2 points per study hour (other variables and random noise also
affect the score), so you can judge whether your answer is sensible.

## Steps

1. **Loss.** Write `mse(w, b, x, y)` = (1/n) Σ (yᵢ − (w xᵢ + b))².
2. **Gradient.** Write `gradient(w, b, x, y)` returning (∂L/∂w, ∂L/∂b) using the formulas from Lesson 01.03,
   worked example 7. Check both partial derivatives against `statkit.mathkit.numerical_derivative` at (w, b) = (1, 1).
3. **Descent.** Start at w = b = 0 and repeat w ← w − η ∂L/∂w, b ← b − η ∂L/∂b. Record the loss at every step.
4. **Learning rates.** Try η = 0.001, 0.005 and 0.009. Plot loss against step for each (use a log scale on the y-axis).
   One of them diverges. Explain why in your report: the size of `study_hours` (up to about 30) makes the loss very
   steep in the w direction.
5. **Standardise.** Replace x with z = (x − x̄) / sₓ, where sₓ is the standard deviation (`x.std()`). Run gradient
   descent on z with η = 0.1 for 100 steps, then convert back: slope = w_z / sₓ, intercept = b_z − w_z x̄ / sₓ.
   Compare how many steps each version needs.
6. **Check.** Compare your (w, b) with the normal-equation solution from Lesson 01.05 and with `np.polyfit(x, y, 1)`.
   They should agree to 3 decimal places.
7. **Report.** Copy `templates/project-report.md` to `report.md` and fill it in. Include the loss-curve figure and a
   scatter plot with your fitted line.

## What "done" looks like

- [ ] `analysis.py` runs from the project root: `python projects/m01-gradient-descent/analysis.py`
- [ ] Your slope is within 0.001 of `np.polyfit`'s answer
- [ ] The report explains the effect of the learning rate and of standardising, with figures
- [ ] Merged into `main` with `git merge --no-ff project/m01-gradient-descent`

## Hints (open only if stuck for more than 30 minutes)

<details>
<summary>Hint 1: step counts</summary>

Unstandardised with η = 0.005, the intercept converges slowly: expect about 20,000 steps. Standardised with η = 0.1,
about 100 steps are enough. That gap is why ML pipelines scale features before training.
</details>

<details>
<summary>Hint 2: the answer to check against</summary>

w ≈ 2.048, b ≈ 43.09, final MSE ≈ 89.58.
</details>
