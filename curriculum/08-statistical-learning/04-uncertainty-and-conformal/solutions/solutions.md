# 08.04 Solutions

1. k = ⌈(99 + 1) × 0.9⌉ = **90**: the 90th smallest score. For 99% with 50 points, k = ⌈51 × 0.99⌉ = 51 > 50, so q̂ is
   **infinite**: with so few calibration points, the only interval guaranteed to have 99% coverage is the whole real
   line. High coverage levels need many calibration points.
2. **Exchangeability** of calibration and test data (i.i.d. is enough). It fails under distribution shift: a model
   calibrated on one hospital deployed in another, last year's prices used for this year, or time series with trends.
3. (a) aleatoric; (b) epistemic; (c) aleatoric; (d) epistemic (model uncertainty).
4. ```python
   def split_conformal_interval(cal_residuals, alpha):
       s = sorted(abs(r) for r in cal_residuals)
       k = math.ceil((len(s) + 1) * (1 - alpha))
       return math.inf if k > len(s) else s[k - 1]
   ```
   With linear regression on the same split, width **162** against the forest's 204: the linear model is the better model
   here, so its intervals are narrower. Coverage on this split is 0.865. Single-split coverage fluctuates; the guarantee
   is about the average.
5. Average coverage over 50 splits: α = 0.05 → 0.948; 0.1 → 0.898; 0.2 → 0.807; 0.3 → 0.710; 0.5 → 0.499. It tracks
   1 − α closely.
6. At 95%: coverage 0.949, average set size 1.00. Lowest coverage for digits **8** (0.87) and **3** (0.88), the most
   confusable digits, which also get most of the size-2 sets.
7. Coverage only about **9%**. The ensemble's spread (average sd 0.11) measures how much *models* disagree (epistemic
   uncertainty), which is small with 2,200 training points, but it ignores the noise in y itself (sd up to 3), the
   aleatoric part. Ensemble spread is not a prediction interval. Conformal calibration fixes this.
8. Conformal 90% width **25.6** (test coverage 0.80 on this split of only 60 test points); statsmodels' 90% prediction
   interval width **28.8** (coverage 0.87). Similar, because the data really are normal and homoscedastic, which is when
   the parametric interval is valid. They'd differ with skewed or heteroscedastic errors, where conformal stays valid and
   the normal-theory interval doesn't.
9. By exchangeability all (n + 1)! orderings of the scores are equally likely, so the test score's rank is uniform on
   1, …, n + 1. P(s_test ≤ s₍ₖ₎) = P(rank ≤ k) = k/(n + 1) ≥ 1 − α when k ≥ (n + 1)(1 − α). ∎
10. Weighting calibration points by an estimated density ratio restores coverage when the test inputs overlap with the
    calibration inputs (moderate shift). When the test inputs lie *outside* the calibration range, the density ratio is
    infinite or undefined, the method has no relevant calibration points, and no reweighting can help. You need data
    from the new region.
11. Even with logits multiplied by 3 (badly overconfident probabilities), conformal sets achieve about 90% coverage,
    because calibration uses the *ranks* of the scores, not their absolute values. Hard questions (flatter true
    distributions) get large sets; easy ones get single answers. This is why conformal methods are used to make LLM
    answers come with a coverage guarantee.
