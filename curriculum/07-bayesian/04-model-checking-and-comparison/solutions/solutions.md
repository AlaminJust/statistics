# 07.04 Solutions

1. Diagnostics check that the sampler has explored the posterior *of the model you wrote*. If the model is wrong (wrong
   distribution, missing predictors, ignored dependence), the sampler can explore its posterior perfectly while the
   model still misrepresents the data. Only checks against the data (PPCs, held-out prediction) test the model itself.
2. The difference (12) is only 1.2 SEs: weak evidence. The models predict about equally well. Prefer the simpler or more
   interpretable one, or average them.
3. With a Normal(0, 100²) prior on the intercept, **96%** of draws give probabilities below 0.01 or above 0.99. The
   "vague" prior is actually a strong belief that outcomes are almost certain one way or the other. Normal(0, 1.5²)
   puts 90% of the implied probability between 0.08 and 0.92, with almost no mass at the extremes.
4. PPCs for the full normal regression (observed; replicated 95% range; p): mean 63.93 [62.2, 65.6] p = 0.50; sd 13.84
   [12.4, 15.5] p = 0.54; minimum 24 [19.4, 37.6] p = 0.89; share above 90 0.035 [0.02, 0.06] p = 0.66. The model fits
   well (not surprising: the data were generated from a normal linear model, with the minimum still far from the
   0-point floor).
5. The full model wins by about **20 elpd** with SE 5.5 (about 3.6 SEs): it predicts new students clearly better. This
   agrees with the nested F-test (F = 21.6, p ≈ 10⁻⁹) from Lesson 06.03.
6. Zero-inflated Poisson vs negative binomial: ZIP is worse by about **90 elpd** (SE 17). Its PPCs reproduce the share
   of zeros (p = 0.52) but not the variance (observed 9.9 vs a replicated range of [4.7, 6.3]) or the maximum (p = 0.002).
   ZIP explains the extra zeros with a "never buys" group, but the real excess variation comes from customers with
   varying rates *among buyers too*, which the NB's gamma mixing captures.
7. ```python
   def credible_interval(samples, level=0.95):
       alpha = (1 - level) / 2
       return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))

   def posterior_predictive_pvalue(replicated_stats, observed_stat):
       return float(np.mean(np.asarray(replicated_stats) >= observed_stat))
   ```
8. | Prior on τ | τ posterior median | θ_A posterior mean |
   | --- | --- | --- |
   | HalfCauchy(5) | 2.8 | 6.1 |
   | HalfNormal(5) | 2.7 | 6.0 |
   | HalfNormal(25) | 4.9 | 8.2 |
   | Exponential(1) | 0.6 | 4.8 |

   Robust: "school A's true effect is much smaller than 28, probably 5–8". Sensitive: how much schools differ (τ's
   median ranges from 0.6 to 4.9). With 8 noisy groups, report the τ conclusions with this sensitivity table.
9. The normal model's replicated skewness has 95% range [−0.33, 0.35]; the observed skewness is 1.37, far outside. The
   normal also predicts that **13% of commutes are negative**, which is impossible. The exponential wins on LOO by
   about **50 elpd** (SE 9.7).
10. By Bayes' theorem p(θ | y₋ᵢ) ∝ p(θ | y)/p(yᵢ | θ). So p(yᵢ | y₋ᵢ) = ∫ p(yᵢ | θ) p(θ | y₋ᵢ) dθ, and normalising gives
    1/∫ [1/p(yᵢ | θ)] p(θ | y) dθ = 1/E_post[1/p(yᵢ | θ)]. With S posterior draws this is the harmonic mean of
    p(yᵢ | θ⁽ˢ⁾). Its variance can be infinite, which is why PSIS smooths the largest importance weights with a fitted
    Pareto tail.
11. A PPC asks whether the model *could have generated* data like ours. LOO asks how well it *predicts* unseen points.
    An overfitted model (say, a polynomial of degree 15) can reproduce every statistic of the training data, so it
    passes PPCs, while predicting held-out points badly, so it loses on LOO. Conversely, a simple model can predict
    well on average while failing a PPC for a feature you care about (the tails). Use both.
