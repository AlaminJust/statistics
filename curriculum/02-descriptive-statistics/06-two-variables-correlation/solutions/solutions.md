# 02.06 Solutions

1. P(Commerce and no) = 37/200 = **0.185**. P(no | Commerce) = 37/49 = **0.755**. P(Commerce | no) = 37/140 = **0.264**.
   The first conditional asks "among Commerce students, what share skip tutoring?" (denominator: 49 Commerce students).
   The second asks "among students who skip tutoring, what share are in Commerce?" (denominator: 140 students).
2. x̄ = 3, ȳ = 3.

   | xᵢ − x̄ | −2 | −1 | 0 | 1 | 2 |
   | --- | --- | --- | --- | --- | --- |
   | yᵢ − ȳ | −1 | −2 | 1 | 0 | 2 |
   | product | 2 | 2 | 0 | 0 | 4 |

   Σ = 8 → cov = 8/4 = **2**. Σdx² = 10 and Σdy² = 10, so s_x = s_y = √2.5 ≈ 1.581. r = 2/2.5 = **0.8**.
3. **0.6** (shifting and positive rescaling don't change r); **−0.6**; **1**.
4. x = angle through a full circle and y = sin(x); or y = x² on symmetric x; or y = how loud the crowd is vs how close
   the game's score is. Any U-shaped or periodic pattern.
5. | year | no | yes |
   | --- | --- | --- |
   | 1 | 0.636 | 0.364 |
   | 2 | 0.619 | 0.381 |
   | 3 | 0.818 | **0.182** |
   | 4 | 0.729 | 0.271 |

   Year 3 attends least (18%). In this synthetic data tutoring was assigned at random, so the differences are pure chance.
   Module 05 shows how to check that.
6. See `../solutions/statkit_descriptive_reference.py`.
7. Spearman: I 0.818, II 0.691, III **0.991**, IV 0.500. Spearman captures set III's near-perfect monotonic line
   (ignoring the outlier) much better than Pearson's 0.816. For set II (a curve that rises then falls) neither describes
   the shape: you need the plot.
8. Arts 0.811, Commerce 0.778, Science 0.557. Science is noticeably weaker. Possible reasons: chance (Module 05) or a
   genuinely different relationship. The overall 0.728 averages over these.
9. Causal: library time improves learning. Reverse: students who already have high grades enjoy studying and use the
   library more. Confounded: motivation (or living on campus) drives both. A randomised study could tell them apart,
   for example randomly offering some students a library-study programme and comparing their grade changes.
10. r = **u**·**v** / (‖**u**‖‖**v**‖) with **u** = x − x̄ and **v** = y − ȳ. Cauchy–Schwarz says |**u**·**v**| ≤
    ‖**u**‖‖**v**‖, so |r| ≤ 1. Equality holds exactly when **v** is a multiple of **u**, i.e. the points lie on a line. ∎
11. With seed 0: **136** of 4,950 pairs have |r| > 0.5, and the largest |r| is about **0.76**, all between variables that
    are unrelated by construction. With small n and many comparisons, impressive correlations appear by chance. Any
    single "discovery" from such a search must be confirmed on new data.
12. Example: group A has x ~ 1–5 with y = 10 − x + noise; group B has x ~ 6–10 with y = 20 − x + noise. Within each group
    y falls as x rises, but B sits higher on both axes, so the pooled cloud rises from A to B. The grouping variable is a
    confounder: the pooled correlation mixes the "between groups" pattern with the "within groups" pattern.
    (Module 09 covers the classic real case, the 1973 UC Berkeley admissions data.)
