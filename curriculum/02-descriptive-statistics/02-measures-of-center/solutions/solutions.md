# 02.02 Solutions

1. Sorted: 3, 4, 5, 6, 8, 8, 9. Mean = 43/7 ≈ **6.14**, median = **6**, mode = **8**.
2. Sorted: 3, 7, 9, 10, 12, 14. Mean = 55/6 ≈ **9.17**, median = (9 + 10)/2 = **9.5**.
3. (a) Median: prices are right-skewed by a few mansions. (b) Mode: the shop wants the most common size. (c) Mean:
   symmetric data, and it uses all the information. (d) Mode: nominal data.
4. (10·72 + 25·65 + 15·80) / 50 = (720 + 1625 + 1200)/50 = **70.9**. Unweighted: (72 + 65 + 80)/3 = 72.33, too high
   because the small groups had the high means.
5. Factors 1.1 × 1.2 × 0.7 = 0.924. GM = 0.924^(1/3) ≈ **0.974** → about **−2.6% per year**. Check: 100 → 110 → 132 →
   92.4, and 100 × 0.974³ ≈ 92.4 ✓. (The arithmetic mean of the rates, 0%, wrongly says you broke even.)
6. Mean = 253/10 = **25.3**, median = (18 + 19)/2 = **18.5**, 10% trimmed (drop 12 and 95) = 146/8 = **18.25**.
   The median and trimmed mean agree; both ignore the extreme 95, which pulls the mean up by about 7.
7. | program | mean | median | mean − median |
   | --- | --- | --- | --- |
   | Arts | 10.52 | 9.95 | 0.57 |
   | Commerce | 11.13 | 10.50 | **0.63** |
   | Science | 9.27 | 9.40 | −0.13 |

   Commerce is the most right-skewed by this measure (Arts is close). The differences are small, and Module 05 will show
   how to judge whether they're more than noise.
8. See `../solutions/statkit_descriptive_reference.py` in the module folder.
9. Adding 10 adds 10 to both the mean and the median. Multiplying by 3 multiplies both by 3. Measures of centre move
   with the data's location and scale (*equivariance*). This is why changing units (minutes → hours) is harmless.
10. Group g has nᵍ values with sum Tᵍ, so its mean is x̄ᵍ = Tᵍ/nᵍ. Then Σnᵍx̄ᵍ / Σnᵍ = ΣTᵍ / N = (sum of all values)/N =
    the overall mean. ∎
11. Typical results (seed 0):

    | Distribution | sd of means | sd of medians | Winner |
    | --- | --- | --- | --- |
    | normal | 2.01 | 2.51 | mean |
    | exponential | 5.00 | 5.00 | tie |
    | t with 2 df (heavy tails) | 0.65 | 0.28 | **median** |

    For clean, symmetric data the mean is more precise (it uses every value). With heavy tails and outliers, the median
    is far more stable. Robust statistics is built on this trade-off.
12. Mean = 3, median = 3 (any value between 1 and 5 splits the data in half; by the even-n rule it is (1 + 5)/2 = 3).
    Both describe no actual reviewer. Better: "Reviews are split: half the buyers rate it 1 star and half 5 stars."
