# 01.05 Solutions

1. **a** + **b** = (3, 3, 3); 3**a** = (6, −3, 9); **a**·**b** = 2 − 4 + 0 = **−2**; ‖**a**‖ = √14 ≈ 3.742;
   ‖**b**‖ = √17 ≈ 4.123.
2. 2 + 0 − 2 = **0**: the vectors are perpendicular (cos θ = 0).
3. (5 × 4); impossible (inner 4 ≠ 5); (2); (1 × 1), a single number (a dot product); (3 × 3).
4. [[2, 0], [1, 3]]·[[1, 4], [2, 1]] = [[2·1 + 0·2, 2·4 + 0·1], [1·1 + 3·2, 1·4 + 3·1]] = **[[2, 8], [7, 7]]**.
   Other order: **[[6, 12], [5, 3]]**. Different, as expected.
5. det = 4·1 − 2·1 = 2. A⁻¹ = (1/2)[[1, −2], [−1, 4]] = **[[0.5, −1], [−0.5, 2]]**.
   Check row 1 of A⁻¹A: (0.5·4 + (−1)·1, 0.5·2 + (−1)·1) = (1, 0) ✓.
6. A = [[3, 1], [1, 2]], **b** = (9, 8), det = 5, A⁻¹ = (1/5)[[2, −1], [−1, 3]].
   **x** = (1/5)(18 − 8, −9 + 24) = (1/5)(10, 15) = **(2, 3)**. Check: 3·2 + 3 = 9 ✓, 2 + 2·3 = 8 ✓.
7. det = 1·4 − 2·2 = 0. The second column (2, 4) is 2 × the first (1, 2), so the matrix squashes the plane onto a line
   and the information can't be recovered: no inverse. In regression this happens when one feature is a multiple of
   another (*perfect multicollinearity*, Module 06).
8. cos(**s₁**, **s₂**) = 0.9997, cos(**s₁**, **s₃**) = 0.738, cos(**s₂**, **s₃**) = 0.722. Students 1 and 2 are most
   similar. Cosine similarity ignores **length**: (10, 7) and (100, 70) would get similarity 1, even though they're
   very different students. Here the variables also have different units. For data like this, distances on
   standardised variables (Module 02, z-scores) are usually better.
9. See `../solutions/statkit_mathkit_reference.py` in the module folder.
10. (XᵀX)ᵀ = Xᵀ(Xᵀ)ᵀ = XᵀX, using (AB)ᵀ = BᵀAᵀ and (Xᵀ)ᵀ = X.
11. X = [[1, 1], [1, 2], [1, 3]], **y** = (2, 3, 5). XᵀX = [[3, 6], [6, 14]], Xᵀ**y** = (10, 23), det = 6.
    **β** = (1/6)[[14, −6], [−6, 3]](10, 23) = (1/6)(2, 9) = **(0.333, 1.5)**. Line: y = 0.333 + 1.5x.
12. **β** ≈ (24.62, 2.09, 2.59). The data were generated as 26 + 2.2·study + 2.0·sleep + 5·tutoring + noise. The study and
    sleep slopes come out close to 2.2 and 2.0. The intercept differs partly because tutoring was left out of X, and
    partly because of noise. Module 06 shows how to put intervals around these estimates.
13. W is **128 × 784**, **b** has **128** entries, **h** has **128** entries. Parameters: 128 × 784 + 128 = **100,480**.
