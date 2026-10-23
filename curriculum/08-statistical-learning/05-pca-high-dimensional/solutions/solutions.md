# 08.05 Solutions

1. det = (4 − λ)(3 − λ) − 4 = λ² − 7λ + 8 = 0 ⇒ λ = (7 ± √17)/2 = **5.56** and **1.44**. PC1 explains 5.56/7 ≈ **79.5%**.
2. With irrelevant dimensions, distances are dominated by noise in those dimensions, and all points look about equally
   far (distance concentration). The true nearest neighbours in the relevant dimensions stop being nearest.
3. The 50 variables are highly redundant: the data lie close to a 2-D plane in 50-D space, so most variables are noisy
   linear combinations of two underlying factors.
4. ```python
   def pca(X, k):
       X = np.asarray(X, float)
       centred = X - X.mean(axis=0)
       cov = np.cov(centred, rowvar=False)
       values, vectors = np.linalg.eigh(cov)
       order = np.argsort(values)[::-1][:k]
       components = vectors[:, order].T
       return components, values[order] / values.sum(), centred @ components.T
   ```
5. CV accuracy (all features vs 10 PCs): 0 noise columns 0.965 vs 0.961; 50 → 0.940 vs 0.954; 200 → 0.893 vs 0.923;
   1,000 → **0.844 vs 0.900**. With many noise dimensions, PCA concentrates the correlated (real) signal in the first
   components and discards much of the uncorrelated noise, so kNN distances become meaningful again. With no noise
   columns, it costs a little.
6. Component 1 roughly contrasts "0-like" with "1-like" shapes (ink in a ring vs a vertical stroke). Later components
   capture stroke position and slant. They're blurry "eigen-digits", mixtures of many digits rather than single digits.
7. 10 components: digits recognisable but blurred. 30 components: nearly indistinguishable from the originals (95.9% of
   the variance).
8. Explained variance (standardised): 0.433, 0.263, 0.243, 0.061. PC1 loadings: study **0.70**, sleep 0.05, commute
   −0.04, score **0.71**. PC1 is the study–score axis, the one strong relationship in the data. The middle components
   are mostly sleep and commute on their own (they're nearly independent of everything else).
9. Var(X**w**) = **w**ᵀΣ**w** (Lesson 03.06). Maximise it subject to **w**ᵀ**w** = 1: L = **w**ᵀΣ**w** − λ(**w**ᵀ**w** − 1),
   ∇L = 2Σ**w** − 2λ**w** = 0 ⇒ Σ**w** = λ**w**. So **w** is an eigenvector, and the variance **w**ᵀΣ**w** = λ is largest
   for the largest eigenvalue. ∎
10. RMSE against the clean images: noisy **4.00**, reconstructed from 20 components **2.73**. Noise is spread evenly over
    all 64 directions, while the digit signal is concentrated in the first 20; projecting onto those keeps most of the
    signal and only about 20/64 of the noise.
11. Average (max − min)/min: d = 1 → huge (the nearest point is almost at distance 0); d = 10 → 2.42; d = 100 → 0.40;
    d = 1,000 → 0.115. On log axes the slope is about **−0.66** over d = 10 to 1,000, approaching the theoretical −1/2
    as d grows: the relative spread of distances shrinks roughly like 1/√d.
