# 08.05 Exercises — PCA and high-dimensional data

`make lesson L=08-statistical-learning/05-pca-high-dimensional`

## ★ Warm-up

1. **(M)** For the covariance matrix [[4, 2], [2, 3]], compute the eigenvalues by hand (solve det(Σ − λI) = 0) and the
   share of variance explained by the first component.
2. **(C)** Why does nearest-neighbour classification get worse when many irrelevant dimensions are added?
3. **(C)** Two components explain 95% of the variance of a 50-variable dataset. What does that tell you?

## ★★ Core

4. **(P)** Implement `pca(X, k)` in `statkit/learning.py`: centre X, compute the covariance matrix, take its top-k
   eigenvectors (`np.linalg.eigh`), and return (components of shape (k, p), explained variance ratios, scores of shape
   (n, k)). Check against scikit-learn on the digits data, up to sign.
5. **(P)** Add 50, 200 and 1,000 pure-noise columns to the breast-cancer features. Compare the 5-fold CV accuracy of
   kNN (k = 5) on (a) all features and (b) the first 10 principal components (standardise first, inside a pipeline).
   When does PCA help?
6. **(P)** Show the first 6 principal components of the digits data as 8 × 8 images. What does each seem to capture?
7. **(P)** Compress the digits with 10 components and reconstruct them. Plot 5 originals next to their reconstructions.
   Then do the same with 30 components.
8. **(P)** Run PCA on the four numeric columns of `students.csv` (standardised). How much variance does PC1 explain, and
   what are its loadings? Interpret it.

## ★★★ Challenge

9. **(M)** Show that the variance of the projection X**w** onto a unit vector **w** is **w**ᵀΣ**w**, and that it's maximised
   by the top eigenvector. (Use a Lagrange multiplier.)
10. **(P)** **Denoising.** Add Gaussian noise (sd 4) to the digits, then reconstruct with 20 components. Compare the RMSE
    to the clean images for the noisy and the reconstructed versions. Why does PCA remove noise?
11. **(P)** **Distance concentration.** For d from 1 to 1,000, compute the ratio (max − min)/min of distances from one
    point to 500 uniform random points, averaged over 20 repetitions. Plot it against d on log axes. What's the slope?

Solutions: [solutions/solutions.md](solutions/solutions.md)
