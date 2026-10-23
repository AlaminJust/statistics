# 08.05 — PCA and high-dimensional data

**Time:** about 4 hours · **Prerequisites:** 01.05, 03.06 (covariance matrices), 04.03, 08.01

## Why this matters

Modern data are wide: images have thousands of pixels, genomics has tens of thousands of genes, embeddings from language
models have hundreds or thousands of dimensions. High dimensions behave counter-intuitively ("the curse of
dimensionality"), and many variables are redundant. **Principal component analysis** (PCA) finds the few directions that
carry most of the variation: for visualisation, compression, denoising, and as preprocessing. It's also the simplest
example of *representation learning*, and its mathematics (eigenvectors of the covariance matrix, the SVD) is everywhere
in ML.

## Learning objectives

- Describe how distances and volumes behave in high dimensions, and why that hurts nearest-neighbour methods.
- Derive PCA as the eigendecomposition of the covariance matrix, and compute it by hand, with the SVD and with scikit-learn.
- Interpret explained variance, choose the number of components, and reconstruct data.
- Explain when to standardise before PCA.
- Use PCA inside a pipeline, and know its limitations.

## Before you start

You pick 500 random points in a 1,000-dimensional cube. How much farther is the farthest point from a given point than
the nearest one: 40 times? 4 times? Barely farther at all?

---

## 1. The curse of dimensionality

**Worked example 1:** 500 uniform points in [0, 1]ᵈ, with distances from the first point to all the others.

| d | nearest | farthest | farthest/nearest |
| --- | --- | --- | --- |
| 2 | 0.025 | 1.06 | **42** |
| 10 | 0.49 | 2.11 | 4.3 |
| 100 | 3.21 | 4.84 | 1.5 |
| 1000 | 12.1 | 13.8 | **1.14** |

Answer to "Before you start": **barely farther**. In high dimensions, all points are nearly equally far apart, so
"nearest neighbour" carries little information. Volume behaves strangely too: the unit ball fills 79% of the cube
[−1, 1]² but only 0.25% of the 10-dimensional cube and essentially none of the 50-dimensional one. Almost all the volume
of a high-dimensional cube is in its corners.

Consequences: k-NN, kernel methods and density estimation need far more data as d grows; distance-based clustering
degrades. The rescue is that **real data usually lie near a much lower-dimensional structure**: a handwritten 8 is
determined by a few strokes, not 64 independent pixels. PCA finds the linear version of that structure.

## 2. PCA: directions of maximum variance

Centre the data (subtract each column's mean). The **first principal component** is the unit vector **w** that
maximises the variance of the projections X**w**:

$$\max_{\|\mathbf{w}\| = 1} \operatorname{Var}(X\mathbf{w}) = \max_{\|\mathbf{w}\| = 1} \mathbf{w}^\top \Sigma\, \mathbf{w}.$$

The solution is the **eigenvector of the covariance matrix Σ with the largest eigenvalue**, and the eigenvalue is the
variance captured. (Use a Lagrange multiplier: ∇[**w**ᵀΣ**w** − λ(**w**ᵀ**w** − 1)] = 0 ⇒ Σ**w** = λ**w**.) Later components
are the next eigenvectors, each orthogonal to the previous ones.

**Worked example 2.** Two correlated variables:

- covariance matrix [[1.007, 0.811], [0.811, 0.817]];
- eigenvalues **1.729** and **0.095**: the first component explains **94.8%** of the variance;
- first direction (0.747, 0.665), the long axis of the elliptical cloud (`pca.png`, left);
- the component scores are **uncorrelated**, and their variances equal the eigenvalues.

The **SVD** of the centred data, X = UΣVᵀ, gives the same thing: the rows of Vᵀ are the components and
(singular values)²/(n − 1) are the eigenvalues. This is how libraries compute PCA, because it's numerically more stable.
(Signs are arbitrary: (0.747, 0.665) and (−0.747, −0.665) describe the same direction.)

## 3. How many components?

**Worked example 3: handwritten digits** (64 pixels):

| Components | 2 | 5 | 10 | 20 | 30 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| cumulative explained variance | 28.5% | 54.5% | 73.8% | 89.4% | 95.9% | 98.8% |

21 components explain 90% and 29 explain 95%. Reconstruction error (RMSE in pixel units, range 0–16): 2.92 with 5
components, 2.22 with 10, 1.41 with 20, 0 with all 64.

Ways to choose k:

- an explained-variance threshold (90% or 95%);
- the "elbow" of the scree plot;
- **the best downstream performance in cross-validation**, often the most relevant.

**Worked example 4: kNN on digits** (5-fold CV): 2 components → 0.512; 5 → 0.827; 10 → 0.893; 20 → 0.937; 40 → 0.939;
all 64 pixels → **0.944**. A third of the features give nearly the same accuracy. Here PCA doesn't *improve* accuracy (the
data aren't very high-dimensional), but it compresses, speeds things up and denoises. With many more noisy features it
can help. Put PCA **inside the pipeline**, so components are learned on training folds only (Lesson 08.02).

## 4. Standardise when units differ

PCA maximises *variance*, so variables with big numbers dominate. **Worked example 5: breast-cancer data.** On raw data,
PC1 explains 98.2% of the variance, almost entirely "worst area" (sd 569, in square units), which is just a measure of
size. After standardising, PC1 explains 44.3%, with loadings spread over concavity and other shape variables, a more
meaningful summary. Two standardised components give logistic regression **95.1%** CV accuracy.

Rule: standardise unless all variables share meaningful units (pixels, gene expression on the same scale).

## 5. Limitations and relatives

| Limitation | Alternative |
| --- | --- |
| linear only | kernel PCA; autoencoders; UMAP or t-SNE (for visualisation only) |
| maximises variance, which isn't necessarily what predicts y | partial least squares; supervised methods |
| sensitive to outliers | robust PCA |
| components can be hard to interpret | sparse PCA; factor analysis |
| no probabilistic model | **probabilistic PCA** (a latent-variable model fitted by EM, Lesson 07.05) |

**Embeddings.** Sentence and image embeddings are high-dimensional vectors compared by cosine similarity (Lesson 01.05).
PCA is routinely used to visualise them, to remove a few dominant "uninformative" directions (which often improves
similarity search), and to compress them for storage.

---

## Common mistakes

| Mistake | Correct |
| --- | --- |
| PCA on unscaled variables with different units | Standardise first |
| Fitting PCA on all data before CV | PCA inside a pipeline |
| Interpreting the sign of a component | Signs are arbitrary |
| Assuming the first components are best for prediction | Variance ≠ relevance to y |
| Reading cluster distances off a t-SNE plot | t-SNE and UMAP distort global distances |

## Summary

- High dimensions: distances concentrate, volume lives in the corners; real data usually lie near a low-dimensional structure.
- PCA: eigenvectors of the covariance matrix (or the SVD of centred data); eigenvalues = variance per component; scores are uncorrelated.
- Choose k by explained variance or downstream CV; reconstruct with the inverse transform.
- Standardise when units differ; fit inside pipelines.

## Self-test

1. Why does nearest-neighbour classification get worse as irrelevant dimensions are added?
2. What do the eigenvalues of the covariance matrix represent in PCA?
3. Two components explain 95% of the variance. What does that tell you about the data?
4. Why might the first principal component be useless for predicting y?

Next: [exercises.md](exercises.md) · run [examples.py](examples.py)
