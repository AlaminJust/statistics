# 08.02 Solutions

1. With thousands of noise features, some correlate with *these* random labels purely by chance, including with the
   labels of the future validation folds. Selecting them uses the validation labels, so the validation folds are no
   longer unseen.
2. (a) Forward-in-time validation (`TimeSeriesSplit`), possibly also grouped by store if you'll forecast new stores.
   (b) `GroupKFold` by patient. (c) Stratified k-fold. (d) Split by time (and by user if the model must generalise to
   new users).
3. "Days late on the last payment" recorded after the default; "sent to collections"; "loan restructured"; the final
   recovered amount. All are consequences of default and unknown at application time.
4. "Wrong" vs "right" CV accuracy (seed 0): p = 100 → 0.77 vs 0.63; 500 → 0.85 vs 0.52; 2,000 → 0.88 vs 0.47;
   10,000 → **0.95** vs 0.60. (The "right" values scatter around 0.5 because n = 100 is small.) With more noise
   features, the best 20 happen to be more strongly correlated with the random labels, so the leak grows.
5. Here, no difference (both 0.965). Scaling uses only means and sds, which barely change between folds with n = 569.
   It matters more for small samples, heavy-tailed features, or steps that learn more from the data (imputation by
   target-dependent models, target encoding, PCA with few samples).
6. Nested CV: logistic **0.975** (sd 0.014), SVM **0.974** (sd 0.024); mean fold difference 0.002 (sd 0.015). No meaningful
   difference. Pick the simpler, more interpretable model.
7. Without duplicates: mean 1-NN test accuracy 0.954 over 30 splits; with 30% near-duplicates: **0.965**. Many test points
   now have an exact twin in the training set, and 1-NN simply looks it up. Nearest-neighbour methods and large
   memorising models (deep nets, big trees) gain most from duplicates. Deduplicate before splitting.
8. See `../solutions/statkit_learning_reference.py`.
9. CV AUC with user_id: 0.554; without: 0.554. Not leaky here, because user IDs were assigned independently of the
   outcome. In real systems IDs are often assigned in time order, or by source (one batch of positive cases imported
   together, one hospital's IDs in a range). Then the ID encodes time or origin, which can correlate with the label and
   won't generalise. Treat IDs as suspicious by default.
10. Shuffled KFold underestimates the forecasting error (it interpolates between neighbours). `TimeSeriesSplit` and the
    final hold-out are closer to the true error on the next 200 points. The final hold-out is noisier but most faithful
    to the deployment setting.
11. A common pattern in such post-mortems: a feature computed with information from after the prediction time (for
    example a cumulative total recomputed nightly on the full table), caught because it had an implausibly high feature
    importance, or because live performance collapsed compared with offline validation.
