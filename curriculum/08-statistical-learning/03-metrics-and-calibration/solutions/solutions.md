# 08.03 Solutions

1. F1 = 2 × 0.8 × 0.4/1.2 ≈ **0.533**. Precision 0.8: 80% of the emails it moves to spam really are spam. Recall 0.4: it
   catches only 40% of all spam.
2. t* = 1/(1 + 10) ≈ **0.091**.
3. TP 15, FP 35, FN 5, TN 945. Accuracy **0.960**, precision **0.30**, recall **0.75**, specificity **0.964**, F1 **0.43**.
4. See `../solutions/statkit_learning_reference.py`.
5. U/(n₁n₀) = **0.8495**, exactly the `roc_auc_score`. The AUC is a rank statistic.
6. Max-F1 threshold ≈ 0.24: precision **0.57**, recall **0.48**, F1 0.52. Recall ≥ 0.8 with the best precision: threshold
   ≈ 0.06, precision **0.16**, recall 0.81. Catching 80% of fraud means reviewing about six cases per real fraud.
7. In the middle bins the forest's predicted probabilities are *higher* than the observed frequencies (e.g. 0.074
   predicted vs 0.037 observed): overconfident in the mid range, with probabilities squeezed towards 0 at the low end.
   Isotonic calibration moves the points onto the diagonal.
8. Trained on balanced data: precision **0.12**, recall 0.81 at threshold 0.5. Unbalanced: precision 0.52, recall 0.07.
   The balanced model believes the prior is 50%, so its probabilities are far too high for a 3.5% world (log loss 0.50).
   Bayes correction: multiply the odds by π/(1 − π) ÷ (0.5/0.5), where π is the true positive rate. That gives log loss
   **0.120**, close to the unbalanced model's 0.116. Balancing changes the *probabilities*, not the ranking information.
9. E[(p − Y)²] = q(p − 1)² + (1 − q)p² = p² − 2qp + q. The derivative 2p − 2q = 0 gives **p = q**. ∎
10. 95% bootstrap intervals: ROC-AUC [0.870, 0.912] (width 0.04); PR-AUC [0.449, 0.556] (width 0.11). PR-AUC is more
    uncertain because it depends mainly on the 351 positives, while ROC-AUC averages over all positive–negative pairs.
11. The MLP (96% accuracy) is already fairly well calibrated here: the fitted T ≈ 1.2, and the top-label ECE moves only
    from 0.0186 to 0.0181. Temperature scaling helps most for large, overconfident networks. For small, well-regularised
    models, the fitted T is close to 1. Always measure before calibrating.
