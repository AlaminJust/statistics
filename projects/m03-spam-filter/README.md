# Mini-project m03 — Probability simulations and a spam filter from scratch

**Module 03 · week 8 · about 6 hours** · Branch: `git switch -c project/m03-spam-filter`

## Part A — three famous problems, exact and simulated (1.5 hours)

For each problem, compute the exact answer, then simulate 100,000 trials and report the estimate with its standard
error √(p̂(1 − p̂)/n) (Lesson 03.06: the standard error of a proportion).

1. **Monty Hall** (Lesson 03.03, exercise 12): P(win) for stay and for switch. Then the variant with 10 doors, where the
   host opens 8 goat doors.
2. **Birthday problem**: P(shared birthday) for n = 2…80, simulated vs exact on one plot.
3. **Coupon collector** (Lesson 03.05, exercise 10): the distribution of the number of rolls needed to see all six faces.
   Plot a histogram and mark the exact mean of 14.7.

## Part B — Naive Bayes spam filter (4 hours)

Data: the SMS Spam Collection (5,574 text messages labelled `ham` or `spam`), from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

```bash
mkdir -p data/raw && cd data/raw
curl -L -o sms.zip "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip sms.zip     # creates SMSSpamCollection: one message per line, "label<TAB>text"
```

`data/raw/` is git-ignored. Note the source and the download date in `data/README.md`.

Steps:

1. **Split** the messages at random (seed 42) into 80% training and 20% test.
2. **Tokenise:** lower-case, keep only letters and digits, split on spaces.
3. **Estimate** from the training set only:
   - the prior P(spam);
   - P(word | class) with Laplace smoothing: (count of word in class + 1) / (total words in class + vocabulary size).
4. **Classify** each test message by comparing log P(class) + Σ log P(word | class) for the two classes (Lesson 03.04).
   Skip words never seen in training.
5. **Evaluate:** accuracy, and a 2 × 2 table of predicted vs true labels. From the table compute P(actually spam |
   predicted spam) and P(predicted spam | actually spam). Which is which in Bayes' theorem?
6. **Compare** with `sklearn.naive_bayes.MultinomialNB` on `CountVectorizer` features. Your accuracy should be close.
7. **Inspect:** the 10 words with the largest log likelihood ratio log P(word | spam) − log P(word | ham). Do they make sense?

Use only the standard library and NumPy for steps 2–5 (`collections.Counter`, `math.log`).

## Deliverables

```text
projects/m03-spam-filter/
├── simulations.py   ← part A
├── spam_filter.py   ← part B
├── report.md        ← results, figures, the confusion table, and what you learned
└── figures/
```

## What "done" looks like

- [ ] Every simulated probability is within 3 standard errors of the exact answer
- [ ] Test accuracy above 95%, and within 1 percentage point of scikit-learn's
- [ ] The report explains why the filter uses logs and smoothing
- [ ] Merged into `main`; tag `m03-done` after the checkpoint
