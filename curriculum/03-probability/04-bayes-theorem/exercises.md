# 03.04 Exercises — Bayes' theorem

`make lesson L=03-probability/04-bayes-theorem`

## ★ Warm-up

1. **(M)** Prevalence 2%, sensitivity 90%, specificity 90%. Find P(disease | positive) with the formula and again with
   natural frequencies for 10,000 people.
2. **(M)** In the lesson's medical test (1%, 99%, 95%), what is P(disease | negative)? What is the likelihood ratio of a
   negative result?
3. **(C)** Explain the base-rate fallacy to a friend in three sentences, without formulas.

## ★★ Core

4. **(M)** **The taxi problem.** 85% of a city's taxis are green and 15% are blue. A witness says the taxi in an accident
   was blue. Tested in similar conditions, the witness identifies colours correctly 80% of the time. P(the taxi was
   blue)?
5. **(M)** A workplace drug test is 99% sensitive and 99% specific, and 0.5% of employees use the drug. P(user | positive)?
   Should one positive test be enough to fire someone?
6. **(M)** A bag holds three coins: one fair, one two-headed, one that lands heads 75% of the time. You pick one at random
   and flip it twice: heads, heads. Find the posterior probability of each coin, and P(the next flip is heads).
7. **(M)** Using the odds form, show that two independent positive tests multiply the prior odds by 19.8² in the lesson's
   example. Check that you get 0.798.
8. **(P)** Implement `bayes_posterior(prior, sensitivity, specificity)` in `statkit/probability.py` and reproduce the
   lesson's prevalence table.
9. **(P)** Extend the two-word spam filter: add the word "prize" with P(prize | spam) = 0.30 and P(prize | ham) = 0.01.
   Classify the emails {free}, {meeting}, {free, prize}, {meeting, prize} in log space and print each posterior.

## ★★★ Challenge

10. **(M)** Show that Bayes' rule in log-odds form is log-odds(A | B) = log-odds(A) + log LR(B). If you see several
    pieces of evidence that are conditionally independent given A and given Aᶜ, what happens to their log likelihood
    ratios?
11. **(P)** Write a function that updates a prior over a coin's heads-probability on a grid θ = 0, 0.01, …, 1 (start
    uniform) after each flip of a sequence. Plot the posterior after 0, 1, 5, 20 and 100 flips of a coin with true
    θ = 0.7. (This is a preview of Module 07.)
12. **(C)** In a criminal trial, an expert says "the chance of a DNA match if the defendant is innocent is one in a
    million". Explain what extra information you need to compute P(innocent | match), and why the answer could still
    be large in a city of 10 million people.

Solutions: [solutions/solutions.md](solutions/solutions.md)
