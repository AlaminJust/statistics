# 03.03 Solutions

1. P(A | B) = 0.2/0.4 = **0.5**; P(B | A) = 0.2/0.5 = **0.4**. P(A)P(B) = 0.2 = P(A ∩ B), so they are **independent**
   (and indeed P(A | B) = P(A)).
2. 13/52 × 12/51 × 11/50 = **11/850 ≈ 0.0129**.
3. 1 − (48/52)(47/51) = **33/221 ≈ 0.149**.
4. P(link) = 0.8 × 0.3 + 0.1 × 0.7 = 0.24 + 0.07 = **0.31**. The tree splits first into spam (0.3) and not spam (0.7),
   then into link and no link.
5. P(2nd red) = P(R₁)P(R₂ | R₁) + P(B₁)P(R₂ | B₁) = (3/5)(2/4) + (2/5)(3/4) = 6/20 + 6/20 = **3/5**. Before you look at
   the first ball, it gives no information, so by symmetry every position in the draw order is equally likely to be red.
6. Equally likely families: BB, BG, GB, GG (older first). (a) Given the older is a boy: {BB, BG}, so P = **1/2**.
   (b) Given at least one boy: {BB, BG, GB}, so P = **1/3**. "At least one boy" rules out less (only GG), so more
   mixed families remain.
7. (a) 0.95³ ≈ **0.857**. (b) 1 − 0.05³ = **0.999875**. Redundancy in parallel is extremely effective, as long as the
   failures really are independent.
8. P(tutoring) = 0.30; by program 0.278, 0.245, 0.354. That's roughly constant, and the data were generated
   independently, so the gaps are noise. P(score ≥ 70) = 0.325, but P(score ≥ 70 | study ≥ 12) = **0.667** and
   P(score ≥ 70 | study < 12) = **0.157**: clearly dependent.
9. About 0.333.
10. P(A ∩ Bᶜ) = P(A) − P(A ∩ B) = P(A) − P(A)P(B) = P(A)(1 − P(B)) = P(A)P(Bᶜ). ∎
11. P(A) = P(B) = P(C) = 1/2. A ∩ B = {HH}, A ∩ C = {HH}, B ∩ C = {HH}, each with probability 1/4 = 1/2 × 1/2, so every
    pair is independent. But A ∩ B ∩ C = {HH} has probability 1/4 ≠ 1/8. Knowing *two* of the events determines the third.
12. Stay wins about **1/3**, switch about **2/3**. Your first pick is right with probability 1/3, and nothing the host
    does changes that. In the other 2/3 of games the car is behind one of the two other doors, and the host's knowledge
    forces him to open the goat door, so switching lands on the car. The host's action is not random: it carries
    information.
