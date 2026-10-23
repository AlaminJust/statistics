# 03.02 Solutions

1. 10! = **3,628,800** seatings; C(10, 4) = **210** committees.
2. STATS has 5 letters with S twice and T twice: 5!/(2! 2!) = **30**. MISSISS has 7 letters with S four times and I twice:
   7!/(4! 2!) = **105**.
3. C(6, 3)/2⁶ = 20/64 = **0.3125**.
4. C(10, 3) × C(8, 2) = 120 × 28 = **3,360**.
5. (a) C(52, 5) = **2,598,960**. (b) C(13, 5)/C(52, 5) = 1,287/2,598,960 ≈ **0.000495**. (c) Four suits, so
   4 × 0.000495 ≈ **0.00198**. (This includes straight flushes.) (d) Choose the rank of the pair (13) and its 2 suits
   (C(4, 2) = 6), then 3 other distinct ranks (C(12, 3) = 220), each in any suit (4³ = 64):
   13 × 6 × 220 × 64 / 2,598,960 ≈ **0.4226**.
6. (C(10, 8) + C(10, 9) + C(10, 10))/2¹⁰ = (45 + 10 + 1)/1024 = 56/1024 ≈ **0.0547**.
7. C(5, 2) × C(3, 2) × C(4, 1) / C(12, 5) = 10 × 3 × 4 / 792 = 120/792 ≈ **0.1515**.
8. ```python
   def n_choose_k(n, k):
       if k < 0 or k > n:
           return 0
       k = min(k, n - k)               # symmetry keeps the loop short
       result = 1
       for i in range(1, k + 1):
           result = result * (n - k + i) // i   # always an exact integer at each step
       return result
   ```
10. Each of the other 29 people misses your birthday with probability 364/365, so P = 1 − (364/365)²⁹ ≈ **0.077**.
    (With 30 *other* people, 1 − (364/365)³⁰ ≈ 0.079.) "Some pair matches" is much likelier because it covers all
    C(30, 2) = 435 pairs, whereas "someone matches *me*" covers only 29.
11. Fix person X. Committees of size k from n people either include X (choose the other k − 1 from the remaining n − 1:
    C(n − 1, k − 1) ways) or exclude X (choose all k from the other n − 1: C(n − 1, k) ways). Every committee falls in
    exactly one group, so the counts add. ∎
12. For N = 2³², about **77,000 records** already give a 50% chance of a duplicate. The rule of thumb is
    n ≈ 1.18 √N. For 64-bit IDs it's about **5 × 10⁹** records. Random 32-bit IDs are unsafe even for small tables. Use
    64- or 128-bit random IDs (UUIDs are 122 random bits) or sequential IDs.
