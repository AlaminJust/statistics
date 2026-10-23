# 02.01 Solutions (concept questions; code answers are in solutions.py)

1. Population: all adults with diabetes. Sample: the 300 patients. Parameter: the mean blood pressure of all adults with
   diabetes. Statistic: the mean blood pressure of the 300.
2. | Variable | Type | Level |
   | --- | --- | --- |
   | shoe size | discrete numeric | ratio-ish (sizes are coded steps; often treated as ordinal) |
   | eye colour | nominal | nominal |
   | Olympic medal | ordinal | ordinal |
   | temperature °F | continuous | interval (0 °F isn't "no heat") |
   | emails per day | discrete | ratio |
   | marathon time | continuous | ratio |
   | postcode | nominal (a label made of digits) | nominal |
   | pain 0–10 | ordinal | ordinal (often treated as interval) |
4. (a) Observational: students chose. Association only; confounders like motivation or work hours. (b) Experiment:
   random assignment allows a causal conclusion. (c) Experiment: an A/B test *is* a randomised experiment.
5. (a) Selection bias (only gym members, who are likely healthier than average). (b) Non-response bias (people with
   strong feelings are more likely to post). (c) Survivorship bias: the planes that were hit in other places didn't
   come back. Armour belongs where the survivors *weren't* hit.
6. `study_hours`. Motivated students may both sign up for tutoring and study more, and study hours raise scores. Part
   of the 8-point gap could come from study hours, not tutoring. (In the synthetic data tutoring students study about
   0.7 hours more per week.)
8. 3rd year 0.220 + 4th year 0.295 = **0.515**, or 1 − 0.485 from the cumulative column.
10. The spread (standard deviation) of the sample means shrinks as n grows: about 4.1, 1.9 and 0.5 for n = 10, 40
    and 160. From 10 to 40 (four times the size) the spread roughly halves: the √n law of Module 05. The n = 160 spread
    is much smaller than √n predicts (about 1.0) because the "population" has only 200 students: a sample of 160 is
    80% of everyone, so all samples are nearly the same. (This is the *finite population correction*.)
12. Steeper dependence → larger bias (repliers' mean rises further above 63.9). No dependence → repliers are a random
    subset, and the bias disappears. Bias comes from *how* units enter the sample, not from how many.
