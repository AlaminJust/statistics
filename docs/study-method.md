# How to study each lesson — the 6-step loop

Reading alone gives the feeling of understanding without the skill. Every lesson in this course is built for the loop
below. Follow it every time; it is the single biggest factor in whether you actually learn this.

## The loop

| Step | What you do | Where | Time |
| --- | --- | --- | --- |
| 1. Preview | Read the lesson's "Why this matters" and objectives. Try the "Before you start" question and write your guess. | `notes.md` | 10 min |
| 2. Read and run | Read `lesson.md` with a pen. Redo every worked example **by hand** before looking at its answer. Then run `examples.py`. | paper + terminal | 60–90 min |
| 3. Explain | Close the lesson. Fill in `notes.md` from memory: big idea, terms, formulas, one worked example with **new numbers**. | `notes.md` | 30 min |
| 4. Practise | Warm-up → core → challenge exercises. Write your expected answer *before* running code. | `exercises.py` | 2–3 h |
| 5. Build | Implement the lesson's `statkit` functions until their tests pass. | `statkit/` | 1–1.5 h |
| 6. Review | Check against `solutions/`, fix mistakes in a separate commit, tick `PROGRESS.md`. | git | 20 min |

## Rules that make it work

1. **Commit your first attempt before opening solutions.** Your git log then shows what you got wrong and how you fixed it.
2. **Never copy-paste code from a lesson.** Type it. Typing forces you to read every symbol.
3. **Predict, then run.** Before every `python ...`, write what you expect in a comment. Wrong predictions are where learning happens.
4. **Stuck for more than 30 minutes?** Write exactly where you are stuck in the journal, move to the next exercise, and come back the next day.
5. **Spaced review.** Every Monday, spend 15 minutes answering last week's "Self-test" questions from memory.
6. **One module checkpoint at a time.** Do not start the next module until you pass the current checkpoint without notes.

## Exercise labels

| Label | Meaning |
| --- | --- |
| **(C)** Concept | Explain in words. If you can't explain it simply, you don't understand it yet. |
| **(M)** Math | Pen and paper. Show every step. |
| **(P)** Python | Code in `workspace/.../exercises.py`. |
| ★ / ★★ / ★★★ | Warm-up / core / challenge. Core exercises are required; challenge exercises are for depth. |
