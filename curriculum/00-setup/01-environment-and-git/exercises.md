# 00.01 Exercises — Environment and git

Start with `make lesson L=00-setup/01-environment-and-git`. Write answers in
`workspace/00-setup/01-environment-and-git/notes.md` (C and M) or `exercises.py` (P).

## ★ Warm-up

1. **(P)** In the terminal, go to the project folder, print where you are with `pwd`, and list the files. Copy the
   output into your notes.
2. **(P)** Activate the environment and run `examples.py`. Is "Inside a venv?" `True`? Write down your numpy version.
3. **(C)** In one sentence each: what is a package, a virtual environment, a commit?

## ★★ Core

4. **(P)** Make three commits in a row:
   - commit 1: start the lesson (`make lesson ...`, then add and commit);
   - commit 2: write your one-sentence summary of the lesson in `notes.md`;
   - commit 3: add the answers to exercises 1–3.

   Use the message format from the lesson. Then run `git log --oneline -5` and paste the output in your notes.
5. **(P)** Change one line in `notes.md` but **don't** commit. Run `git diff` and explain what the `+` and `-` lines mean.
   Then run `git restore` on the file and confirm the change is gone.
6. **(C)** Explain the staging area using an analogy (for example, packing a box before posting it).
7. **(P)** In `exercises.py`, write a function that creates `rng = np.random.default_rng(2026)` and returns 10 coin
   flips (0 or 1). Run it twice. Are the outputs equal? Now remove the seed and run twice again. Explain the difference.

## ★★★ Challenge

8. **(P)** Find the ID of your *first* commit with `git log --reverse --oneline | head -1`. Use `git show <id>` to see
   what it contained. Then use `git show <id>:README.md | head -5` to print the README as it was then.
9. **(C)** A labmate's result can't be reproduced. They used `pandas` without a version number and no random seed.
   List two separate causes of the problem and how this repository prevents each one.
10. **(P)** Create a private GitHub repository and push this project (see `docs/setup.md`, section 5). Note the URL in
    your journal.

Solutions: [solutions/solutions.md](solutions/solutions.md) — open only after committing your attempt.
