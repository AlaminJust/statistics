# 00.01 Solutions

1. `pwd` prints something like `/Users/<you>/MyProject/statistics`; `ls` lists `Makefile PLAN.md PROGRESS.md README.md
   curriculum data docs ...`.
2. "Inside a venv? True" after `source .venv/bin/activate`. The numpy version should match `requirements.txt` (2.5.4).
3. - A **package** is reusable code written by others that you install (e.g. NumPy).
   - A **virtual environment** is a project-private copy of Python plus exact package versions.
   - A **commit** is a permanent, named snapshot of the staged changes, with author, date and message.
4. Example log (IDs will differ):
   ```text
   c41e2d9 exercise(00.01): answer warm-up exercises 1-3
   8a7b310 learn(00.01): one-sentence summary of venv and git
   2f9d0aa learn(00.01): start environment and git
   ```
5. In `git diff`, lines starting with `-` are the old version and `+` the new version of the changed lines. After
   `git restore notes.md`, `git diff` prints nothing: the file matches the last commit again.
6. Packing a box: the working folder is everything on your desk; `git add` puts chosen items in the box; `git commit`
   seals and labels the box and posts it. You can keep working at your desk without changing what's already posted.
7. ```python
   def ex_7(seed=2026):
       rng = np.random.default_rng(seed)
       return rng.integers(0, 2, size=10)
   ```
   With a seed the 10 flips are identical every run; without one (`default_rng()`) they change, because the generator
   starts from fresh entropy each time.
8. `git log --reverse --oneline | head -1` prints the first commit. `git show <id>:README.md` prints a file as it was
   in that commit; this is how you recover any old version.
9. (a) **Different package versions** can change defaults and results → fixed by pinned versions in
   `requirements.txt` and `make setup`. (b) **Unseeded randomness** gives different samples each run → fixed by always
   using `np.random.default_rng(<seed>)`.
10. `gh repo create statistics-from-scratch --private --source=. --push` prints the repository URL.
