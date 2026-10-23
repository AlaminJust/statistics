# 00.01 — Environment and git

**Time:** about 3 hours · **Prerequisites:** none

## Why this matters

Every statistics result you produce in this course — and later in research — must be **reproducible**: someone else
(or you, in six months) must be able to rerun it and get the same numbers. Reproducibility needs two things:

1. **The same software**: a virtual environment with fixed package versions (`requirements.txt`).
2. **A record of every change**: git.

Research labs and ML teams work exactly this way. Learning it now means you never lose work and you build a public
record of your skills.

## Learning objectives

By the end you can:

- move around folders in the terminal;
- explain what a virtual environment is and activate one;
- explain git's three areas (working folder, staging area, history);
- make commits with good messages, view history and differences, and tag a milestone.

## Before you start

Write your guess in `notes.md`: *If you delete a file by mistake, what would you need to get it back exactly as it was
last Tuesday?*

---

## 1. The terminal in 6 commands

The terminal is a text window where you type commands. On macOS open **Terminal** (or the terminal panel in VS Code,
`` Ctrl+` ``).

| Command | Meaning | Example |
| --- | --- | --- |
| `pwd` | **p**rint **w**orking **d**irectory — where am I? | `/Users/you/MyProject/statistics` |
| `ls` | **l**i**s**t files here | `README.md  curriculum  data ...` |
| `cd <folder>` | **c**hange **d**irectory | `cd curriculum/00-setup` |
| `cd ..` | go up one folder | |
| `mkdir <name>` | make a folder | `mkdir scratch` |
| `cat <file>` | print a file | `cat README.md` |

**Worked example.** Starting in your home folder:

```bash
pwd                         # /Users/you
cd MyProject/statistics
pwd                         # /Users/you/MyProject/statistics
ls                          # README.md  PLAN.md  curriculum  data ...
cd curriculum/00-setup
cd ../..                    # back up two levels
pwd                         # /Users/you/MyProject/statistics
```

Tip: press `Tab` to auto-complete folder names.

## 2. Python, packages and virtual environments

- **Python** is the language.
- A **package** is code someone else wrote that you install: NumPy (fast arrays), pandas (tables), SciPy (statistics
  functions), matplotlib (plots).
- A **virtual environment** (`.venv/`) is a private folder holding one Python plus the exact package versions for
  *this* project, separate from every other project on your computer.

Why separate? Suppose project A needs pandas 2.0 and project B needs pandas 3.1. Without virtual environments one of
them breaks. With them, each project has its own copy.

```bash
make setup                    # = python3 -m venv .venv + pip install -r requirements.txt
source .venv/bin/activate     # your prompt now starts with (.venv)
which python                  # .../statistics/.venv/bin/python   ← the project's own Python
python --version
deactivate                    # leave the environment
```

`requirements.txt` lists exact versions such as `numpy==2.5.4`. Anyone who runs `make setup` gets the same versions, so
the same code gives the same results.

## 3. Git: the mental model

Git keeps a history of **snapshots** of your project. Each snapshot is a **commit**. There are three areas:

```text
 working folder  ──git add──▶  staging area  ──git commit──▶  history (commits)
 (files you edit)              (what goes in the               (permanent snapshots,
                                next snapshot)                   each with an ID)
```

- **Working folder**: your files as they are now.
- **Staging area**: the changes you have *chosen* for the next snapshot. This lets you commit related changes
  together, e.g. notes for one lesson, but not a half-finished exercise.
- **History**: commits, each with an ID (like `a3f9c21`), author, date and message. You can return to any of them.

### Worked example: your first three commits

```bash
cd ~/MyProject/statistics
make lesson L=00-setup/01-environment-and-git    # creates workspace/00-setup/01-environment-and-git/notes.md
git status
```

Git shows the new folder as **untracked** (red): it exists but git isn't recording it yet.

```bash
git add workspace/00-setup/01-environment-and-git
git status                    # now green: staged
git commit -m "learn(00.01): start environment and git"
git log --oneline -3          # newest commit at the top
```

Now edit `notes.md` — write the big idea of this lesson in one sentence — and look at exactly what changed:

```bash
git diff                      # lines starting with + were added, - were removed
git add workspace/00-setup/01-environment-and-git/notes.md
git commit -m "learn(00.01): notes on venv and git's three areas"
```

### Undoing things

| Situation | Command |
| --- | --- |
| I edited a file and want the last committed version back | `git restore <file>` |
| I staged something by mistake | `git restore --staged <file>` |
| I want to see a file as it was 3 commits ago | `git show HEAD~3:<path/to/file>` |
| I want to see what changed in one commit | `git show <commit-id>` |

Answer to "Before you start": you need a commit from last Tuesday; then
`git log --until="last tuesday"` finds it and `git restore --source=<id> <file>` brings the file back.

### Good commit messages

A commit message tells future-you *what* changed and makes your history searchable. This course uses:

```text
<type>(<module>.<lesson>): <what you did>
```

| Weak | Strong |
| --- | --- |
| `update` | `exercise(02.03): attempt core exercises 1-6` |
| `fixed stuff` | `fix(02.03): use n-1 for sample variance in ex 4` |
| `notes` | `learn(02.03): notes on why variance squares deviations` |

The full list of types is in [docs/git-workflow.md](../../../docs/git-workflow.md).

### Tags mark milestones

```bash
git tag -a m00-done -m "Module 00 complete"
git tag                       # list tags
git log --oneline --decorate  # tags appear next to commits
```

## 4. Reproducible randomness (a first statistics idea)

Statistics uses random numbers constantly: simulations, random samples, shuffling data. Computers generate
*pseudo-random* numbers from a starting value called a **seed**. Same seed → same numbers → reproducible results.

```python
import numpy as np
rng = np.random.default_rng(seed=7)
print(rng.integers(1, 7, size=5))    # five dice rolls — same five every time you run with seed 7
```

Run `examples.py` twice and check that the "seeded" numbers stay the same while the "unseeded" ones change.
Rule for the whole course: **always create `rng = np.random.default_rng(<seed>)` and use it for all randomness.**

---

## Common mistakes

| Mistake | Symptom | Fix |
| --- | --- | --- |
| Forgetting to activate `.venv` | `ModuleNotFoundError` | `source .venv/bin/activate` |
| `git add .` from the wrong folder | Random files committed | Run `git status` before every commit |
| Committing `.venv/` | Huge repository | It's in `.gitignore`; never force-add it |
| One giant commit per week | History tells no story | Commit after each step of the study loop |

## Summary

- The terminal: `pwd`, `ls`, `cd`, `mkdir`, `cat`.
- A virtual environment pins the exact software; `requirements.txt` records it.
- Git: edit → `git add` (stage) → `git commit` (snapshot). `git status`, `git diff`, `git log` show where you are.
- Seeds make randomness reproducible.

## Self-test (answer from memory next Monday)

1. What is the difference between the staging area and a commit?
2. Why does every project get its own virtual environment?
3. Write the command that shows what changed in your files since the last commit.
4. What does a random seed guarantee?

Next: [exercises.md](exercises.md)
