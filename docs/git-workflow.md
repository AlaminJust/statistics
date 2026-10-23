# Git workflow — tracking every piece of my work

The goal: anyone (including future me) can open `git log` and see exactly what I studied, when, what I got wrong,
and how I fixed it.

## What gets committed

| Commit it | Never commit it |
| --- | --- |
| `workspace/` notes and exercise answers | `.venv/` (re-created by `make setup`) |
| `statkit/` code and my extra tests | Large or private data (put it in `data/raw/`, which is ignored) |
| `projects/` code and reports | Passwords, API keys, tokens |
| `journal/` weekly logs, `PROGRESS.md` | Notebook outputs (nbstripout removes them automatically) |

## When to commit (about 4–8 commits per lesson)

1. After `make lesson L=...` — starting the lesson.
2. After writing `notes.md`.
3. After the **first attempt** of the exercises — *before* opening solutions.
4. After correcting mistakes — a separate commit, so the fix is visible.
5. Each time a `statkit` function passes its tests.
6. End of the week: journal + `PROGRESS.md`, then `git push`.

## Commit message format

```text
<type>(<module>.<lesson>): <what you did, in the present tense>
```

| Type | Use for | Example |
| --- | --- | --- |
| `learn` | notes, summaries | `learn(02.03): notes on variance and why we divide by n-1` |
| `exercise` | first attempts | `exercise(02.03): attempt core exercises 1-6` |
| `fix` | corrections after checking solutions | `fix(02.03): correct sample vs population variance in ex 4` |
| `statkit` | library code | `statkit(02.03): implement variance and std with ddof` |
| `project` | mini-projects and capstone | `project(m02): EDA report on diabetes dataset` |
| `journal` | weekly log | `journal: week 2026-W44` |
| `progress` | ticking PROGRESS.md | `progress: finish 02.03 measures of spread` |
| `chore` | setup, tooling | `chore: add seaborn to requirements` |
| `course` | new or updated course material in `curriculum/` (not your own work) | `course(03): add probability lessons 03.01-03.06` |

## The daily commands

```bash
git status                       # what changed?
git add workspace/02-descriptive-statistics/03-measures-of-spread
git commit -m "exercise(02.03): attempt core exercises 1-6"
git push                         # at the end of every session
```

## Milestones with tags

When a module checkpoint is passed:

```bash
git tag -a m02-done -m "Module 02 descriptive statistics complete"
git push --tags
```

## Branches (only for projects)

Daily lesson work goes straight to `main`. Use a branch for each mini-project so you can experiment freely:

```bash
git switch -c project/m02-eda
# ... work and commit ...
git switch main
git merge --no-ff project/m02-eda -m "project(m02): merge EDA report"
```

## Review your own progress

```bash
git log --oneline --graph --decorate              # the whole story
git log --oneline -- workspace/02-*               # everything I did in Module 02
git log --since="1 week ago" --oneline            # this week's work
git diff m01-done m02-done --stat                 # what changed during Module 02
git shortlog -s --since="1 month ago"             # commits this month
git log --grep="^fix" --oneline                   # every mistake I corrected
```
