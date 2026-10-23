# Module 00 — Setup and study system

**Week 0 · about 6 hours**

Before learning statistics you need three things working: a Python environment, git, and a study routine.
This module sets up all three so that from Module 01 onwards you only think about statistics.

## Lessons

| # | Lesson | You will be able to |
| --- | --- | --- |
| 00.01 | [Environment and git](01-environment-and-git/lesson.md) | Use the terminal, run Python in a virtual environment, and track work with git |
| 00.02 | [Python for data](02-python-for-data/lesson.md) | Work with NumPy arrays, pandas DataFrames and matplotlib plots |

## statkit functions for this module

`statkit/basics.py`: `count`, `total`, `minimum`, `maximum` — written with plain loops, no built-ins.
Test with `make test M=basics`.

## Checkpoint (do this without notes)

1. Create a new file, commit it, change it, view the diff, and commit again with a correct message.
2. Load `data/students.csv`, print the average `exam_score` per `program`, and save a histogram of `study_hours`.
3. `make test M=basics` passes.

When all three work: `git tag -a m00-done -m "Module 00 complete"`.
