# Statistics from Scratch → AI, ML & Research

A complete, self-paced statistics course I am working through from zero to research level, with every note,
exercise, project and line of code tracked in git.

- **Goal:** read ML papers comfortably, build models I can defend, and design experiments that survive peer review.
- **Pace:** about 8 hours a week, 34 weeks (see [PLAN.md](PLAN.md)).
- **Language:** Python (NumPy, pandas, SciPy, matplotlib, statsmodels, scikit-learn; PyMC from Module 07).

## How this repository works

There are three kinds of content, kept apart so that git history shows **my** progress clearly:

| Folder | Who writes it | What it holds |
| --- | --- | --- |
| `curriculum/` | The course (read-only for me) | Lessons, worked examples, exercises, reference solutions |
| `workspace/` | **Me** | My notes, my exercise answers, my experiments — mirrors `curriculum/` |
| `statkit/` | **Me** | My own statistics library, built from scratch one function at a time, checked by `tests/` |
| `projects/` | **Me** | One mini-project per module plus the capstone |
| `journal/` | **Me** | One short log per week: what I studied, what was hard, what's next |

## Folder architecture

```text
statistics/
├── README.md                 ← you are here
├── PLAN.md                   ← week-by-week plan for all 34 weeks
├── PROGRESS.md               ← checklist I tick as I finish lessons (committed every week)
├── Makefile                  ← shortcuts: make setup / lesson / journal / test / examples
├── requirements.txt          ← exact package versions
├── .gitignore  .gitattributes
│
├── docs/                     ← how to use this course
│   ├── setup.md              ← install Python, venv, Jupyter, git
│   ├── study-method.md       ← how to study a lesson (the 6-step loop)
│   ├── git-workflow.md       ← commit rules, message format, tags, branches
│   └── resources.md          ← free books, courses and videos by module
│
├── curriculum/               ← THE COURSE
│   ├── 00-setup/
│   ├── 01-math-toolkit/
│   ├── 02-descriptive-statistics/
│   ├── 03-probability/
│   ├── 04-distributions/
│   ├── 05-inference/
│   ├── 06-regression/
│   ├── 07-bayesian/
│   ├── 08-statistical-learning/
│   ├── 09-causal-and-research/
│   └── 10-capstone/
│       Each module:
│       ├── README.md                  ← goals, lesson list, checkpoint, mini-project
│       └── NN-lesson-name/
│           ├── lesson.md              ← explanation + worked examples by hand
│           ├── examples.py            ← runs every worked example in code
│           ├── exercises.md           ← warm-up, core and challenge exercises
│           └── solutions/             ← reference answers (try first!)
│
├── workspace/                ← MY WORK (mirrors curriculum/)
│   └── 02-descriptive-statistics/03-measures-of-spread/
│       ├── notes.md                   ← my notes in my own words
│       └── exercises.py               ← my answers
│
├── statkit/                  ← MY LIBRARY: functions I implement from scratch
│   ├── descriptive.py
│   └── ...
├── tests/                    ← pytest checks for statkit (they fail until I implement)
│
├── projects/                 ← one mini-project per module + capstone
├── journal/                  ← weekly logs, e.g. 2026-W44.md
├── data/                     ← small datasets (large raw files are git-ignored)
├── templates/                ← lesson notes, journal, project report
└── scripts/                  ← helpers used by the Makefile
```

## Quick start

```bash
cd ~/MyProject/statistics
make setup                                   # create .venv and install packages (once)
source .venv/bin/activate
make lesson L=00-setup/01-environment-and-git  # copy note templates into workspace/
python curriculum/00-setup/01-environment-and-git/examples.py
make journal                                 # start this week's journal entry
```

Then follow the loop in [docs/study-method.md](docs/study-method.md) and commit as described in
[docs/git-workflow.md](docs/git-workflow.md).

## Course map

| # | Module | Weeks | Status of the written content |
| --- | --- | --- | --- |
| 00 | [Setup and study system](curriculum/00-setup/README.md) | 0 | Complete |
| 01 | [Math toolkit](curriculum/01-math-toolkit/README.md) | 1–2 | Complete |
| 02 | [Descriptive statistics](curriculum/02-descriptive-statistics/README.md) | 3–5 | Complete |
| 03 | [Probability](curriculum/03-probability/README.md) | 6–8 | Complete |
| 04 | [Distributions](curriculum/04-distributions/README.md) | 9–11 | Outline |
| 05 | [Statistical inference](curriculum/05-inference/README.md) | 12–15 | Outline |
| 06 | [Regression and GLMs](curriculum/06-regression/README.md) | 16–19 | Outline |
| 07 | [Bayesian statistics](curriculum/07-bayesian/README.md) | 20–23 | Outline |
| 08 | [Statistical learning and ML evaluation](curriculum/08-statistical-learning/README.md) | 24–27 | Outline |
| 09 | [Experiments, causal inference and research methods](curriculum/09-causal-and-research/README.md) | 28–31 | Outline |
| 10 | [Capstone](curriculum/10-capstone/README.md) | 32–34 | Outline |
