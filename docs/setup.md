# Setup

You need this once, at the start of the course (Lesson 00.01 walks through it with explanations).

## 1. Tools

| Tool | Check it is installed | Install if missing (macOS) |
| --- | --- | --- |
| Python ≥ 3.11 | `python3 --version` | [python.org/downloads](https://www.python.org/downloads/) or `brew install python` |
| git | `git --version` | `xcode-select --install` |
| make | `make --version` | comes with `xcode-select --install` |
| VS Code (editor) | — | [code.visualstudio.com](https://code.visualstudio.com/) + the Python and Jupyter extensions |

## 2. Create the environment

```bash
cd ~/MyProject/statistics
make setup                    # creates .venv/ and installs requirements.txt
source .venv/bin/activate     # do this every time you open a new terminal
python -c "import numpy, pandas, scipy; print('ready')"
```

In VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** → choose `./.venv/bin/python`.

## 3. Notebooks (optional)

Lessons use plain `.py` scripts because they diff cleanly in git. If you prefer notebooks for your own exploration:

```bash
jupyter lab
```

`make setup` already ran `nbstripout --install`, which removes notebook outputs before each commit so your git
history stays small and readable.

## 4. Git identity (check once)

```bash
git config user.name     # should print your name
git config user.email    # should print the email you use on GitHub
```

## 5. Back up to GitHub (recommended)

Create a **private** repository and push (needs the GitHub CLI, `brew install gh`, then `gh auth login`):

```bash
gh repo create statistics-from-scratch --private --source=. --push
```

After that, `git push` at the end of every study session. See [git-workflow.md](git-workflow.md).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'numpy'` | You forgot `source .venv/bin/activate` |
| `make: command not found` | Run `xcode-select --install` |
| Figures don't appear | Examples save figures to `figures/generated/` next to the script; open them from there |
| pip fails on a package | Run `python3 --version`; if older than 3.11, install a newer Python and rerun `make setup` |
