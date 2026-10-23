# Shortcuts for the course. Run `make help` to list them.
PY := .venv/bin/python

.PHONY: help setup lesson journal test examples verify

help:
	@echo "make setup                 create .venv, install packages, enable nbstripout"
	@echo "make lesson L=<module>/<lesson>   start a lesson: copy note templates into workspace/"
	@echo "                           e.g. make lesson L=02-descriptive-statistics/03-measures-of-spread"
	@echo "make journal               create this week's journal entry"
	@echo "make test                  run all statkit tests"
	@echo "make test M=descriptive    run tests for one statkit module"
	@echo "make examples              run every curriculum examples.py (smoke check)"
	@echo "make verify                run the tests against the reference solutions (course maintenance)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	.venv/bin/nbstripout --install

lesson:
	@test -n "$(L)" || (echo "Usage: make lesson L=<module>/<lesson>" && exit 1)
	$(PY) scripts/start_lesson.py $(L)

journal:
	$(PY) scripts/new_journal.py

test:
	$(PY) -m pytest -q $(if $(M),tests/test_$(M).py,tests)

examples:
	$(PY) scripts/run_examples.py

verify:
	$(PY) scripts/verify_reference.py
