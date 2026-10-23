# data/

| File | Rows | What it is | Made by |
| --- | --- | --- | --- |
| `students.csv` | 200 | **Synthetic** students: program, year, study hours, sleep, commute, tutoring, exam score | `make_students.py` (seed 42) |

Datasets bundled with scikit-learn (no download needed) are also used in later lessons:
`sklearn.datasets.load_diabetes()`, `load_iris()`, `load_breast_cancer()`, `load_wine()`.

## Rules

- Small files (< 5 MB) that lessons or projects need: commit them here, and add a row to the table above.
- Large or private files: put them in `data/raw/` (git-ignored) and write down in this README where they came from.
- Never edit a raw file by hand. Write a script that cleans it and saves a new file, so every step is reproducible.

## students.csv columns

| Column | Type | Description |
| --- | --- | --- |
| student_id | integer | 1–200 |
| program | categorical (nominal) | Science, Arts, Commerce |
| year | integer (ordinal) | 1st–4th year |
| study_hours | continuous | study hours per week |
| sleep_hours | continuous | average sleep per night |
| commute_minutes | discrete count | one-way commute time |
| tutoring | categorical (binary) | yes / no |
| exam_score | discrete, 0–100 | final exam score |
