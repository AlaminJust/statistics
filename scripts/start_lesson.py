"""Start a lesson: create workspace/<module>/<lesson>/ with notes and exercise templates.

Usage:
    python scripts/start_lesson.py 02-descriptive-statistics/03-measures-of-spread

Existing files are never overwritten, so it is safe to run twice.
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    lesson = sys.argv[1].strip("/")
    source = ROOT / "curriculum" / lesson
    if not (source / "lesson.md").exists():
        print(f"No lesson found at curriculum/{lesson}/lesson.md")
        print("Available lessons:")
        for path in sorted((ROOT / "curriculum").glob("*/*/lesson.md")):
            print("  ", path.parent.relative_to(ROOT / "curriculum"))
        return 1

    target = ROOT / "workspace" / lesson
    target.mkdir(parents=True, exist_ok=True)

    replacements = {"{{LESSON}}": lesson, "{{DATE}}": date.today().isoformat()}
    for template_name, out_name in [("lesson-notes.md", "notes.md"), ("lesson-exercises.py", "exercises.py")]:
        out = target / out_name
        if out.exists():
            print(f"exists, kept: {out.relative_to(ROOT)}")
            continue
        text = (TEMPLATES / template_name).read_text()
        for key, value in replacements.items():
            text = text.replace(key, value)
        out.write_text(text)
        print(f"created: {out.relative_to(ROOT)}")

    print(f"\nRead:  curriculum/{lesson}/lesson.md")
    print(f"Run:   python curriculum/{lesson}/examples.py")
    module, name = lesson.split("/")
    print(f"Then:  git add workspace/{lesson} && git commit -m \"learn({module[:2]}.{name[:2]}): start {name}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
