"""Create this week's journal entry, journal/YYYY-Www.md, from templates/weekly-journal.md."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    today = date.today()
    year, week, _ = today.isocalendar()
    out = ROOT / "journal" / f"{year}-W{week:02d}.md"
    if out.exists():
        print(f"exists, kept: {out.relative_to(ROOT)}")
        return
    text = (ROOT / "templates" / "weekly-journal.md").read_text()
    text = text.replace("{{WEEK}}", f"{year}-W{week:02d}").replace("{{DATE}}", today.isoformat())
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(f"created: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
