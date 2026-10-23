"""Run every curriculum/*/*/examples.py and report which ones fail (a smoke check)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    scripts = sorted((ROOT / "curriculum").glob("*/*/examples.py"))
    failed = []
    for script in scripts:
        name = script.parent.relative_to(ROOT / "curriculum")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
        status = "ok  " if result.returncode == 0 else "FAIL"
        print(f"{status} {name}")
        if result.returncode != 0:
            failed.append(name)
            print(result.stderr[-2000:])
    print(f"\n{len(scripts) - len(failed)}/{len(scripts)} examples ran cleanly")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
