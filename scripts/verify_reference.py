"""Course-maintenance check: run the test suite against the reference statkit implementations.

It copies every curriculum/*/solutions/statkit_<name>_reference.py into a temporary statkit package and runs pytest
there, so your own statkit/ is never touched.

Run:  python scripts/verify_reference.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "statkit").mkdir()
        (tmp / "statkit" / "__init__.py").write_text("")
        for ref in sorted(ROOT.glob("curriculum/*/solutions/statkit_*_reference.py")):
            name = ref.stem.removeprefix("statkit_").removesuffix("_reference")
            shutil.copy(ref, tmp / "statkit" / f"{name}.py")
        shutil.copytree(ROOT / "tests", tmp / "tests")
        shutil.copytree(ROOT / "data", tmp / "data")
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=tmp).returncode


if __name__ == "__main__":
    sys.exit(main())
