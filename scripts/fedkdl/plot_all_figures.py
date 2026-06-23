"""Generate all thesis figures, skipping only missing real learning inputs."""

from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for idx, script in enumerate(sorted(HERE.glob("K*.py")), start=1):
    print(f"\n[{idx}] {script.name}")
    try:
        runpy.run_path(str(script), run_name="__main__")
    except FileNotFoundError as exc:
        print(f"SKIP: {exc}")
