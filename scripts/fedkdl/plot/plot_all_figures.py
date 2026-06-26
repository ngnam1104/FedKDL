"""Generate final paper figures from results/metrics_final."""

from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
script = HERE / "plot_final_metrics.py"
print(f"\n[final] {script.name}")
runpy.run_path(str(script), run_name="__main__")
