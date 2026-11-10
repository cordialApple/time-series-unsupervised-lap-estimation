import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lap_estimation.datasets import DATASET_DEFINITIONS
from lap_estimation.plotting_physics import render_course_runs, render_null_calibration
from lap_estimation.runners import run_course_alignment, run_physics_laps


def main():
    for dataset_key in DATASET_DEFINITIONS:
        result = run_physics_laps(ROOT, dataset_key=dataset_key)
        print(f"Wrote {len(result.predictions)} period estimates to {result.output_dir}")
    result = run_course_alignment(ROOT)
    print(f"Wrote {len(result.predictions)} course runs to {result.output_dir}")
    print(f"Wrote {render_course_runs(result.output_dir)}")
    print(f"Wrote {render_null_calibration(ROOT / 'artifacts', ROOT / 'artifacts' / 'physics_null_calibration.png')}")


if __name__ == "__main__":
    main()
