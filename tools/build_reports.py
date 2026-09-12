import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lap_estimation.reporting import write_reports


def main():
    written = write_reports(ROOT)
    print(f"Wrote {len(written)} report files to {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
