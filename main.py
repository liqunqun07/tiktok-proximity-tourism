"""Run the full analysis pipeline in order.

    python main.py               # full pipeline (about 2 minutes)
    python main.py --skip-clean  # reuse data/processed/02_features.csv

Stage 00 (raw API collection) is a reference script and is not run.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"
STAGES = [
    ("01_clean_data.py", "clean raw scrape"),
    ("02_feature_engineering.py", "feature engineering"),
    ("05_coverage.py", "sample composition (Appendix Tables A3-A4)"),
    ("10_nb_funnel.py", "descriptives + NB funnel models (H1-H4)"),
    ("15_engagement_types.py", "affective vs instrumental conversion (H5)"),
    ("20_robustness.py", "robustness of funnel estimates"),
    ("25_complementarity.py", "triad complementarity (H6)"),
    ("30_qca.py", "fsQCA configurations (H7)"),
    ("52_table_cue_stage_map.py", "cue-stage map (Appendix Table A5)"),
    ("50_fig_diagrams.py", "framework and method-flow diagrams"),
    ("51_fig_results.py", "result figures"),
]


def main() -> None:
    skip_clean = "--skip-clean" in sys.argv
    for script, label in STAGES:
        if skip_clean and script.startswith(("01_", "02_")):
            continue
        if not (CODE / script).exists():
            sys.exit(f"Missing stage script: code/{script}")
        print(f"\n=== {script}: {label}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, script], cwd=CODE, check=True)
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
