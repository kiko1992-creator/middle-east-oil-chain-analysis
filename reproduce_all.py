"""
Full pipeline reproduction script.

Runs every data pipeline and model step in the correct dependency order,
measures wall-clock time for each step, and prints a summary.  Exits with
code 1 if any step fails, after completing all remaining steps.

Usage (from project root):
    python reproduce_all.py
    python reproduce_all.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent


class Step(NamedTuple):
    label: str
    module: str
    args: list[str] = []


_STEPS: list[Step] = [
    Step(" 1. Fetch World Bank panel",        "src.data.fetch_world_bank"),
    Step(" 2. Fetch IMF WEO panel",           "src.data.fetch_imf_weo"),
    Step(" 3. Clean World Bank panel",        "src.data.clean_world_bank"),
    Step(" 4. OCVI vulnerability index",      "src.model.vulnerability_index"),
    Step(" 5. Chain transmission severity",   "src.model.chain_transmission", ["--fit-ols"]),
    Step(" 6. Historical risk index 2015-2024","src.model.historical_index"),
    Step(" 7. Right Now Risk composite",      "src.model.right_now_risk"),
    Step(" 8. 2020 oil crash retrospective",  "src.model.retrospective"),
    Step(" 9. IMF/WB cross-validation",       "src.model.cross_validation"),
    Step("10. Sensitivity analysis (OAT)",    "src.model.sensitivity"),
    Step("11. Validate reference data",       "src.data.validate_reference", ["--strict"]),
]


def _run_step(step: Step, python: str) -> tuple[bool, float]:
    """Run one pipeline step and return (success, elapsed_seconds)."""
    cmd = [python, "-m", step.module] + step.args
    log.debug("Running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=_ROOT, capture_output=False)
    elapsed = time.perf_counter() - t0
    return result.returncode == 0, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python reproduce_all.py",
        description="Reproduce all pipeline outputs from scratch.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    python = sys.executable

    print(f"\n{'=' * 62}")
    print("  MENA Oil Chain Analysis — Full Pipeline Reproduction")
    print(f"  Python: {python}")
    print(f"  Root:   {_ROOT}")
    print(f"{'=' * 62}\n")

    results: list[tuple[str, bool, float]] = []
    total_start = time.perf_counter()

    for step in _STEPS:
        print(f">>  {step.label} ...", flush=True)
        ok, elapsed = _run_step(step, python)
        status = "OK " if ok else "FAIL"
        print(f"   {status}  ({elapsed:.1f}s)\n", flush=True)
        results.append((step.label, ok, elapsed))

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print(f"{'=' * 62}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for label, ok, elapsed in results:
        sym = "PASS" if ok else "FAIL"
        print(f"  {sym}  {label:<44}  {elapsed:>6.1f}s")
    print(f"{'=' * 62}")
    print(f"  TOTAL: {n_pass} PASS  {n_fail} FAIL  ({total_elapsed:.1f}s)\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
