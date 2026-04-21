"""Quick HELICS + OpenDSS co-simulation smoke test (run from project root).

    python -m helics_federates.smoke_test_cosim

Uses 600 s (two 5-min steps). Exits 0 on success.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python helics_federates/smoke_test_cosim.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from helics_federates.run_cosimulation import run_cosimulation  # noqa: E402


def main() -> int:
    out = run_cosimulation(sim_duration_sec=600, verbose=True)
    if not out.exists() or out.stat().st_size < 50:
        print("FAIL: log missing or empty", out, file=sys.stderr)
        return 1
    print("OK:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
