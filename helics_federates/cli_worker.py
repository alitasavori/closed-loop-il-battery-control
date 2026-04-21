"""Run one federate from a separate interpreter (used by ``run_cosimulation``).

    python -m helics_federates.cli_worker network tcp://127.0.0.1:24000 28800 --verbose

This avoids HELICS + ``threading`` deadlocks after ``enter_executing_mode()`` and
works from notebooks (``multiprocessing`` spawn often breaks there).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="HELICS single-federate worker")
    p.add_argument(
        "federate",
        choices=("ems", "ems_bc", "droop", "network", "logger"),
        help="which federate to run",
    )
    p.add_argument("broker_address", help="HELICS broker address, e.g. tcp://127.0.0.1:24000")
    p.add_argument("duration_sec", type=float, help="simulation duration [s]")
    p.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Logger JSON path (required for logger federate)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--bc-model-pt", type=Path, default=None, help="Path to BC .pt weights (for ems_bc)")
    p.add_argument(
        "--bc-model-artifacts",
        type=Path,
        default=None,
        help="Path to BC artifacts .joblib (for ems_bc)",
    )
    p.add_argument(
        "--bc-policy-kind",
        type=str,
        default=None,
        help="BC policy kind: lstm_pt or tabular_joblib (for ems_bc)",
    )
    p.add_argument(
        "--bc-model-joblib",
        type=Path,
        default=None,
        help="Path to BC tabular model bundle .joblib (for ems_bc)",
    )
    args = p.parse_args()

    if args.federate == "logger" and args.log_path is None:
        print("cli_worker: --log-path required for logger", file=sys.stderr)
        return 2

    vb = args.verbose

    if args.federate == "ems":
        from helics_federates.ems_federate import run

        run(args.broker_address, args.duration_sec, vb)
    elif args.federate == "ems_bc":
        from helics_federates.bc_ems_federate import run

        run(
            args.broker_address,
            args.duration_sec,
            vb,
            str(args.bc_model_pt) if args.bc_model_pt else None,
            str(args.bc_model_artifacts) if args.bc_model_artifacts else None,
            args.bc_policy_kind or "lstm_pt",
            str(args.bc_model_joblib) if args.bc_model_joblib else None,
        )
    elif args.federate == "droop":
        from helics_federates.droop_federate import run

        run(args.broker_address, args.duration_sec, vb)
    elif args.federate == "network":
        from helics_federates.network_federate import run

        run(args.broker_address, args.duration_sec, vb)
    else:
        from helics_federates.logger_federate import run

        assert args.log_path is not None
        run(args.broker_address, args.log_path, args.duration_sec, vb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
