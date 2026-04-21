"""Start HELICS broker + four federates, return path to logger JSON.

Federates are started as **separate Python processes** via ``python -m helics_federates.cli_worker``.
Running multiple HELICS value federates in **one process with threads** commonly deadlocks at the
first ``request_time()`` after ``enter_executing_mode()``; separate processes matches typical
HELICS usage and works from Jupyter (``multiprocessing`` spawn is often broken there).
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

import helics as h

from . import common
from . import network_federate


def run_cosimulation(
    sim_duration_sec: float | None = None,
    log_path: Path | None = None,
    verbose: bool = False,
    ems_mode: str = "lp",
    bc_model_pt: Path | None = None,
    bc_model_artifacts: Path | None = None,
    bc_policy_kind: str = "lstm_pt",
    bc_model_joblib: Path | None = None,
) -> Path:
    duration = sim_duration_sec if sim_duration_sec is not None else common.DEFAULT_SIM_DURATION_SEC
    out = log_path or (Path(__file__).resolve().parent / "last_run_log.json")
    vb = common.cosim_verbose(verbose)
    n_steps = int(round(duration / common.DELTA_T_SEC))

    project_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    ems_worker = "ems"
    if ems_mode == "bc_pt":
        ems_worker = "ems_bc"
        if bc_model_pt is None or bc_model_artifacts is None:
            raise ValueError("ems_mode='bc_pt' requires bc_model_pt and bc_model_artifacts paths.")
    elif ems_mode == "bc_tabular":
        ems_worker = "ems_bc"
        if bc_model_joblib is None:
            raise ValueError("ems_mode='bc_tabular' requires bc_model_joblib path.")
    elif ems_mode != "lp":
        raise ValueError("ems_mode must be one of {'lp', 'bc_pt', 'bc_tabular'}.")

    def _cmd(federate: str) -> list[str]:
        cmd: list[str] = [
            py,
            "-m",
            "helics_federates.cli_worker",
            federate,
            addr,
            str(duration),
        ]
        if federate == "logger":
            cmd.extend(["--log-path", str(out.resolve())])
        if federate == "ems_bc":
            if ems_mode == "bc_pt":
                cmd.extend(
                    [
                        "--bc-model-pt",
                        str(bc_model_pt.resolve()),
                        "--bc-model-artifacts",
                        str(bc_model_artifacts.resolve()),
                        "--bc-policy-kind",
                        "lstm_pt",
                    ]
                )
            elif ems_mode == "bc_tabular":
                cmd.extend(
                    [
                        "--bc-policy-kind",
                        "tabular_joblib",
                        "--bc-model-joblib",
                        str(bc_model_joblib.resolve()),
                    ]
                )
        if vb:
            cmd.append("--verbose")
        return cmd

    # Same order as before: Network / Droop / Logger / EMS
    order = ("network", "droop", "logger", ems_worker)
    max_start_attempts = 3
    last_errors: list[str] = []
    logs_dir = Path(__file__).resolve().parent / "cosim_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_start_attempts + 1):
        port = random.randint(23700, 24900)
        broker = h.helicsCreateBroker("zmq", "", f"--federates=4 -p {port}")
        addr = broker.address

        network_federate.reset_circuit_built_flag()
        network_federate.ensure_ieee34_bess_circuit_built(vb)

        common.cosim_print(
            vb,
            f"[cosim] duration={duration}s (~{duration/3600:.2f}h)  steps~{n_steps}  "
            f"broker={addr}  verbose={vb}  attempt={attempt}/{max_start_attempts}  "
            "(set HELICS_COSIM_VERBOSE=1 or verbose=True)",
        )
        common.cosim_print(
            vb,
            "[cosim] Federates run as separate Python processes (see helics_federates.cli_worker).",
        )
        common.cosim_print(
            vb,
            "[cosim] Runtime scales ~linearly with steps (OpenDSS solve each 5 min). "
            "There is no special slowdown at hour 10 vs hour 11.",
        )

        procs: list[subprocess.Popen] = []
        log_files: list[TextIO] = []
        federate_logs: dict[str, Path] = {}
        for name in order:
            log_path = logs_dir / f"attempt_{attempt}_{name}.log"
            fh = open(log_path, "w", encoding="utf-8")
            log_files.append(fh)
            federate_logs[name] = log_path
            procs.append(
                subprocess.Popen(
                    _cmd(name),
                    cwd=str(project_root),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
            )
            time.sleep(0.05)

        errors: list[str] = []
        try:
            for name, proc in zip(order, procs, strict=True):
                rc = proc.wait()
                if rc != 0:
                    errors.append(f"{name} exit code {rc}")
        finally:
            for fh in log_files:
                fh.close()

        if not errors:
            common.cosim_print(vb, "[cosim] all federates finished OK")
            time.sleep(0.15)
            broker.disconnect()
            return out

        last_errors = errors
        tail_bits: list[str] = []
        for err in errors:
            fed = err.split(" exit code ")[0]
            p = federate_logs.get(fed)
            if p is None or not p.exists():
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
                lines = txt.splitlines()
                tail = "\n".join(lines[-30:])
                tail_bits.append(f"[{fed} log: {p}]\n{tail}")
            except OSError:
                continue
        broker.disconnect()
        time.sleep(0.2)
        if tail_bits:
            common.cosim_print(
                vb,
                "[cosim] attempt failed; tail logs follow:\n" + "\n\n".join(tail_bits),
            )

    raise RuntimeError(
        "Federate subprocess failure(s) after startup retries: "
        + "; ".join(last_errors)
        + f". Check logs in {logs_dir}"
    )
