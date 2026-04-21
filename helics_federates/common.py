"""Shared constants and helpers for HELICS BESS co-simulation.

Units (for EMS objective and logs)
----------------------------------
- Price in ``prices_hourly.csv``: treated as **$/kWh**
  (or any consistent monetary unit per kWh). EMS maximizes ``sum(pi_i * P_i)``
  with **P_i in kW** and **1 h** per stage → **revenue-like units = $/h per stage**
  when pi is $/kWh.
- Power: **kW**.
- SOC: **kWh** (device state; integrated **only** in the Droop federate).

Sign convention (enforced across EMS, Droop, OpenDSS Generator)
---------------------------------------------------------------
- **P > 0** → active power **injected to the grid** (battery **discharging**).
- **P < 0** → active power **drawn from the grid** (battery **charging**).
- OpenDSS ``Generator`` **kW > 0** = generation into the network → matches **P_act**.

Rolling horizon (EMS)
---------------------
- **Fixed 24-hour window** (non-shrinking): at each EMS update, the LP always
  optimizes **P_0..P_23** with a full 24-step forecast. Only **P_0** is executed;
  the window **re-rolls** at the next clock hour with updated SOC (MPC-style).
- Optional **EMS_PRICE_NOISE_STD**: Gaussian noise on the rolling horizon tail
  (see ``lp_price_horizon_with_suffix_noise``): one new perturbation is added
  each hour for the newly introduced far-horizon slot; no daily reset.

Temporal discretization (EMS vs Droop) — explicit assumption
-------------------------------------------------------------
- The **LP** uses **1-hour** SOC dynamics between decision stages **P_i**.
- The **Droop** federate integrates SOC every **5 minutes** with the **same** η.
- **SOC bounds** in the LP are enforced only on **hourly** implied states, **not**
  on intra-hour SOC from the 5-minute integrator. This is an intentional
  modeling approximation (possible small intra-hour bound slack).

EMS observation of SOC (ML / BC interpretation)
------------------------------------------------
- The EMS reads **SOC** from the Droop publication **only when it re-solves**
  (at clock **hour** changes). It does **not** track intra-hour SOC from the
  5-minute integrator. Treat this as **discretized state at decision times only**,
  not as continuous-time full-state feedback optimal control.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any


def cosim_verbose(flag: bool) -> bool:
    """True if user passed verbose=True or env HELICS_COSIM_VERBOSE is 1/true/yes/debug."""
    if flag:
        return True
    v = os.environ.get("HELICS_COSIM_VERBOSE", "").strip().lower()
    return v in ("1", "true", "yes", "debug")


def cosim_trace_enabled(flag: bool) -> bool:
    """Whether to print request_time / Solve boundary lines. HELICS_COSIM_TRACE=1 or verbose."""
    if cosim_verbose(flag):
        return True
    v = os.environ.get("HELICS_COSIM_TRACE", "").strip().lower()
    return v in ("1", "true", "yes", "debug")


def cosim_print(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def configure_federate_info_thread_safe(fedinfo: object) -> None:
    """Enable thread-safe HELICS cores when multiple federates run in one process (``threading``).

    Default federate cores are not safe for concurrent use from several threads; without
    ``HELICS_FLAG_MULTI_THREAD_CORE``, co-simulation often **deadlocks** on the first
    ``request_time()`` after ``enter_executing_mode()`` (no per-step logs, run never ends).
    """
    import helics as h

    h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_MULTI_THREAD_CORE, True)


def configure_federate_index_group(fedinfo: object, group: int) -> None:
    """Order federates at the same time grant: **lower** ``group`` runs earlier.

    Without this, parallel federates can run in arbitrary order; Droop may read ``P_ref``
    before EMS publishes the new hour's reference (one-step lag vs ``P_act`` / logger rows).
    Use: EMS=0, Droop=1, Network=2, Logger=3.
    """
    import helics as h

    h.helicsFederateInfoSetIntegerProperty(fedinfo, h.HELICS_PROPERTY_INT_INDEX_GROUP, int(group))


# --- time ---
DELTA_T_SEC = 300.0  # 5 minutes
STEPS_PER_HOUR = 12
SECONDS_PER_HOUR = 3600.0

# Default: one hour of 5-minute steps (12 points) for interactive runs; use 86400 for a full day
DEFAULT_SIM_DURATION_SEC = 3600

# Voltage tap for feedback: OpenDSS bus.node (e.g. phase 1 → ".1"). Must match the BESS connection point.
BESS_BUS_NODE = "830.3"

# --- BESS / EMS (physical units) ---
SOC_INIT_KWH = 1000.0
SOC_MIN_KWH = 200.0
SOC_MAX_KWH = 1800.0
P_MAX_KW = 500.0
ETA_CH = 0.95  # efficiency applied to active power in SOC update (simple scalar)

# Sentinel from HELICS before first publication
INVALID_HELICS_DOUBLE = -1e49


def safe_double(x: float, default: float) -> float:
    if x is None or x < -1e40 or x > 1e40 or x != x:
        return default
    return float(x)


def project_root() -> Path:
    """helics_federates/ -> project root."""
    return Path(__file__).resolve().parent.parent


def ieee34_dir() -> Path:
    d = project_root() / "IEEE34busfile"
    if not (d / "IEEE34_PV.dss").exists():
        raise FileNotFoundError(f"Missing IEEE34_PV.dss under {d}")
    return d


def parse_bus_node_selector(s: str) -> tuple[str, int]:
    """Return (OpenDSS bus selector, phase index) for PuVoltage magnitude list.

    For example 830.3 → selector '830.3' and index 2; plain 830 → ('830', 0).
    """
    if "." in s:
        head, tail = s.rsplit(".", 1)
        if tail.isdigit() and head:
            return s, int(tail) - 1
    return s, 0


def load_hourly_price_profile() -> list[float]:
    """Load 24 hourly $/kWh values from ``prices_hourly.csv`` (no averaging)."""
    csv_path = ieee34_dir() / "prices_hourly.csv"
    hourly: list[float | None] = [None] * 24
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                h = int(float(parts[0]))
                p = float(parts[1])
            except ValueError:
                continue
            if 0 <= h <= 23:
                hourly[h] = p
    missing = [i for i, v in enumerate(hourly) if v is None]
    if missing:
        raise ValueError(f"prices_hourly.csv: missing hours {missing}")
    return [float(x) for x in hourly]


def rolling_forecast_24(hourly24: list[float], start_hour: int) -> list[float]:
    """Return **24** hourly prices for a **non-shrinking** horizon: indices ``start_hour .. start_hour+23`` (wrap mod 24)."""
    out: list[float] = []
    for i in range(24):
        out.append(hourly24[(start_hour + i) % 24])
    return out


def ems_price_noise_std() -> float:
    """RMS $/kWh for Gaussian noise on the rolling suffix (``EMS_PRICE_NOISE_STD`` env, default 0)."""
    v = os.environ.get("EMS_PRICE_NOISE_STD", "").strip()
    if not v:
        return 0.0
    try:
        return max(0.0, float(v))
    except ValueError:
        return 0.0


def ems_price_noise_seed() -> int | None:
    """Optional RNG seed for repeatable price noise (``EMS_PRICE_NOISE_SEED``)."""
    v = os.environ.get("EMS_PRICE_NOISE_SEED", "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def lp_price_horizon_with_suffix_noise(
    hourly24: list[float],
    sim_clock_hour: int,
    rng: random.Random,
    std: float,
    epsilon_state: dict[int, float],
) -> tuple[list[float], list[float]]:
    """Return ``(pi_csv, pi_lp)`` for the EMS objective.

    * ``pi_csv`` is always ``rolling_forecast_24(hourly24, sim_clock_hour % 24)``.

    * Absolute-hour formulation: stage ``i`` in the LP horizon corresponds to absolute
      hour ``a = sim_clock_hour + i``.
      - If ``a < 24`` (first simulation day), slot stays unperturbed.
      - If ``a >= 24``, slot is ``hourly24[a % 24] + ε_a``.
        ``ε_a`` is drawn once (first time that absolute hour appears in a horizon)
        and then fixed for all later solves.

    **No reset:** ``epsilon_state`` persists through the whole simulation; there is no
    day-boundary clearing. Perturbations always use the CSV base ``hourly24[a % 24]``
    (never previously perturbed LP values), so noise does not compound.

    If ``std <= 0``, ``pi_lp`` matches ``pi_csv``.
    """
    r = sim_clock_hour % 24
    pi_csv = rolling_forecast_24(hourly24, r)
    if std <= 0:
        return list(pi_csv), list(pi_csv)

    pi_lp = list(pi_csv)
    for i in range(24):
        abs_hour = sim_clock_hour + i
        if abs_hour < 24:
            continue
        if abs_hour not in epsilon_state:
            epsilon_state[abs_hour] = rng.gauss(0.0, std)
        pi_lp[i] = hourly24[abs_hour % 24] + epsilon_state[abs_hour]
    return list(pi_csv), pi_lp


def write_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
