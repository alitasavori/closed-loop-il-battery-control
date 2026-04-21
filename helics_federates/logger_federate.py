"""HELICS federate: subscribe to all signals and write JSON for offline plots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import helics as h

from . import common
from .common import (
    BESS_BUS_NODE,
    DEFAULT_SIM_DURATION_SEC,
    DELTA_T_SEC,
    SOC_INIT_KWH,
    cosim_print,
    safe_double,
    write_log,
)


def _parse_ems_lp_snapshot(
    sub: object,
) -> tuple[
    float | None,
    list[float] | None,
    bool | None,
    int | None,
    list[float] | None,
    list[float] | None,
]:
    """Return EMS JSON fields: LP wall time, P horizon, success, forecast origin, CSV/LP price horizons."""
    try:
        raw = sub.string  # type: ignore[attr-defined]
    except Exception:
        raw = ""
    if not (isinstance(raw, str) and raw.strip()):
        return None, None, None, None, None, None
    try:
        o = json.loads(raw)
        ph = o.get("P_horizon_kW")
        if ph is not None:
            ph = [float(x) for x in ph]
        pc = o.get("price_horizon_csv")
        if pc is not None:
            pc = [float(x) for x in pc]
        plp = o.get("price_horizon_lp")
        if plp is not None:
            plp = [float(x) for x in plp]
        return (
            float(o["lp_solve_wall_s"]) if "lp_solve_wall_s" in o else None,
            ph,
            bool(o["lp_success"]) if "lp_success" in o else None,
            int(o["forecast_origin_mod24"]) if "forecast_origin_mod24" in o else None,
            pc,
            plp,
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None, None, None, None, None, None


def run(
    broker_address: str,
    out_path: Path,
    sim_duration_sec: float = DEFAULT_SIM_DURATION_SEC,
    verbose: bool = False,
) -> None:
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"logger_core_{os.getpid()}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(
        fedinfo, f"--federates=1 --broker_address={broker_address}"
    )
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, DELTA_T_SEC)
    common.configure_federate_info_thread_safe(fedinfo)
    common.configure_federate_index_group(fedinfo, 3)

    fed = h.helicsCreateValueFederate("Logger", fedinfo)
    s_pref = fed.register_subscription("P_ref", "kW")
    s_pact = fed.register_subscription("P_act", "kW")
    s_soc = fed.register_subscription("SOC", "kWh")
    s_v = fed.register_subscription("V_node_pu", "pu")
    s_lp = fed.register_subscription("EMS_LP_snapshot", "")
    cosim_print(verbose, "[Logger] HELICS federate created; calling enter_initializing_mode() ...")

    # Must match other federates: HELICS blocks enter_initializing_mode() until all
    # federates call it; skipping this caused subscribe-only Logger to deadlock startup.
    fed.enter_initializing_mode()
    fed.enter_executing_mode()
    cosim_print(verbose, "[Logger] enter_executing_mode() OK - starting time steps")

    times_s: list[float] = []
    p_ref: list[float] = []
    p_act: list[float] = []
    soc: list[float] = []
    v_node: list[float] = []
    lp_solve_wall_s: list[float | None] = []
    P_horizon_kW: list[list[float] | None] = []
    lp_success: list[bool | None] = []
    ems_snapshot_forecast_origin_mod24: list[int | None] = []
    price_horizon_csv: list[list[float] | None] = []
    price_horizon_lp: list[list[float] | None] = []

    n_steps = int(round(sim_duration_sec / DELTA_T_SEC))
    step_i = 0
    t_next = DELTA_T_SEC
    while t_next <= sim_duration_sec + 1e-6:
        granted = fed.request_time(t_next)
        step_i += 1

        times_s.append(float(granted))
        p_ref.append(safe_double(s_pref.value, 0.0))
        p_act.append(safe_double(s_pact.value, 0.0))
        soc.append(safe_double(s_soc.value, SOC_INIT_KWH))
        v_node.append(safe_double(s_v.value, 1.0))
        w, ph, ok, fo, pc, plp = _parse_ems_lp_snapshot(s_lp)
        lp_solve_wall_s.append(w)
        P_horizon_kW.append(ph)
        lp_success.append(ok)
        ems_snapshot_forecast_origin_mod24.append(fo)
        price_horizon_csv.append(pc)
        price_horizon_lp.append(plp)

        cosim_print(
            verbose,
            f"[Logger] step {step_i}/{n_steps}  granted={granted:.0f}s  "
            f"P_ref={p_ref[-1]:.2f}  P_act={p_act[-1]:.2f}  SOC={soc[-1]:.1f}",
        )

        t_next += DELTA_T_SEC

    fed.disconnect()

    hours = [t / 3600.0 for t in times_s]
    # Aligns with EMS: hour = int(granted // 3600); rolling_forecast_24(..., hour % 24)
    hour_index = [int(t // 3600) for t in times_s]
    forecast_origin_hour = [h % 24 for h in hour_index]
    # Backward-compatible aliases
    clock_hour_index = hour_index
    lp_forecast_origin_mod24 = forecast_origin_hour
    # P_ref from init LP until first clock hour completes; after that, hourly RHEMS
    p_ref_source_flag = [
        "initializing_rhems" if h == 0 else "hourly_lp" for h in hour_index
    ]

    meta = {
        "time_axis": "HELICS granted time [s] at the **end** of each 300 s interval",
        "interval_sec": DELTA_T_SEC,
        "alignment_note": (
            "Federates use HELICS INT_INDEX_GROUP (EMS=0, Droop=1, Network=2, Logger=3) so EMS publishes "
            "P_ref before Droop reads it; EMS re-publishes P_ref every 5 min. Remaining lag should be negligible."
        ),
        "sign_convention": (
            "EMS LP, Droop, and OpenDSS Generator use one convention: "
            "P_kW > 0 = injection to grid (discharge); P_kW < 0 = absorption (charge); "
            "Generator.kW > 0 = generation into the network."
        ),
        "price_units": "prices_hourly.csv; treated as $/kWh for EMS objective",
        "power_units": "kW",
        "soc_units": "kWh (integrated in Droop federate only; OpenDSS has no Storage/SOC for BESS)",
        "voltage": f"pu magnitude at OpenDSS node {BESS_BUS_NODE} (single phase)",
        "V_node_pu_semantics": (
            "**Post-action voltage (logged):** single-phase pu magnitude **after** power flow with "
            "this step's **P_act** on Generator.BESS. "
            "Logger rows align **P_act** and **V_node_pu** to the same interval end time; "
            "Droop subscription may still see a one-step older **V_node_pu** (see alignment_note)."
        ),
        "rolling_horizon": "EMS uses a fixed 24-hour window (non-shrinking); recomputed each clock hour",
        "forecast_type": (
            "Rolling 24h from prices_hourly.csv; optional EMS_PRICE_NOISE_STD adds Gaussian noise to newly "
            "introduced far-horizon absolute hours (a>=24): each hour adds one new ε and keeps prior ε fixed."
        ),
        "price_noise": (
            "EMS_PRICE_NOISE_STD ($/kWh) and optional EMS_PRICE_NOISE_SEED; LP objective uses "
            "price_horizon_lp. Each perturbed slot is always **CSV[a mod 24] + ε_a**, never based on an "
            "already-perturbed LP value, and ε state does not reset at day boundaries."
        ),
        "temporal_discretization": (
            "EMS LP uses **hourly** SOC constraints; Droop uses **5-minute** SOC integration. "
            "LP does not enforce SOC bounds at intra-hour Droop steps (see common.py)."
        ),
        "ems_soc_observation": (
            "EMS uses **downsampled / decision-time SOC only** (when it re-solves each clock hour). "
            "Not intra-hour continuous-time state; do not interpret as dense full-state OC."
        ),
        "droop_voltage_feedback": (
            "V_node_pu is used by Droop in a deadband active-power law around V_DROOP_REF_PU; "
            "outside the deadband, P_act is corrected by K_DROOP_KW_PER_PU * voltage error and then clipped "
            "to inverter limits."
        ),
        "bc_fields": (
            "Per row: hour_index, forecast_origin_hour (= hour mod 24), p_ref_source_flag "
            "(initializing_rhems during first clock hour after init LP; hourly_lp thereafter). "
            "clock_hour_index / lp_forecast_origin_mod24 are the same arrays (legacy names). "
            "For LP-to-(o,a) pairing when P_ref is held intra-hour, align on hour boundaries."
        ),
        "ems_lp_snapshot": (
            "Per row (same length as times_s): lp_solve_wall_s is HiGHS linprog wall time [s] when EMS solved; "
            "P_horizon_kW is [P_0..P_23] kW for that solve (null if parse failed). "
            "price_horizon_csv / price_horizon_lp are 24-vector $/kWh inputs (LP uses lp column). "
            "EMS only recomputes at hour boundaries; between updates the last snapshot is repeated on the wire."
        ),
    }
    write_log(
        out_path,
        {
            "meta": meta,
            "delta_t_sec": DELTA_T_SEC,
            "times_s": times_s,
            "hours": hours,
            "hour_index": hour_index,
            "forecast_origin_hour": forecast_origin_hour,
            "p_ref_source_flag": p_ref_source_flag,
            "clock_hour_index": clock_hour_index,
            "lp_forecast_origin_mod24": lp_forecast_origin_mod24,
            "P_ref_kW": p_ref,
            "P_act_kW": p_act,
            "SOC_kWh": soc,
            "V_node_pu": v_node,
            "lp_solve_wall_s": lp_solve_wall_s,
            "P_horizon_kW": P_horizon_kW,
            "lp_success": lp_success,
            "lp_forecast_origin_mod24_ems": ems_snapshot_forecast_origin_mod24,
            "price_horizon_csv": price_horizon_csv,
            "price_horizon_lp": price_horizon_lp,
        },
    )
