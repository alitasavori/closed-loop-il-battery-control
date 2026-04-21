"""HELICS federate: actuator — owns SOC; receives P_ref and V; outputs P_act.

**SOC is device state and is integrated only here** (not in OpenDSS).

Voltage feedback
----------------
- ``V_node_pu`` is used in a deadband droop law around ``V_DROOP_REF_PU``.
- Above the upper band, reduce injection / increase charging.
- Below the lower band, increase injection / reduce charging.
- Final command still respects inverter bounds: ``P_act = clip(P_ref + droop_term)``.

Flow each 5-minute step:
  - Read P_ref (EMS), V_node_pu (network; may lag one exchange — see logger meta).
  - P_act = clip(P_ref + voltage droop correction).
  - Update SOC from P_act.

Sign: P_act > 0 = discharge / injection to grid; P_act < 0 = charge (see common module docstring).
"""

from __future__ import annotations

import os

import helics as h

from . import common
from .common import (
    DEFAULT_SIM_DURATION_SEC,
    DELTA_T_SEC,
    ETA_CH,
    P_MAX_KW,
    SOC_INIT_KWH,
    SOC_MAX_KWH,
    SOC_MIN_KWH,
    cosim_print,
    safe_double,
)

# Enable voltage-driven active-power droop correction.
VOLTAGE_FEEDBACK_IN_CONTROL = True

# Droop tuning (kW/pu) with symmetric deadband around reference voltage.
V_DROOP_REF_PU = 1.00
V_DROOP_DEADBAND_PU = 0.01
K_DROOP_KW_PER_PU = 2000.0


def _clip_power(p_kw: float) -> float:
    return max(-P_MAX_KW, min(P_MAX_KW, p_kw))


def _droop_correction_kw(v_pu: float) -> float:
    """Return additive kW correction to P_ref from measured voltage.

    Sign convention:
    - Positive P injects/discharges (tends to raise local voltage).
    - Therefore at high V, correction is negative (reduce injection / move toward charging).
    - At low V, correction is positive (increase injection / move toward discharging).
    """
    v_hi = V_DROOP_REF_PU + V_DROOP_DEADBAND_PU
    v_lo = V_DROOP_REF_PU - V_DROOP_DEADBAND_PU
    if v_pu > v_hi:
        return -K_DROOP_KW_PER_PU * (v_pu - v_hi)
    if v_pu < v_lo:
        return K_DROOP_KW_PER_PU * (v_lo - v_pu)
    return 0.0


def run(
    broker_address: str,
    sim_duration_sec: float = DEFAULT_SIM_DURATION_SEC,
    verbose: bool = False,
) -> None:
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"droop_core_{os.getpid()}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(
        fedinfo, f"--federates=1 --broker_address={broker_address}"
    )
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, DELTA_T_SEC)
    common.configure_federate_info_thread_safe(fedinfo)
    common.configure_federate_index_group(fedinfo, 1)

    fed = h.helicsCreateValueFederate("Droop", fedinfo)
    pub_pact = fed.register_global_publication("P_act", h.HELICS_DATA_TYPE_DOUBLE, "kW")
    pub_soc = fed.register_global_publication("SOC", h.HELICS_DATA_TYPE_DOUBLE, "kWh")
    sub_pref = fed.register_subscription("P_ref", "kW")
    sub_v = fed.register_subscription("V_node_pu", "pu")
    cosim_print(verbose, "[Droop] HELICS federate created; calling enter_initializing_mode() ...")

    fed.enter_initializing_mode()
    pub_pact.publish(0.0)
    pub_soc.publish(SOC_INIT_KWH)
    fed.enter_executing_mode()
    cosim_print(verbose, "[Droop] enter_executing_mode() OK - starting time steps")

    soc_kwh = SOC_INIT_KWH
    dt_h = DELTA_T_SEC / 3600.0
    n_steps = int(round(sim_duration_sec / DELTA_T_SEC))
    step_i = 0

    t_next = DELTA_T_SEC
    while t_next <= sim_duration_sec + 1e-6:
        granted = fed.request_time(t_next)
        step_i += 1

        pref = safe_double(sub_pref.value, 0.0)
        v_node_pu = safe_double(sub_v.value, 1.0)

        if VOLTAGE_FEEDBACK_IN_CONTROL:
            droop_kw = _droop_correction_kw(v_node_pu)
            p_cmd = pref + droop_kw
            p_act = _clip_power(p_cmd)
        else:
            droop_kw = 0.0
            p_act = _clip_power(pref)
        soc_kwh = soc_kwh - ETA_CH * p_act * dt_h
        soc_kwh = max(SOC_MIN_KWH, min(SOC_MAX_KWH, soc_kwh))

        pub_pact.publish(p_act)
        pub_soc.publish(soc_kwh)

        cosim_print(
            verbose,
            f"[Droop] step {step_i}/{n_steps}  granted={granted:.0f}s  "
            f"V={v_node_pu:.4f}  P_ref={pref:.2f}  droop={droop_kw:+.2f}  P_act={p_act:.2f}  SOC={soc_kwh:.1f}",
        )

        t_next += DELTA_T_SEC

    fed.disconnect()
