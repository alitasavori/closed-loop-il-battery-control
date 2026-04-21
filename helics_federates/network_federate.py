"""HELICS federate: OpenDSS power flow → single-node pu voltage (no SOC).

The BESS is modeled as a ``Generator`` with ``kW = P_act``. There is **no**
``Storage`` element and **no** internal energy or SOC state inside OpenDSS for
this battery — all SOC is integrated in the Droop federate only.

OpenDSS receives P_act (kW) and publishes the voltage magnitude (pu) at
``BESS_BUS_NODE`` **after** the power flow with that ``P_act`` (post-action V).
"""

from __future__ import annotations

import math
import os
import time

import helics as h
import opendssdirect as dss

from . import common
from .common import (
    BESS_BUS_NODE,
    DEFAULT_SIM_DURATION_SEC,
    DELTA_T_SEC,
    cosim_print,
    cosim_trace_enabled,
    ieee34_dir,
    parse_bus_node_selector,
    safe_double,
)

# Set True after IEEE34+BESS is compiled once (main thread in run_cosimulation, or first run()).
_circuit_built: bool = False


def reset_circuit_built_flag() -> None:
    """Call before each co-simulation so OpenDSS is recompiled for a clean run."""
    global _circuit_built
    _circuit_built = False


def ensure_ieee34_bess_circuit_built(verbose: bool = False) -> None:
    """Compile IEEE34 + BESS and run one power flow.

    Call **once on the main thread** before starting federate threads so OpenDSS
    does not hold the GIL during compile/Solve while other threads must reach
    ``enter_initializing_mode()`` — otherwise HELICS init can deadlock.
    Safe to call again from ``network_federate.run`` alone (tests): second call no-ops.
    """
    global _circuit_built
    if _circuit_built:
        os.chdir(ieee34_dir().parent)
        return

    master = ieee34_dir() / "IEEE34_PV.dss"
    os.chdir(master.parent)

    t0 = time.perf_counter()
    dss.Text.Command("Clear")
    dss.Text.Command(f'Compile "{master.resolve()}"')
    dss.Text.Command(
        "New Generator.BESS Phases=3 Bus1=830 kV=24.9 kW=0 kvar=0 Model=1 Vminpu=0.7 Vmaxpu=1.3"
    )
    dss.Text.Command("Set Mode=Daily")
    dss.Text.Command("Set Number=1")
    dss.Text.Command("Set StepSize=5m")
    dss.Text.Command("Set ControlMode=Static")
    dss.Solution.MaxIterations(50)
    dss.Solution.MaxControlIterations(30)
    dss.Solution.Solve()
    _circuit_built = True
    cosim_print(
        verbose,
        f"[Network] OpenDSS IEEE34+BESS compiled + initial Solve  wall_s={time.perf_counter() - t0:.3f}",
    )


def _phase_pu_at_bus(bus_selector: str, phase_idx: int) -> float:
    """Magnitude in pu for one phase from rectangular PuVoltage."""
    dss.Circuit.SetActiveBus(bus_selector)
    pv = dss.Bus.PuVoltage()
    mags = [math.hypot(pv[i], pv[i + 1]) for i in range(0, len(pv), 2)]
    if not mags:
        return 1.0
    if phase_idx < 0 or phase_idx >= len(mags):
        return mags[0]
    return mags[phase_idx]


def run(
    broker_address: str,
    sim_duration_sec: float = DEFAULT_SIM_DURATION_SEC,
    verbose: bool = False,
) -> None:
    ensure_ieee34_bess_circuit_built(verbose)
    bus_sel, phase_idx = parse_bus_node_selector(BESS_BUS_NODE)

    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"network_core_{os.getpid()}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(
        fedinfo, f"--federates=1 --broker_address={broker_address}"
    )
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, DELTA_T_SEC)
    common.configure_federate_info_thread_safe(fedinfo)
    common.configure_federate_index_group(fedinfo, 2)

    fed = h.helicsCreateValueFederate("Network", fedinfo)
    # Single-phase pu magnitude at the BESS connection point (see common.BESS_BUS_NODE)
    pub_v = fed.register_global_publication("V_node_pu", h.HELICS_DATA_TYPE_DOUBLE, "pu")
    # P_act > 0 = injection (discharge); matches Generator kW > 0 in OpenDSS
    sub_p = fed.register_subscription("P_act", "kW")

    cosim_print(verbose, "[Network] HELICS federate created; reading V from loaded circuit")
    v0 = _phase_pu_at_bus(bus_sel, phase_idx)

    cosim_print(verbose, "[Network] calling enter_initializing_mode() ...")
    fed.enter_initializing_mode()
    pub_v.publish(v0)
    fed.enter_executing_mode()
    cosim_print(verbose, "[Network] enter_executing_mode() OK - starting time steps")

    n_steps = int(round(sim_duration_sec / DELTA_T_SEC))
    step_i = 0
    t_next = DELTA_T_SEC
    while t_next <= sim_duration_sec + 1e-6:
        cosim_print(
            cosim_trace_enabled(verbose),
            f"[Network] trace before request_time  step={step_i + 1}  t_next={t_next:.0f}s",
        )
        granted = fed.request_time(t_next)
        cosim_print(
            cosim_trace_enabled(verbose),
            f"[Network] trace after request_time  step={step_i + 1}  granted={granted:.0f}s",
        )
        step_i += 1

        p_act = safe_double(sub_p.value, 0.0)
        dss.Text.Command(f"Edit Generator.BESS kW={p_act}")
        cosim_print(
            cosim_trace_enabled(verbose),
            f"[Network] trace before Solve  step={step_i}  P_act={p_act:.4f} kW",
        )
        t_solve0 = time.perf_counter()
        solve_err = ""
        try:
            dss.Solution.Solve()
        except Exception as e:  # OpenDSS can raise on warning (#485 max control iterations)
            solve_err = str(e)
        solve_s = time.perf_counter() - t_solve0
        cosim_print(
            cosim_trace_enabled(verbose),
            f"[Network] trace after Solve  step={step_i}  converged={dss.Solution.Converged()}  "
            f"wall_s={solve_s:.4f}{'  err=' + solve_err if solve_err else ''}",
        )
        conv = dss.Solution.Converged() and not solve_err
        extra = 0.0
        if not conv:
            t_fb0 = time.perf_counter()
            dss.Text.Command("Set ControlMode=Off")
            try:
                dss.Solution.Solve()
            except Exception as e:
                solve_err = f"{solve_err}; fallback={e}" if solve_err else f"fallback={e}"
            dss.Text.Command("Set ControlMode=Static")
            extra = time.perf_counter() - t_fb0
            conv = dss.Solution.Converged()

        v_pu = _phase_pu_at_bus(bus_sel, phase_idx)
        pub_v.publish(v_pu)

        cosim_print(
            verbose,
            f"[Network] step {step_i}/{n_steps}  granted={granted:.0f}s  P_act={p_act:.2f} kW  "
            f"Solve_wall_s={solve_s:.3f}  extra_fallback_s={extra:.3f}  converged={conv}  "
            f"V={v_pu:.4f} pu{'  solve_err=' + solve_err if solve_err else ''}",
        )

        t_next += DELTA_T_SEC

    fed.disconnect()
