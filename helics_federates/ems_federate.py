"""HELICS federate: hourly EMS (RHEMS expert) -> P_ref.

EMS updates on **clock hour** changes only. Between updates, **P_ref is held
constant** for all **12** five-minute substeps (no races: same publication is
re-read by droop). At each new hour, SOC is taken from the Droop federate's
last published value.
"""

from __future__ import annotations

import json
import os
import random
import time

import helics as h

from .common import (
    DEFAULT_SIM_DURATION_SEC,
    DELTA_T_SEC,
    SECONDS_PER_HOUR,
    SOC_INIT_KWH,
    safe_double,
    cosim_print,
)
from .rhemes import rhemes_lp_solve
from . import common


def run(
    broker_address: str,
    sim_duration_sec: float = DEFAULT_SIM_DURATION_SEC,
    verbose: bool = False,
) -> None:
    hourly_prices = common.load_hourly_price_profile()
    noise_std = common.ems_price_noise_std()
    noise_seed = common.ems_price_noise_seed()
    rng = random.Random(noise_seed) if noise_seed is not None else random.Random()

    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"ems_core_{os.getpid()}")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetCoreInitString(
        fedinfo, f"--federates=1 --broker_address={broker_address}"
    )
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_PERIOD, DELTA_T_SEC)
    common.configure_federate_info_thread_safe(fedinfo)
    common.configure_federate_index_group(fedinfo, 0)

    fed = h.helicsCreateValueFederate("EMS", fedinfo)
    pub_pref = fed.register_global_publication("P_ref", h.HELICS_DATA_TYPE_DOUBLE, "kW")
    pub_lp = fed.register_global_publication("EMS_LP_snapshot", h.HELICS_DATA_TYPE_STRING, "")
    sub_soc = fed.register_subscription("SOC", "kWh")
    cosim_print(verbose, "[EMS] HELICS federate created")
    if noise_std > 0:
        cosim_print(
            verbose,
            f"[EMS] EMS_PRICE_NOISE_STD={noise_std} (suffix / end-of-horizon); seed={noise_seed!r}",
        )

    epsilon_state: dict[int, float] = {}
    pi0_csv, pi0_lp = common.lp_price_horizon_with_suffix_noise(
        hourly_prices, 0, rng, noise_std, epsilon_state
    )
    t0 = time.perf_counter()
    p_vec, lp_wall_s, lp_ok = rhemes_lp_solve(SOC_INIT_KWH, pi0_lp)
    pref0 = float(p_vec[0])
    cosim_print(
        verbose,
        f"[EMS] init LP  total_wall_s={time.perf_counter()-t0:.4f}  lp_solve_wall_s={lp_wall_s:.4f}  "
        f"P_ref={pref0:.3f} kW",
    )
    cosim_print(verbose, "[EMS] calling enter_initializing_mode() ...")
    fed.enter_initializing_mode()
    pub_pref.publish(pref0)
    pub_lp.publish(
        json.dumps(
            {
                "lp_solve_wall_s": lp_wall_s,
                "P_horizon_kW": p_vec.tolist(),
                "lp_success": lp_ok,
                "forecast_origin_mod24": 0,
                "price_horizon_csv": pi0_csv,
                "price_horizon_lp": pi0_lp,
            }
        )
    )
    fed.enter_executing_mode()
    cosim_print(verbose, "[EMS] enter_executing_mode() OK - starting time steps")

    t_next = DELTA_T_SEC
    last_hour_computed = 0
    current_pref = pref0

    while t_next <= sim_duration_sec + 1e-6:
        granted = fed.request_time(t_next)

        hour = int(granted // SECONDS_PER_HOUR)
        if hour != last_hour_computed:
            last_hour_computed = hour
            soc_kwh = safe_double(sub_soc.value, SOC_INIT_KWH)

            pi_csv, pi_lp = common.lp_price_horizon_with_suffix_noise(
                hourly_prices, hour, rng, noise_std, epsilon_state
            )
            t_lp = time.perf_counter()
            p_vec, lp_wall_s, lp_ok = rhemes_lp_solve(soc_kwh, pi_lp)
            current_pref = float(p_vec[0])
            cosim_print(
                verbose,
                f"[EMS] hour={hour} granted={granted:.0f}s  total_wall_s={time.perf_counter()-t_lp:.4f}  "
                f"lp_solve_wall_s={lp_wall_s:.4f}  SOC={soc_kwh:.1f}  forecast@mod24={hour%24}  P_ref={current_pref:.3f} kW",
            )
            pub_lp.publish(
                json.dumps(
                    {
                        "lp_solve_wall_s": lp_wall_s,
                        "P_horizon_kW": p_vec.tolist(),
                        "lp_success": lp_ok,
                        "forecast_origin_mod24": hour % 24,
                        "price_horizon_csv": pi_csv,
                        "price_horizon_lp": pi_lp,
                    }
                )
            )

        # Re-publish every step so Droop/Logger see P_ref for *this* grant (same index group ordering).
        pub_pref.publish(current_pref)

        t_next += DELTA_T_SEC

    fed.disconnect()
