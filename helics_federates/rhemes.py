"""Rolling-horizon expert: maximize sum(pi * P) with linear SOC constraints.

The horizon is **non-shrinking**: each solve always has **24** hourly decision
variables (P_0..P_23). Only P_0 is applied; at the next EMS clock hour the
problem is re-solved with updated SOC (fixed 24-step window, MPC-style).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

from .common import ETA_CH, P_MAX_KW, SOC_MAX_KWH, SOC_MIN_KWH


def rhemes_lp_solve(
    soc_kwh: float,
    pi_forecast_24: list[float],
) -> tuple[np.ndarray, float, bool]:
    """Solve 24-hour LP; return (P_0..P_23 [kW], linprog wall time [s], success).

    **Sign:** P_i > 0 = discharge / export to grid (same as Droop and Generator.kW).

    SOC_{k+1} = SOC_k - eta * P_k * 1h (hourly EMS step; P_k average kW).
    Prices pi_i in $/kWh (or consistent $/kWh-equivalent); P_i in kW.

    ``lp_solve_wall_s`` is **only** the HiGHS ``linprog`` call (excludes Python setup).
    """
    eta = ETA_CH
    n = 24
    pi = np.asarray(pi_forecast_24, dtype=float)
    if pi.shape != (n,):
        raise ValueError("pi_forecast_24 must have length 24")

    c = -pi  # minimize -sum(pi*P)
    bounds = [(-P_MAX_KW, P_MAX_KW)] * n

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for k in range(1, n + 1):
        row = np.zeros(n)
        row[:k] = eta
        a_ub.append(row.tolist())
        b_ub.append(soc_kwh - SOC_MIN_KWH)
        row2 = np.zeros(n)
        row2[:k] = -eta
        a_ub.append(row2.tolist())
        b_ub.append(-(soc_kwh - SOC_MAX_KWH))

    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    wall_s = time.perf_counter() - t0

    if not res.success or res.x is None:
        return np.zeros(n, dtype=float), wall_s, False
    return np.asarray(res.x, dtype=float), wall_s, True


def rhemes_p_ref(
    soc_kwh: float,
    pi_forecast_24: list[float],
) -> float:
    """Solve 24-hour LP; return first-hour power P_0^* (kW) only (backward compatible)."""
    p, _, _ = rhemes_lp_solve(soc_kwh, pi_forecast_24)
    return float(p[0])
