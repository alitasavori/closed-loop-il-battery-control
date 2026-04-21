"""HELICS federate: BC policy replacement for EMS.

This federate mirrors EMS I/O:
- subscribes: ``SOC``
- publishes: ``P_ref`` and ``EMS_LP_snapshot``

Policy input matches the EMS-style BC dataset:
``[soc_kwh, price_0..price_23]`` where ``price_*`` comes from the same
rolling horizon logic as EMS.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import helics as h
import joblib
import numpy as np
import torch
import torch.nn as nn

from . import common
from .common import (
    DEFAULT_SIM_DURATION_SEC,
    DELTA_T_SEC,
    P_MAX_KW,
    SECONDS_PER_HOUR,
    SOC_INIT_KWH,
    cosim_print,
    safe_double,
)


class LSTMSocBC(nn.Module):
    """Price sequence + SOC -> P_ref (``n_outputs=1``) or full horizon (``n_outputs=24``)."""

    def __init__(
        self,
        hidden_size=64,
        layers=2,
        soc_hidden=16,
        head_hidden=64,
        dropout=0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.soc_branch = nn.Sequential(
            nn.Linear(1, soc_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + soc_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.n_outputs),
        )

    def forward(self, x_price_seq, x_soc):
        out, _ = self.lstm(x_price_seq)
        h_last = out[:, -1, :]
        h_soc = self.soc_branch(x_soc)
        h = torch.cat([h_last, h_soc], dim=1)
        return self.head(h)


class LSTMSocMLPBC(nn.Module):
    """LSTM over price sequence + SOC branch, then MLP head -> single P_ref (main task only)."""

    def __init__(
        self,
        hidden_size: int = 64,
        layers: int = 2,
        soc_hidden: int = 16,
        mlp_hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        mlp_hidden = tuple(int(x) for x in mlp_hidden)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.soc_branch = nn.Sequential(
            nn.Linear(1, soc_hidden),
            nn.ReLU(),
        )
        fused = hidden_size + soc_hidden
        parts: list[nn.Module] = []
        prev = fused
        for i, h in enumerate(mlp_hidden):
            parts.append(nn.Linear(prev, h))
            parts.append(nn.ReLU())
            if dropout > 0 and i < len(mlp_hidden) - 1:
                parts.append(nn.Dropout(p=dropout))
            prev = h
        parts.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*parts)

    def forward(self, x_price_seq, x_soc):
        out, _ = self.lstm(x_price_seq)
        h_last = out[:, -1, :]
        h_soc = self.soc_branch(x_soc)
        h = torch.cat([h_last, h_soc], dim=1)
        return self.mlp(h)


def _resolve_model_paths(model_pt: str | None, artifacts_joblib: str | None) -> tuple[Path, Path]:
    pt = model_pt or os.environ.get("BC_MODEL_PT", "")
    art = artifacts_joblib or os.environ.get("BC_MODEL_ARTIFACTS", "")
    if not pt or not art:
        raise ValueError(
            "BC model paths missing. Provide --bc-model-pt/--bc-model-artifacts "
            "or set BC_MODEL_PT/BC_MODEL_ARTIFACTS env vars."
        )
    return Path(pt), Path(art)


def _resolve_tabular_path(model_joblib: str | None) -> Path:
    p = model_joblib or os.environ.get("BC_MODEL_JOBLIB", "")
    if not p:
        raise ValueError(
            "Tabular BC model path missing. Provide --bc-model-joblib "
            "or set BC_MODEL_JOBLIB env var."
        )
    return Path(p)


def _load_bc_policy(model_pt: Path, artifacts_joblib: Path):
    art = joblib.load(artifacts_joblib)
    kwargs = dict(art.get("model_kwargs", {}))
    mtype = art.get("model_type", "lstm_soc")
    if mtype == "lstm_soc_mlp":
        model = LSTMSocMLPBC(**kwargs)
    else:
        model = LSTMSocBC(**kwargs)
    state = torch.load(model_pt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, art


def _build_row_dict(soc_kwh: float, price_horizon_lp: list[float]) -> dict[str, float]:
    p = np.asarray(price_horizon_lp, dtype=float)
    row: dict[str, float] = {"soc_kwh": float(soc_kwh)}
    for i in range(24):
        row[f"price_{i}"] = float(p[i])

    # Engineered features used in advanced tabular exploration.
    row["p_mean"] = float(np.mean(p))
    row["p_std"] = float(np.std(p))
    row["p_min"] = float(np.min(p))
    row["p_max"] = float(np.max(p))
    row["p_span"] = row["p_max"] - row["p_min"]
    row["p0"] = float(p[0])
    row["p1"] = float(p[1])
    row["p2"] = float(p[2])
    row["p23"] = float(p[23])
    row["p0_minus_mean"] = row["p0"] - row["p_mean"]
    row["p0_minus_p23"] = row["p0"] - row["p23"]
    row["p12_mean"] = float(np.mean(p[:12]))
    row["p12_tail_mean"] = float(np.mean(p[12:]))
    row["p12_gap"] = row["p12_mean"] - row["p12_tail_mean"]
    dp = np.diff(p)
    row["dp_mean"] = float(np.mean(dp))
    row["dp_std"] = float(np.std(dp))
    row["dp_max"] = float(np.max(dp))
    row["dp_min"] = float(np.min(dp))
    row["soc_x_p0"] = row["soc_kwh"] * row["p0"]
    row["soc_x_p_mean"] = row["soc_kwh"] * row["p_mean"]
    row["soc_x_span"] = row["soc_kwh"] * row["p_span"]
    return row


def _load_tabular_policy(model_joblib: Path):
    bundle = joblib.load(model_joblib)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        features = list(bundle.get("features", ["soc_kwh", *[f"price_{k}" for k in range(24)]]))
    else:
        model = bundle
        features = ["soc_kwh", *[f"price_{k}" for k in range(24)]]
    return model, features


def _predict_pref_kw_tabular(model, features: list[str], soc_kwh: float, price_horizon_lp: list[float]) -> float:
    row = _build_row_dict(soc_kwh, price_horizon_lp)
    try:
        x = np.asarray([[row[f] for f in features]], dtype=float)
    except KeyError as e:
        raise KeyError(f"Tabular model requested unsupported feature: {e}") from e
    y = float(model.predict(x)[0])
    return float(max(-P_MAX_KW, min(P_MAX_KW, y)))


def _predict_pref_kw(model, art, soc_kwh: float, price_horizon_lp: list[float]) -> float:
    sc_price = art["sc_price"]
    sc_soc = art["sc_soc"]
    sc_y = art["sc_y"]

    x_price = np.asarray(price_horizon_lp, dtype=float).reshape(1, 24)
    x_soc = np.asarray([[soc_kwh]], dtype=float)

    x_price_s = sc_price.transform(x_price).reshape(1, 24, 1)
    x_soc_s = sc_soc.transform(x_soc)

    with torch.no_grad():
        y_s = model(
            torch.tensor(x_price_s, dtype=torch.float32),
            torch.tensor(x_soc_s, dtype=torch.float32),
        ).cpu().numpy()
    # Multi-output (horizon): use column 0 as current-hour P_ref for EMS wire.
    y_s2 = np.atleast_2d(y_s)
    y_inv = sc_y.inverse_transform(y_s2)[0, 0]
    return float(max(-P_MAX_KW, min(P_MAX_KW, y_inv)))


def run(
    broker_address: str,
    sim_duration_sec: float = DEFAULT_SIM_DURATION_SEC,
    verbose: bool = False,
    model_pt: str | None = None,
    artifacts_joblib: str | None = None,
    policy_kind: str = "lstm_pt",
    model_joblib: str | None = None,
) -> None:
    hourly_prices = common.load_hourly_price_profile()
    noise_std = common.ems_price_noise_std()
    noise_seed = common.ems_price_noise_seed()
    rng = random.Random(noise_seed) if noise_seed is not None else random.Random()
    epsilon_state: dict[int, float] = {}

    model = None
    art = None
    tab_features: list[str] | None = None
    model_name = ""
    if policy_kind == "lstm_pt":
        pt_path, art_path = _resolve_model_paths(model_pt, artifacts_joblib)
        model, art = _load_bc_policy(pt_path, art_path)
        model_name = pt_path.name
    elif policy_kind == "tabular_joblib":
        joblib_path = _resolve_tabular_path(model_joblib)
        model, tab_features = _load_tabular_policy(joblib_path)
        model_name = joblib_path.name
    else:
        raise ValueError("policy_kind must be one of {'lstm_pt', 'tabular_joblib'}.")

    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, f"ems_bc_core_{os.getpid()}")
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
    cosim_print(verbose, f"[EMS-BC] HELICS federate created using model={model_name} kind={policy_kind}")

    pi0_csv, pi0_lp = common.lp_price_horizon_with_suffix_noise(
        hourly_prices, 0, rng, noise_std, epsilon_state
    )
    t0 = time.perf_counter()
    if policy_kind == "lstm_pt":
        pref0 = _predict_pref_kw(model, art, SOC_INIT_KWH, pi0_lp)
    else:
        assert tab_features is not None
        pref0 = _predict_pref_kw_tabular(model, tab_features, SOC_INIT_KWH, pi0_lp)
    infer_s = time.perf_counter() - t0
    cosim_print(verbose, f"[EMS-BC] init infer_s={infer_s:.4f}  P_ref={pref0:.3f} kW")

    fed.enter_initializing_mode()
    pub_pref.publish(pref0)
    pub_lp.publish(
        json.dumps(
            {
                "lp_solve_wall_s": infer_s,
                "P_horizon_kW": [pref0] * 24,
                "lp_success": True,
                "forecast_origin_mod24": 0,
                "price_horizon_csv": pi0_csv,
                "price_horizon_lp": pi0_lp,
                "policy_type": f"bc_{policy_kind}",
            }
        )
    )
    fed.enter_executing_mode()

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
            t_inf = time.perf_counter()
            if policy_kind == "lstm_pt":
                current_pref = _predict_pref_kw(model, art, soc_kwh, pi_lp)
            else:
                assert tab_features is not None
                current_pref = _predict_pref_kw_tabular(model, tab_features, soc_kwh, pi_lp)
            infer_s = time.perf_counter() - t_inf
            pub_lp.publish(
                json.dumps(
                    {
                        "lp_solve_wall_s": infer_s,
                        "P_horizon_kW": [current_pref] * 24,
                        "lp_success": True,
                        "forecast_origin_mod24": hour % 24,
                        "price_horizon_csv": pi_csv,
                        "price_horizon_lp": pi_lp,
                        "policy_type": f"bc_{policy_kind}",
                    }
                )
            )
            cosim_print(
                verbose,
                f"[EMS-BC] hour={hour} granted={granted:.0f}s infer_s={infer_s:.4f} "
                f"SOC={soc_kwh:.1f} P_ref={current_pref:.3f}",
            )

        pub_pref.publish(current_pref)
        t_next += DELTA_T_SEC

    fed.disconnect()

