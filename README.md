# Closed-Loop Imitation Learning for Battery Control in Distribution Systems

**Course:** Advanced AI, Spring 2025 — University of Utah  
**Author:** Ali Tasavvori · Department of Electrical and Computer Engineering  
**Report:** [`final_project_report.pdf`](./final_project_report.pdf)

---

## Overview

This project applies imitation learning — specifically **Behavioral Cloning (BC)** and **Dataset Aggregation (DAgger)** — to train a learned dispatch policy for a Battery Energy Storage System (BESS) in an unbalanced distribution network.

The standard control approach is a rolling-horizon **Linear Program (LP) Energy Management System (EMS)** re-solved every hour. While near-optimal, this LP becomes a bottleneck when embedded in planning loops requiring hundreds of co-simulation evaluations. A single forward pass through a trained **Random Forest** policy replaces the LP entirely at inference time, while a local droop layer continues to handle sub-second voltage regulation identically in all setups.

Experiments run on the **IEEE 34-bus unbalanced three-phase feeder** in a high-fidelity **OpenDSS + HELICS** co-simulation over 40-day runs across three price-noise levels.

---

## Key Results

| Controller | Noise σ | Revenue [$] | Voltage RMSE [pu] | Setpoint MSE [kW²] |
|------------|---------|-------------|-------------------|-------------------|
| LP-EMS     | 0.004   | 6,272       | 0.02304           | —                 |
| BC (RF)    | 0.004   | 6,272       | 0.02304           | —                 |
| LP-EMS     | 0.04    | 13,631      | 0.02496           | —                 |
| BC (RF)    | 0.04    | 12,291      | 0.02320           | 30,623            |
| LP-EMS     | 0.4     | 120,712     | 0.02523           | —                 |
| BC (RF)    | 0.4     | 108,440     | 0.02389           | 39,168            |
| DAgger     | 0.4     | 109,820     | 0.02391           | 32,514            |

- BC matches LP exactly at low noise; captures ~90% of revenue at high noise  
- DAgger reduces setpoint MSE by **17%** over BC in the hardest scenario  
- Voltage RMSE is lower for BC/DAgger than LP — the droop layer provides a robust safety floor

---

## Repository Structure

```
bess-imitation-learning/
├── advanced_AI.ipynb        # Main notebook: EMS runs, BC/DAgger training, evaluation
├── helics_federates/        # HELICS co-simulation federates
│   ├── __init__.py
│   ├── run_cosimulation.py  # Entry point for co-sim
│   ├── common.py            # Shared constants (SOC bounds, timestep, price loader)
│   ├── ems_federate.py      # Rolling-horizon LP energy management system
│   ├── droop_federate.py    # Volt-VAR / Volt-Watt droop controller
│   ├── grid_federate.py     # OpenDSS network solver
│   └── logger_federate.py   # Data recording
├── bc_models/
│   └── bc_rf.joblib         # Trained Random Forest BC policy
├── IEEE34busfile/
│   ├── IEEE34_PV.dss        # Master OpenDSS feeder file with PV
│   └── ...                  # Load shapes, line data, bus coordinates
├── final_project_report.pdf # Full project report (NeurIPS template)
└── README.md
```

---

## Setup & Installation

### Requirements

- Python 3.9+
- [OpenDSS Direct](https://github.com/dss-extensions/OpenDSSDirect.py): `pip install opendssdirect.py`
- [HELICS](https://helics.org): `pip install helics`
- Standard scientific stack: `pip install numpy pandas scipy matplotlib scikit-learn joblib`

Install everything at once:

```bash
pip install opendssdirect.py helics numpy pandas scipy matplotlib scikit-learn joblib
```

### Running the Notebook

1. Clone the repository and open the project folder:
   ```bash
   git clone https://github.com/ali-tasavvori/bess-imitation-learning.git
   cd bess-imitation-learning
   ```

2. Launch Jupyter and open `advanced_AI.ipynb`:
   ```bash
   jupyter notebook advanced_AI.ipynb
   ```

3. The notebook is organized into sequential sections:
   - **Cell group 1–2:** Baseline OpenDSS feeder validation (no HELICS)
   - **Cell group 3–4:** LP-EMS closed-loop HELICS run (low noise, σ=0.004)
   - **Cell group 5–6:** LP-EMS run (medium noise, σ=0.04)
   - **Cell group 7–8:** BC Random Forest closed-loop run
   - **Cell group 9+:** DAgger run and final comparison

   Run cells top-to-bottom. Each run group ends with a KPI summary cell that prints revenue, voltage RMSE, and SOC saturation statistics.

---

## Co-simulation Architecture

The HELICS broker coordinates five federates:

```
HELICS Broker
├── Grid Federate        (OpenDSS, 5-min resolution) ← solves power flow
├── EMS / Policy Federate (LP or RF, 60-min resolution) ← hourly dispatch
├── Droop Control Federate (5-min resolution) ← fast voltage regulation
├── Random Price Generator ← σ-noise price signal
└── Logger Federate      ← records SOC, voltage, power, prices
```

To switch between LP-EMS and BC policy, set the flag in the notebook run cell:

```python
USE_BC_POLICY = False   # LP-EMS
USE_BC_POLICY = True    # Learned RF policy
```

To change price noise level:

```python
os.environ["EMS_PRICE_NOISE_STD"] = "0.004"   # low
os.environ["EMS_PRICE_NOISE_STD"] = "0.04"    # medium
os.environ["EMS_PRICE_NOISE_STD"] = "0.4"     # high (stress scenario)
```

---

## Method Summary

**Expert oracle:** A shrinking-horizon LP solved at each hourly boundary, maximizing energy arbitrage revenue subject to SOC dynamics and linearized voltage constraints.

**Observation space:** `[V_bus (pu), SOC (kWh), sin(2πh/24), cos(2πh/24)]` — no future load or price information required at inference time.

**BC training:** Supervised regression on expert (EMS) demonstrations collected across four seasonal load profiles and three PV penetration levels.

**DAgger:** Iterative dataset aggregation — roll out current policy, query EMS for corrective labels at learner-visited states, retrain.

**Models evaluated:** MLP, LSTM, LSTM + auxiliary task, Random Forest. RF performed best in BC setting.

---

## Citation

If you use this code or build on this work, please cite:

```
Ali Tasavvori. "Closed-Loop Imitation Learning for Battery Control in Distribution Systems."
Advanced AI Course Project, University of Utah, Spring 2025.
```

---

## License

This repository is released for academic and educational use.
