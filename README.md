
# ExpertV5 – Safe Reinforcement Learning for Non‑Stationary Driving

This repository contains experiments for **safe reinforcement learning in non‑stationary environments** using `highway-env` and Stable‑Baselines3.

Key features:
- Context‑dependent non‑stationarity (Markov context switching)
- Safety shield for action filtering
- Adjustment‑speed safety constraint
- Training pipelines for DQN / PPO / SAC
- Logging and plotting utilities

---

# Installation

Create virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

# Training Examples

## Baseline (no shield, no adjustment speed)

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 5000 --seed 0 --no_mpc --no_conformal --run_dir T1_baseline
```

## Adjustment‑speed constraint only

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 8000 --seed 0 --no_mpc --no_conformal --adjust_speed --run_dir T2_adj_only
```

## Shield only

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 12000 --seed 0 --run_dir T3_shield_only_dqn
```

## Shield + adjustment speed

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 20000 --seed 0 --adjust_speed --run_dir T4_shield_plus_adj_dqn
```

---

# IMPORTANT: `--adjust_speed` Flag

`--adjust_speed` is a **boolean flag**.

Correct usage:

Enable adjustment‑speed constraint:

```
--adjust_speed
```

Disable adjustment‑speed constraint:

Do **not include the flag**.

Incorrect command (will cause error):

```
--adjust_speed 0
```

Error produced:

```
train_discrete.py: error: unrecognized arguments: 0
```

---

# Correct Examples

Adjustment speed OFF:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 12000 --seed 0 --run_dir T3_shield_only_dqn
```

Adjustment speed ON:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 20000 --seed 0 --adjust_speed --run_dir T4_shield_plus_adj_dqn
```

---

# Plotting Results

Single run:

```powershell
python -m scripts.plot_run --run_dir F_dqn_seed0
```

Compare experiments:

```powershell
python -m scripts.plot_compare --groups baseline=B_dqn_seed* adj=F_dqn_seed* --metric violation
```

---

# Live Visualization

```powershell
python -m scripts.plot_live_sanity --run_dir T4_shield_plus_adj_dqn
```

Panels show:
- context switching
- violation rate
- adjustment risk
- unsafe trigger

---

# Repository Structure

```
src/
    safety.py
    wrappers.py
    adjust_speed/

scripts/
    train_discrete.py
    train_continuous.py
    plot_run.py
    plot_compare.py
    plot_live_sanity.py

runs/
    experiment outputs
```

---

# Recommended Experiment Order

1. Baseline
2. Adjustment‑speed constraint
3. Shield only
4. Shield + adjustment speed
5. Multi‑seed experiments
6. Plot aggregated results



---

# Expected Diagnostics (How to Know the System Works)

During training you will log several diagnostic variables into:

runs/<run_dir>/train_monitor.csv

The most important ones are:

Variable | Meaning | Expected Behaviour
--- | --- | ---
ctx_id | Context / regime id | Should switch occasionally (Markov context switching)
violation | Safety violation (collision) | Should decrease when safety mechanisms activate
shield_used | Shield intervention | Should activate occasionally in risky states
adj_risk | Adjustment-speed risk estimate | Between 0 and 1, spikes after context shifts
adj_unsafe | Unsafe adaptation trigger | Turns 1 when environment changes faster than agent adapts

---

Typical diagnostic behaviour:

1. Context Switching

ctx_id changes occasionally:

0 0 0 1 1 2 2 2 0 0

This confirms the nonstationary environment is active.

2. Adjustment-Speed Risk

adj_risk spikes after distribution shifts:

0.1 0.2 0.15 0.8 0.9 0.7 0.2

This means the algorithm detected fast environment change.

3. Unsafe Trigger

adj_unsafe becomes 1 when

S_env > S_agent + margin

Example:

0 0 0 1 1 0 0

When this happens the system tightens safety constraints.

4. Shield Usage

shield_used should occasionally activate:

0 0 0 1 0 0 1

This means the safety shield prevented unsafe actions.

---

Live visualization:

python -m scripts.plot_live_sanity --run_dir <run_dir>

Panels show:
1. Context switching
2. Violation rate
3. Adjustment risk
4. Unsafe trigger

If these signals behave as described above, your nonstationary safety system is working correctly.

---
