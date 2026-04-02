# Kigali Job Market RL — Reinforcement Learning Summative

**Author:** Jean Jacques JABO  
**Program:** BSE Pre-Capstone  
**Supervisor:** GATERA Thadde  
**Institution:** African Leadership University (ALU)  
**Year:** 2026  

---

## Project Overview

This project implements a reinforcement learning solution for the **Job Market Forecasting** capstone mission. The agent represents a job seeker in Kigali, Rwanda, learning to acquire in-demand skills and secure employment in one of five key sectors (Technology, Business, Agriculture, Construction, Healthcare), aligned with Rwanda's Vision 2050 and NST1 priorities.

The environment simulates the real structural challenge described in the capstone: **a 20.6% youth unemployment rate driven by skill-market misalignment** (NISR, 2023).

---

## Repository Structure

```
jabo_jean_jacques_rl_summative/
├── environment/
│   ├── custom_env.py        # KigaliJobMarketEnv (Gymnasium)
│   └── rendering.py         # Pygame 2D visualization
├── training/
│   ├── dqn_training.py      # DQN — 10 hyperparameter runs
│   ├── pg_training.py       # PPO — 10 hyperparameter runs
│   └── reinforce_algorithm.py  # REINFORCE (manual PyTorch) — 10 runs
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved PPO and REINFORCE models
├── results/                 # CSV tables and JSON model info
├── plots/                   # All generated plots
├── static_demo.py           # Random agent visualization (no model)
├── evaluate.py              # Cross-algorithm comparison
├── main.py                  # Entry point — run best agent
├── requirements.txt
└── README.md
```

---

## Environment: `KigaliJobMarketEnv`

| Component | Description |
|-----------|-------------|
| **Observation Space** | `Box(0, 1, shape=(13,))` — 5 skill levels + 5 market demands + time + energy + employed |
| **Action Space** | `Discrete(12)` — Study (×5), Apply (×5), Network, Rest |
| **Reward** | +5–10 (hired), +0.05–0.2 (study aligned), −0.5–1.5 (rejected), −2.0 (burnout) |
| **Termination** | Employed (success) OR Energy = 0 (burnout) OR max_steps=200 |

### Sectors
| Sector | Base Demand | Notes |
|--------|------------|-------|
| Technology | 0.85 | NST1 digital economy priority |
| Business | 0.72 | SME and service sector |
| Agriculture | 0.55 | Large informal sector |
| Construction | 0.50 | Urban development |
| Healthcare | 0.68 | Vision 2050 priority |

---

## Algorithms Implemented

| Algorithm | Type | Library | Notes |
|-----------|------|---------|-------|
| **DQN** | Value-Based | Stable-Baselines3 | Experience replay, target network |
| **REINFORCE** | Policy Gradient | Manual PyTorch | Monte Carlo returns, optional baseline |
| **PPO** | Policy Gradient | Stable-Baselines3 | Clipped surrogate, actor-critic |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. View the random agent demo (no training required)
```bash
python static_demo.py
```

### 3. Train all models
```bash
python training/dqn_training.py        # DQN (10 hyperparameter runs)
python training/reinforce_algorithm.py # REINFORCE (10 runs)
python training/pg_training.py         # PPO (10 runs)
```

### 4. Compare algorithms
```bash
python evaluate.py
```

### 5. Run the best agent
```bash
python main.py                    # Auto-select best algorithm
python main.py --algo PPO         # Specific algorithm
python main.py --algo DQN --no-gui --episodes 10
```

---

## Hyperparameter Experiments

Each algorithm is trained with **10 different hyperparameter combinations** covering:

**DQN:** learning rate, gamma, batch size, buffer size, exploration fraction, target update interval  
**REINFORCE:** learning rate, gamma, hidden dim, baseline usage, entropy coefficient  
**PPO:** learning rate, gamma, n_steps, batch size, n_epochs, entropy coef, clip range  

Results saved to `results/` as CSV tables and JSON model info files.

---

## Outputs

After running all scripts, `plots/` will contain:
- `static_random_demo.png` — Random agent episode visualization
- `environment_diagram.png` — Environment architecture diagram
- `dqn_hyperparameter_analysis.png` — DQN training curves + analysis
- `reinforce_hyperparameter_analysis.png` — REINFORCE analysis
- `ppo_hyperparameter_analysis.png` — PPO analysis
- `algorithm_comparison.png` — Cross-algorithm comparison

---

## References

- NISR (2023). Labour Force Survey Annual Report 2023.
- World Economic Forum (2023). Future of Jobs Report.
- Mnih et al. (2015). Human-level control through deep reinforcement learning.
- Williams (1992). Simple statistical gradient-following algorithms.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Rwanda Development Board (2023). Investment and Employment Report.
