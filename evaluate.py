"""
evaluate.py — Cross-Algorithm Comparison & Analysis
======================================================
Loads saved models (DQN, REINFORCE, PPO) and produces:
  1. Side-by-side cumulative reward comparison
  2. Convergence analysis
  3. Generalization test (new seeds)
  4. Summary statistics table

Run AFTER training all models:
    python evaluate.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import KigaliJobMarketEnv
from stable_baselines3 import DQN, PPO

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PLOTS_DIR, exist_ok=True)


# =============================================================================
# Load best models
# =============================================================================

def load_best_models():
    """Load all best-performing models from saved JSON info files."""
    models = {}

    # DQN
    dqn_info_path = os.path.join(RESULTS_DIR, "dqn_best.json")
    if os.path.exists(dqn_info_path):
        with open(dqn_info_path) as f:
            info = json.load(f)
        try:
            models["DQN"] = {
                "model": DQN.load(info["model_path"]),
                "info": info,
                "type": "sb3"
            }
            print(f"✅ DQN loaded: Run {info['run']} | Reward: {info['mean_reward']:.3f}")
        except Exception as e:
            print(f"⚠️  DQN load failed: {e}")

    # PPO
    ppo_info_path = os.path.join(RESULTS_DIR, "ppo_best.json")
    if os.path.exists(ppo_info_path):
        with open(ppo_info_path) as f:
            info = json.load(f)
        try:
            models["PPO"] = {
                "model": PPO.load(info["model_path"]),
                "info": info,
                "type": "sb3"
            }
            print(f"✅ PPO loaded: Run {info['run']} | Reward: {info['mean_reward']:.3f}")
        except Exception as e:
            print(f"⚠️  PPO load failed: {e}")

    # REINFORCE
    rf_info_path = os.path.join(RESULTS_DIR, "reinforce_best.json")
    if os.path.exists(rf_info_path):
        with open(rf_info_path) as f:
            info = json.load(f)
        try:
            from training.reinforce_algorithm import REINFORCEAgent
            params = info["params"]
            agent = REINFORCEAgent(
                lr=params["lr"],
                gamma=params["gamma"],
                hidden_dim=params["hidden_dim"],
                use_baseline=params["use_baseline"],
                entropy_coef=params["entropy_coef"],
            )
            agent.load(info["model_path"])
            models["REINFORCE"] = {
                "model": agent,
                "info": info,
                "type": "reinforce"
            }
            print(f"✅ REINFORCE loaded: Run {info['run']} | Reward: {info['mean_reward']:.3f}")
        except Exception as e:
            print(f"⚠️  REINFORCE load failed: {e}")

    return models


# =============================================================================
# Evaluate a single model over multiple episodes
# =============================================================================

def evaluate_model(model_entry: dict, n_episodes: int = 50, seeds=None) -> dict:
    """Run evaluation episodes and collect metrics."""
    env = KigaliJobMarketEnv(max_steps=200)
    if seeds is None:
        seeds = list(range(5000, 5000 + n_episodes))

    rewards   = []
    lengths   = []
    employed  = []

    m_type = model_entry["type"]
    model  = model_entry["model"]

    for ep, seed in enumerate(seeds[:n_episodes]):
        obs, _ = env.reset(seed=seed)
        total_r = 0.0
        steps   = 0

        if m_type == "sb3":
            for _ in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
                total_r += r
                steps   += 1
                if done or trunc:
                    break
            rewards.append(total_r)
            lengths.append(steps)
            employed.append(info.get("employed", False))

        elif m_type == "reinforce":
            model.policy.eval()
            with torch.no_grad():
                for _ in range(200):
                    state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                    probs   = model.policy(state_t)
                    action  = probs.argmax(dim=-1).item()
                    obs, r, done, trunc, info = env.step(action)
                    total_r += r
                    steps   += 1
                    if done or trunc:
                        break
            rewards.append(total_r)
            lengths.append(steps)
            employed.append(info.get("employed", False))

    env.close()
    return {
        "rewards":        rewards,
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards)),
        "median_reward":  float(np.median(rewards)),
        "mean_length":    float(np.mean(lengths)),
        "employment_rate": float(np.mean(employed)),
        "success_count":  sum(employed),
    }


# =============================================================================
# Master comparison plot
# =============================================================================

def plot_comparison(models: dict, eval_results: dict):
    """Comprehensive multi-panel comparison of all algorithms."""
    fig = plt.figure(figsize=(20, 14), facecolor="#12162A")
    fig.suptitle(
        "RL Algorithm Comparison — Kigali Job Market\n"
        "DQN vs REINFORCE vs PPO  |  Jean Jacques JABO | ALU BSE 2026",
        color="white", fontsize=16, fontweight="bold", y=0.98
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.07, left=0.06, right=0.97)

    algo_colors = {
        "DQN":       "#4680FF",
        "REINFORCE": "#46C864",
        "PPO":       "#FFB932",
    }

    # ── Load training CSVs if available ─────────────────────────────────────
    training_data = {}
    csv_map = {
        "DQN":       os.path.join(RESULTS_DIR, "dqn_hyperparameter_table.csv"),
        "REINFORCE": os.path.join(RESULTS_DIR, "reinforce_hyperparameter_table.csv"),
        "PPO":       os.path.join(RESULTS_DIR, "ppo_hyperparameter_table.csv"),
    }

    # ── 1. Eval reward comparison (bar) ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1C2234")
    algos = list(eval_results.keys())
    means = [eval_results[a]["mean_reward"]    for a in algos]
    stds  = [eval_results[a]["std_reward"]     for a in algos]
    cols  = [algo_colors.get(a, "#AAAAAA")     for a in algos]
    bars = ax1.bar(algos, means, color=cols, alpha=0.9, yerr=stds,
                   capsize=6, error_kw=dict(color="white", lw=2), width=0.5)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{mean:.2f}", ha="center", va="bottom", color="white",
                 fontsize=12, fontweight="bold")
    ax1.set_title("Algorithm Comparison: Mean Evaluation Reward", color="white", fontsize=12)
    ax1.set_ylabel("Mean Reward ± Std (50 episodes)", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA", labelsize=13)
    ax1.axhline(0, color="#555555", lw=0.8)
    for sp in ax1.spines.values(): sp.set_color("#3A4060")

    # ── 2. Employment success rate ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1C2234")
    emp_rates = [eval_results[a]["employment_rate"] * 100 for a in algos]
    wedge_colors = [algo_colors.get(a, "#AAAAAA") for a in algos]
    bars2 = ax2.bar(algos, emp_rates, color=wedge_colors, alpha=0.9, width=0.5)
    for bar, rate in zip(bars2, emp_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{rate:.1f}%", ha="center", va="bottom", color="white",
                 fontsize=11, fontweight="bold")
    ax2.set_title("Job Success Rate (%)", color="white", fontsize=12)
    ax2.set_ylabel("% Episodes Employed", color="#AAAAAA")
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors="#AAAAAA", labelsize=11)
    for sp in ax2.spines.values(): sp.set_color("#3A4060")

    # ── 3. Reward distribution boxplot ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1C2234")
    data = [eval_results[a]["rewards"] for a in algos]
    bp = ax3.boxplot(data, labels=algos, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color="#AAAAAA"),
                     capprops=dict(color="#AAAAAA"),
                     flierprops=dict(marker="o", color="#FF6060", alpha=0.5))
    for patch, col in zip(bp["boxes"], [algo_colors.get(a, "#AAAAAA") for a in algos]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax3.set_title("Reward Distribution (50 eval episodes each)", color="white", fontsize=12)
    ax3.set_ylabel("Episode Reward", color="#AAAAAA")
    ax3.tick_params(colors="#AAAAAA", labelsize=12)
    ax3.axhline(0, color="#555555", lw=0.8, ls="--")
    for sp in ax3.spines.values(): sp.set_color("#3A4060")

    # ── 4. Episode length comparison ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1C2234")
    lengths = [eval_results[a]["mean_length"] for a in algos]
    bars4 = ax4.bar(algos, lengths, color=wedge_colors, alpha=0.8, width=0.5)
    for bar, length in zip(bars4, lengths):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{length:.0f}", ha="center", va="bottom", color="white",
                 fontsize=11, fontweight="bold")
    ax4.axhline(200, color="#FF6060", lw=1, ls="--", alpha=0.7, label="Max steps")
    ax4.set_title("Avg Episode Length", color="white", fontsize=12)
    ax4.set_ylabel("Steps", color="#AAAAAA")
    ax4.tick_params(colors="#AAAAAA", labelsize=11)
    ax4.legend(facecolor="#2A2E42", labelcolor="white", fontsize=9)
    for sp in ax4.spines.values(): sp.set_color("#3A4060")

    # ── 5. Best hyperparameter performance from each algorithm ───────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor("#1C2234")

    for algo, csv_path in csv_map.items():
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                reward_col = "Mean Eval Reward"
                if reward_col in df.columns:
                    vals = df[reward_col].values
                    x    = range(1, len(vals) + 1)
                    color = algo_colors.get(algo, "#AAAAAA")
                    ax5.plot(x, vals, "o-", color=color, lw=2.5,
                             markersize=7, label=algo, alpha=0.9)
                    # Mark best
                    best_idx = int(np.argmax(vals))
                    ax5.plot(x[best_idx], vals[best_idx], "*",
                             color=color, markersize=18, alpha=1.0,
                             markeredgecolor="white", markeredgewidth=0.8)
            except Exception as e:
                print(f"  ⚠ Could not plot {algo} CSV: {e}")

    ax5.axhline(0, color="#555555", lw=0.8, ls="--")
    ax5.set_title(
        "Mean Eval Reward Across 10 Hyperparameter Runs per Algorithm  "
        "(★ = Best Run)",
        color="white", fontsize=12
    )
    ax5.set_xlabel("Hyperparameter Run", color="#AAAAAA")
    ax5.set_ylabel("Mean Eval Reward", color="#AAAAAA")
    ax5.set_xticks(range(1, 11))
    ax5.tick_params(colors="#AAAAAA")
    ax5.legend(facecolor="#2A2E42", labelcolor="white", fontsize=12)
    for sp in ax5.spines.values(): sp.set_color("#3A4060")

    out = os.path.join(PLOTS_DIR, "algorithm_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ Comparison plot saved → {out}")
    plt.close()


def print_summary_table(eval_results: dict):
    """Print and save final summary table."""
    rows = []
    for algo, res in eval_results.items():
        rows.append({
            "Algorithm":        algo,
            "Mean Reward":      round(res["mean_reward"],   3),
            "Std Reward":       round(res["std_reward"],    3),
            "Median Reward":    round(res["median_reward"], 3),
            "Employment Rate":  f"{res['employment_rate']*100:.1f}%",
            "Avg Episode Len":  round(res["mean_length"],   1),
            "Successful Jobs":  res["success_count"],
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "final_comparison_table.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 65)
    print("  FINAL ALGORITHM COMPARISON — Kigali Job Market RL")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\n📊 Table saved → {out}")

    best = df.loc[df["Mean Reward"].idxmax(), "Algorithm"]
    print(f"\n🏆 Best Overall Algorithm: {best}")
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Cross-Algorithm Evaluation — Kigali Job Market RL")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print("=" * 65)

    print("\n[1/3] Loading best models...")
    models = load_best_models()

    if not models:
        print("\n⚠️  No trained models found. Please run training scripts first:")
        print("    python training/dqn_training.py")
        print("    python training/pg_training.py")
        print("    python training/reinforce_algorithm.py")
        sys.exit(1)

    print(f"\n[2/3] Evaluating {len(models)} models (50 episodes each)...")
    eval_results = {}
    for algo, model_entry in models.items():
        print(f"  Evaluating {algo}...")
        eval_results[algo] = evaluate_model(model_entry, n_episodes=50)
        print(f"    → Mean: {eval_results[algo]['mean_reward']:.3f} | "
              f"Success: {eval_results[algo]['employment_rate']*100:.1f}%")

    print("\n[3/3] Generating plots and summary...")
    plot_comparison(models, eval_results)
    print_summary_table(eval_results)
