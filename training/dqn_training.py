"""
DQN Training — Kigali Job Market RL
======================================
Trains Deep Q-Network (DQN) agents on the KigaliJobMarketEnv using
Stable-Baselines3. Runs 10 hyperparameter combinations and logs
performance metrics for comparison.

Value-Based Method: DQN
  - Uses experience replay buffer
  - Target network for stable Q-value estimation
  - Epsilon-greedy exploration

Run:
    python training/dqn_training.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import KigaliJobMarketEnv

# ── Constants ─────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS   = 80_000   # training budget per run
EVAL_EPISODES     = 20
SAVE_DIR          = "models/dqn"
RESULTS_DIR       = "results"
PLOTS_DIR         = "plots"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# =============================================================================
# Reward-logging callback
# =============================================================================

class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards at each step for plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list  = []
        self.episode_lengths: list  = []
        self.timesteps: list        = []
        self._episode_reward: float = 0.0
        self._episode_length: int   = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self._episode_reward += reward
        self._episode_length += 1

        if done:
            self.episode_rewards.append(self._episode_reward)
            self.episode_lengths.append(self._episode_length)
            self.timesteps.append(self.num_timesteps)
            self._episode_reward = 0.0
            self._episode_length = 0
        return True


# =============================================================================
# 10 Hyperparameter Combinations
# =============================================================================

DQN_HYPERPARAMS = [
    # Run 1 — Baseline
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64,
         buffer_size=10_000, exploration_fraction=0.20,
         target_update_interval=500, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),

    # Run 2 — Lower LR
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64,
         buffer_size=10_000, exploration_fraction=0.20,
         target_update_interval=500, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),

    # Run 3 — Lower gamma (short-sighted)
    dict(learning_rate=1e-3, gamma=0.90, batch_size=64,
         buffer_size=10_000, exploration_fraction=0.20,
         target_update_interval=500, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),

    # Run 4 — Larger batch
    dict(learning_rate=1e-3, gamma=0.99, batch_size=128,
         buffer_size=10_000, exploration_fraction=0.20,
         target_update_interval=500, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),

    # Run 5 — Large replay buffer
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64,
         buffer_size=50_000, exploration_fraction=0.20,
         target_update_interval=500, learning_starts=2000,
         train_freq=4, policy="MlpPolicy"),

    # Run 6 — Low exploration (exploit early)
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64,
         buffer_size=10_000, exploration_fraction=0.10,
         target_update_interval=500, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),

    # Run 7 — High exploration + small batch
    dict(learning_rate=2e-4, gamma=0.99, batch_size=32,
         buffer_size=10_000, exploration_fraction=0.35,
         target_update_interval=1000, learning_starts=500,
         train_freq=8, policy="MlpPolicy"),

    # Run 8 — Large batch + large buffer
    dict(learning_rate=1e-3, gamma=0.99, batch_size=256,
         buffer_size=100_000, exploration_fraction=0.15,
         target_update_interval=1000, learning_starts=2000,
         train_freq=4, policy="MlpPolicy"),

    # Run 9 — Balanced moderate settings
    dict(learning_rate=5e-4, gamma=0.95, batch_size=128,
         buffer_size=50_000, exploration_fraction=0.25,
         target_update_interval=750, learning_starts=1500,
         train_freq=4, policy="MlpPolicy"),

    # Run 10 — Minimal exploration (greedy)
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64,
         buffer_size=10_000, exploration_fraction=0.05,
         target_update_interval=250, learning_starts=1000,
         train_freq=4, policy="MlpPolicy"),
]


# =============================================================================
# Training function
# =============================================================================

def train_dqn(run_id: int, params: dict, seed: int = 42) -> dict:
    """Train a DQN agent with given hyperparameters and return metrics."""
    print(f"\n{'─'*60}")
    print(f"  DQN Run {run_id+1}/10")
    print(f"  LR={params['learning_rate']} | γ={params['gamma']} | "
          f"batch={params['batch_size']} | buffer={params['buffer_size']} | "
          f"exp_frac={params['exploration_fraction']}")
    print(f"{'─'*60}")

    env      = Monitor(KigaliJobMarketEnv(max_steps=200))
    eval_env = Monitor(KigaliJobMarketEnv(max_steps=200))

    policy = params.pop("policy")

    model = DQN(
        policy,
        env,
        verbose=0,
        seed=seed,
        **params,
    )

    callback = RewardLoggerCallback()
    t_start = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=True,
    )

    elapsed = time.time() - t_start

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    # Q-value loss proxy (from SB3 logger if available)
    ep_rewards = callback.episode_rewards
    final_avg  = float(np.mean(ep_rewards[-20:])) if len(ep_rewards) >= 20 else float(np.mean(ep_rewards)) if ep_rewards else 0.0
    best_reward = float(np.max(ep_rewards)) if ep_rewards else 0.0

    params["policy"] = policy  # restore
    result = {
        "run":                  run_id + 1,
        "learning_rate":        params["learning_rate"],
        "gamma":                params["gamma"],
        "batch_size":           params["batch_size"],
        "buffer_size":          params["buffer_size"],
        "exploration_fraction": params["exploration_fraction"],
        "target_update_interval": params["target_update_interval"],
        "mean_eval_reward":     round(mean_reward, 3),
        "std_eval_reward":      round(std_reward, 3),
        "final_avg_reward":     round(final_avg, 3),
        "best_episode_reward":  round(best_reward, 3),
        "total_episodes":       len(ep_rewards),
        "train_time_s":         round(elapsed, 1),
        "episode_rewards":      ep_rewards,
        "timesteps":            callback.timesteps,
    }

    print(f"  ✅ Mean eval reward: {mean_reward:.3f} ± {std_reward:.3f} | "
          f"Best: {best_reward:.2f} | Time: {elapsed:.1f}s")

    # Save model
    model_path = os.path.join(SAVE_DIR, f"dqn_run{run_id+1}")
    model.save(model_path)
    print(f"  💾 Model saved → {model_path}")

    env.close()
    eval_env.close()

    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_dqn_results(all_results: list):
    """Generate training curves and hyperparameter comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#12162A")
    fig.suptitle(
        "DQN Hyperparameter Experiments — Kigali Job Market RL\n"
        "Jean Jacques JABO | ALU BSE 2026",
        color="white", fontsize=14, fontweight="bold"
    )

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 10))

    # ── 1. All training curves ───────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1C2234")
    for r, result in enumerate(all_results):
        rewards = result["episode_rewards"]
        if len(rewards) < 2:
            continue
        # Smooth with rolling window
        window = min(20, len(rewards))
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().values
        ax1.plot(smoothed, color=colors[r], label=f"Run {r+1}", lw=1.5, alpha=0.85)
    ax1.axhline(0, color="#555555", lw=0.8, ls="--")
    ax1.set_title("DQN Training Curves (smoothed)", color="white", fontsize=11)
    ax1.set_xlabel("Episode", color="#AAAAAA")
    ax1.set_ylabel("Episode Reward", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA")
    ax1.legend(facecolor="#2A2E42", labelcolor="white", fontsize=7, ncol=2)
    for sp in ax1.spines.values():
        sp.set_color("#3A4060")

    # ── 2. Mean eval reward per run ──────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1C2234")
    means = [r["mean_eval_reward"] for r in all_results]
    stds  = [r["std_eval_reward"]  for r in all_results]
    runs  = [r["run"] for r in all_results]
    bars = ax2.bar(runs, means, color=colors, alpha=0.9, yerr=stds,
                   capsize=4, error_kw=dict(color="white", lw=1.5))
    ax2.axhline(0, color="#555555", lw=0.8)
    best_run = runs[int(np.argmax(means))]
    ax2.bar(best_run, means[int(np.argmax(means))],
            color="#FFD700", alpha=1.0, label=f"Best: Run {best_run}")
    ax2.set_title("Mean Evaluation Reward per Hyperparameter Run", color="white", fontsize=11)
    ax2.set_xlabel("Run", color="#AAAAAA")
    ax2.set_ylabel("Mean Reward ± Std", color="#AAAAAA")
    ax2.tick_params(colors="#AAAAAA")
    ax2.legend(facecolor="#2A2E42", labelcolor="white")
    for sp in ax2.spines.values():
        sp.set_color("#3A4060")

    # ── 3. Learning rate effect ──────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1C2234")
    lrs    = [r["learning_rate"] for r in all_results]
    mr     = [r["mean_eval_reward"] for r in all_results]
    sc = ax3.scatter(lrs, mr, c=[r["gamma"] for r in all_results],
                     cmap="plasma", s=120, alpha=0.9, edgecolors="white", lw=0.8)
    for i, (lr, m) in enumerate(zip(lrs, mr)):
        ax3.annotate(f"R{i+1}", (lr, m), textcoords="offset points",
                     xytext=(5, 4), fontsize=8, color="#CCCCCC")
    cb3 = plt.colorbar(sc, ax=ax3)
    cb3.set_label("Gamma", color="white", fontsize=9)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white")
    ax3.set_xscale("log")
    ax3.set_title("Learning Rate vs Mean Reward (color = gamma)", color="white", fontsize=11)
    ax3.set_xlabel("Learning Rate (log scale)", color="#AAAAAA")
    ax3.set_ylabel("Mean Eval Reward", color="#AAAAAA")
    ax3.tick_params(colors="#AAAAAA")
    for sp in ax3.spines.values():
        sp.set_color("#3A4060")

    # ── 4. Buffer size vs Batch size heatmap ────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1C2234")
    buffers = sorted(set(r["buffer_size"]  for r in all_results))
    batches = sorted(set(r["batch_size"]   for r in all_results))
    mat = np.full((len(batches), len(buffers)), np.nan)
    for res in all_results:
        bi = batches.index(res["batch_size"])
        fi = buffers.index(res["buffer_size"])
        existing = mat[bi, fi]
        if np.isnan(existing) or res["mean_eval_reward"] > existing:
            mat[bi, fi] = res["mean_eval_reward"]
    im = ax4.imshow(mat, cmap="RdYlGn", aspect="auto")
    ax4.set_xticks(range(len(buffers)))
    ax4.set_xticklabels([f"{b//1000}k" for b in buffers], color="#CCCCCC")
    ax4.set_yticks(range(len(batches)))
    ax4.set_yticklabels(batches, color="#CCCCCC")
    ax4.set_xlabel("Buffer Size", color="#AAAAAA")
    ax4.set_ylabel("Batch Size", color="#AAAAAA")
    ax4.set_title("Reward Heatmap: Buffer × Batch Size", color="white", fontsize=11)
    cbar4 = plt.colorbar(im, ax=ax4)
    cbar4.set_label("Mean Reward", color="white", fontsize=9)
    cbar4.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar4.ax.yaxis.get_ticklabels(), color="white")
    # Annotate cells
    for i in range(len(batches)):
        for j in range(len(buffers)):
            if not np.isnan(mat[i, j]):
                ax4.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                         fontsize=9, color="black", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "dqn_hyperparameter_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ DQN plots saved → {out}")
    plt.close()


def save_results_table(all_results: list):
    """Save clean hyperparameter table to CSV."""
    rows = []
    for r in all_results:
        rows.append({
            "Run":                   r["run"],
            "Learning Rate":         r["learning_rate"],
            "Gamma":                 r["gamma"],
            "Batch Size":            r["batch_size"],
            "Buffer Size":           r["buffer_size"],
            "Exploration Fraction":  r["exploration_fraction"],
            "Target Update Interval":r["target_update_interval"],
            "Mean Eval Reward":      r["mean_eval_reward"],
            "Std Eval Reward":       r["std_eval_reward"],
            "Final Avg Reward":      r["final_avg_reward"],
            "Best Episode Reward":   r["best_episode_reward"],
            "Total Episodes":        r["total_episodes"],
            "Train Time (s)":        r["train_time_s"],
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "dqn_hyperparameter_table.csv")
    df.to_csv(out, index=False)
    print(f"\n📊 DQN results table saved → {out}")
    print("\n" + df[["Run","Learning Rate","Gamma","Batch Size",
                      "Buffer Size","Mean Eval Reward","Best Episode Reward"]].to_string())
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  DQN Training — Kigali Job Market RL")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print(f"  Total runs: 10  |  Timesteps per run: {TOTAL_TIMESTEPS:,}")
    print("=" * 65)

    all_results = []
    for i, params in enumerate(DQN_HYPERPARAMS):
        result = train_dqn(i, params.copy(), seed=42 + i)
        all_results.append(result)

    print("\n\n" + "=" * 65)
    print("  Saving results and plots...")
    print("=" * 65)

    df = save_results_table(all_results)
    plot_dqn_results(all_results)

    # Save best model identifier
    best_idx = int(np.argmax([r["mean_eval_reward"] for r in all_results]))
    best_info = {
        "algorithm": "DQN",
        "run": all_results[best_idx]["run"],
        "mean_reward": all_results[best_idx]["mean_eval_reward"],
        "model_path": f"models/dqn/dqn_run{all_results[best_idx]['run']}",
        "params": {
            k: v for k, v in DQN_HYPERPARAMS[best_idx].items()
            if k != "policy"
        }
    }
    with open(os.path.join(RESULTS_DIR, "dqn_best.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"\n🏆 Best DQN: Run {best_info['run']} | "
          f"Mean reward = {best_info['mean_reward']:.3f}")
    print(f"   Model path: {best_info['model_path']}")
