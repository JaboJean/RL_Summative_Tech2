"""
PPO Training — Kigali Job Market RL
======================================
Trains Proximal Policy Optimization (PPO) agents on KigaliJobMarketEnv
using Stable-Baselines3. Runs 10 hyperparameter combinations.

PPO is an on-policy policy gradient method that:
- Uses a clipped surrogate objective to prevent large policy updates
- Combines actor-critic architecture
- Is sample-efficient and stable compared to vanilla REINFORCE

Run:
    python training/pg_training.py
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import KigaliJobMarketEnv

# ── Constants ─────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 80_000
EVAL_EPISODES   = 20
SAVE_DIR        = "models/pg"
RESULTS_DIR     = "results"
PLOTS_DIR       = "plots"

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)


# =============================================================================
# Callback
# =============================================================================

class PPORewardCallback(BaseCallback):
    """Collect episode rewards and policy entropy during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards:  list  = []
        self.episode_lengths:  list  = []
        self.entropy_log:      list  = []
        self.value_loss_log:   list  = []
        self.policy_loss_log:  list  = []
        self._ep_reward:       float = 0.0
        self._ep_length:       int   = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self._ep_reward += reward
        self._ep_length += 1

        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self._ep_reward = 0.0
            self._ep_length = 0

        # Log entropy and loss if available from SB3 logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                logs = self.model.logger.name_to_value
                if "train/entropy_loss" in logs:
                    self.entropy_log.append(logs["train/entropy_loss"])
                if "train/value_loss" in logs:
                    self.value_loss_log.append(logs["train/value_loss"])
                if "train/policy_gradient_loss" in logs:
                    self.policy_loss_log.append(logs["train/policy_gradient_loss"])
            except Exception:
                pass
        return True


# =============================================================================
# 10 Hyperparameter Combinations
# =============================================================================

PPO_HYPERPARAMS = [
    # Run 1 — SB3 defaults (baseline)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 2 — Lower learning rate
    dict(learning_rate=1e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 3 — Lower gamma
    dict(learning_rate=3e-4, gamma=0.95, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 4 — Short rollout
    dict(learning_rate=3e-4, gamma=0.99, n_steps=512, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 5 — Large mini-batch
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=256,
         n_epochs=10, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 6 — More epochs per update
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=20, ent_coef=0.01, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 7 — High entropy coefficient (exploration)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.05, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 8 — Larger clip range (allows bigger policy updates)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, clip_range=0.3, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 9 — Balanced moderate: good LR, 1024 steps, 128 batch
    dict(learning_rate=5e-4, gamma=0.99, n_steps=1024, batch_size=128,
         n_epochs=15, ent_coef=0.02, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),

    # Run 10 — Long rollout + zero entropy (exploit)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=4096, batch_size=64,
         n_epochs=10, ent_coef=0.0, clip_range=0.2, vf_coef=0.5,
         max_grad_norm=0.5, policy="MlpPolicy"),
]


# =============================================================================
# Training function
# =============================================================================

def train_ppo(run_id: int, params: dict, seed: int = 42) -> dict:
    print(f"\n{'─'*60}")
    print(f"  PPO Run {run_id+1}/10")
    print(f"  LR={params['learning_rate']} | γ={params['gamma']} | "
          f"n_steps={params['n_steps']} | batch={params['batch_size']} | "
          f"n_epochs={params['n_epochs']} | ent={params['ent_coef']} | "
          f"clip={params['clip_range']}")
    print(f"{'─'*60}")

    env      = Monitor(KigaliJobMarketEnv(max_steps=200))
    eval_env = Monitor(KigaliJobMarketEnv(max_steps=200))

    policy = params.pop("policy")

    model = PPO(
        policy,
        env,
        verbose=0,
        seed=seed,
        **params,
    )

    callback = PPORewardCallback()
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

    ep_rewards = callback.episode_rewards
    final_avg  = float(np.mean(ep_rewards[-20:])) if len(ep_rewards) >= 20 else float(np.mean(ep_rewards)) if ep_rewards else 0.0
    best_reward = float(np.max(ep_rewards)) if ep_rewards else 0.0

    params["policy"] = policy  # restore
    result = {
        "run":             run_id + 1,
        "learning_rate":   params["learning_rate"],
        "gamma":           params["gamma"],
        "n_steps":         params["n_steps"],
        "batch_size":      params["batch_size"],
        "n_epochs":        params["n_epochs"],
        "ent_coef":        params["ent_coef"],
        "clip_range":      params["clip_range"],
        "mean_eval_reward":round(mean_reward, 3),
        "std_eval_reward": round(std_reward, 3),
        "final_avg_reward":round(final_avg, 3),
        "best_episode_reward": round(best_reward, 3),
        "total_episodes":  len(ep_rewards),
        "train_time_s":    round(elapsed, 1),
        "episode_rewards": ep_rewards,
        "entropy_log":     callback.entropy_log,
        "value_loss_log":  callback.value_loss_log,
        "policy_loss_log": callback.policy_loss_log,
    }

    print(f"  ✅ Mean eval reward: {mean_reward:.3f} ± {std_reward:.3f} | "
          f"Best: {best_reward:.2f} | Time: {elapsed:.1f}s")

    model_path = os.path.join(SAVE_DIR, f"ppo_run{run_id+1}")
    model.save(model_path)
    print(f"  💾 Saved → {model_path}")

    env.close()
    eval_env.close()
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_ppo_results(all_results: list):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#12162A")
    fig.suptitle(
        "PPO Hyperparameter Experiments — Kigali Job Market RL\n"
        "Jean Jacques JABO | ALU BSE 2026",
        color="white", fontsize=14, fontweight="bold"
    )

    colors = plt.cm.spring(np.linspace(0.1, 0.9, 10))

    # ── 1. Training curves ───────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1C2234")
    for r, res in enumerate(all_results):
        rews = res["episode_rewards"]
        if len(rews) < 2:
            continue
        sm = pd.Series(rews).rolling(20, min_periods=1).mean().values
        ax1.plot(sm, color=colors[r], label=f"R{r+1}", lw=1.5, alpha=0.85)
    ax1.axhline(0, color="#555555", lw=0.8, ls="--")
    ax1.set_title("PPO Training Curves (smoothed)", color="white", fontsize=11)
    ax1.set_xlabel("Episode", color="#AAAAAA")
    ax1.set_ylabel("Episode Reward", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA")
    ax1.legend(facecolor="#2A2E42", labelcolor="white", fontsize=7, ncol=2)
    for sp in ax1.spines.values(): sp.set_color("#3A4060")

    # ── 2. Eval reward comparison ────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1C2234")
    means = [r["mean_eval_reward"] for r in all_results]
    stds  = [r["std_eval_reward"]  for r in all_results]
    runs  = [r["run"] for r in all_results]
    ax2.bar(runs, means, color=colors, alpha=0.9, yerr=stds,
            capsize=4, error_kw=dict(color="white", lw=1.5))
    best = int(np.argmax(means))
    ax2.bar(runs[best], means[best], color="#FFD700", alpha=1.0,
            label=f"Best: Run {runs[best]}")
    ax2.set_title("Mean Evaluation Reward", color="white", fontsize=11)
    ax2.set_xlabel("Run", color="#AAAAAA")
    ax2.set_ylabel("Mean Reward ± Std", color="#AAAAAA")
    ax2.tick_params(colors="#AAAAAA")
    ax2.legend(facecolor="#2A2E42", labelcolor="white")
    for sp in ax2.spines.values(): sp.set_color("#3A4060")

    # ── 3. n_steps vs n_epochs effect ───────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1C2234")
    n_steps_vals  = [r["n_steps"]   for r in all_results]
    n_epochs_vals = [r["n_epochs"]  for r in all_results]
    sc = ax3.scatter(n_steps_vals, means, c=n_epochs_vals,
                     cmap="cool", s=140, alpha=0.9, edgecolors="white", lw=0.8)
    for i, (ns, m) in enumerate(zip(n_steps_vals, means)):
        ax3.annotate(f"R{i+1}", (ns, m), textcoords="offset points",
                     xytext=(5, 4), fontsize=8, color="#CCCCCC")
    cb = plt.colorbar(sc, ax=ax3)
    cb.set_label("n_epochs", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax3.set_title("n_steps vs Mean Reward (color = n_epochs)", color="white", fontsize=11)
    ax3.set_xlabel("n_steps (rollout length)", color="#AAAAAA")
    ax3.set_ylabel("Mean Eval Reward", color="#AAAAAA")
    ax3.tick_params(colors="#AAAAAA")
    for sp in ax3.spines.values(): sp.set_color("#3A4060")

    # ── 4. Entropy coefficient effect ────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1C2234")
    ent_coefs = sorted(set(r["ent_coef"] for r in all_results))
    ent_means = []
    for ec in ent_coefs:
        group = [r["mean_eval_reward"] for r in all_results if r["ent_coef"] == ec]
        ent_means.append(np.mean(group))
    bars4 = ax4.bar(range(len(ent_coefs)), ent_means,
                    color=["#46C864", "#FFB932", "#FF6060"][:len(ent_coefs)], alpha=0.85)
    ax4.set_xticks(range(len(ent_coefs)))
    ax4.set_xticklabels([f"ent={ec}" for ec in ent_coefs], color="#CCCCCC", fontsize=9)
    ax4.set_title("Effect of Entropy Coefficient on Reward", color="white", fontsize=11)
    ax4.set_ylabel("Avg Mean Reward", color="#AAAAAA")
    ax4.tick_params(colors="#AAAAAA")
    for sp in ax4.spines.values(): sp.set_color("#3A4060")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "ppo_hyperparameter_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ PPO plots saved → {out}")
    plt.close()


def save_ppo_table(all_results: list):
    rows = []
    for r in all_results:
        rows.append({
            "Run":               r["run"],
            "Learning Rate":     r["learning_rate"],
            "Gamma":             r["gamma"],
            "n_steps":           r["n_steps"],
            "Batch Size":        r["batch_size"],
            "n_epochs":          r["n_epochs"],
            "Entropy Coef":      r["ent_coef"],
            "Clip Range":        r["clip_range"],
            "Mean Eval Reward":  r["mean_eval_reward"],
            "Std Eval Reward":   r["std_eval_reward"],
            "Final Avg Reward":  r["final_avg_reward"],
            "Best Episode":      r["best_episode_reward"],
            "Total Episodes":    r["total_episodes"],
            "Train Time (s)":    r["train_time_s"],
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "ppo_hyperparameter_table.csv")
    df.to_csv(out, index=False)
    print(f"\n📊 PPO table saved → {out}")
    print("\n" + df[["Run","Learning Rate","Gamma","n_steps","n_epochs",
                      "Entropy Coef","Mean Eval Reward","Best Episode"]].to_string())
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PPO Training — Kigali Job Market RL")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print(f"  Total runs: 10  |  Timesteps per run: {TOTAL_TIMESTEPS:,}")
    print("=" * 65)

    all_results = []
    for i, params in enumerate(PPO_HYPERPARAMS):
        result = train_ppo(i, params.copy(), seed=42 + i)
        all_results.append(result)

    df = save_ppo_table(all_results)
    plot_ppo_results(all_results)

    best_idx = int(np.argmax([r["mean_eval_reward"] for r in all_results]))
    best_info = {
        "algorithm":   "PPO",
        "run":         all_results[best_idx]["run"],
        "mean_reward": all_results[best_idx]["mean_eval_reward"],
        "model_path":  f"models/pg/ppo_run{all_results[best_idx]['run']}",
        "params":      {k: v for k, v in PPO_HYPERPARAMS[best_idx].items() if k != "policy"},
    }
    with open(os.path.join(RESULTS_DIR, "ppo_best.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"\n🏆 Best PPO: Run {best_info['run']} | "
          f"Mean reward = {best_info['mean_reward']:.3f}")
