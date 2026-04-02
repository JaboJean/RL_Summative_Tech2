"""
REINFORCE Algorithm — Manual PyTorch Implementation
=====================================================
REINFORCE (Williams, 1992) is the foundational policy gradient algorithm.
It directly optimizes the policy by computing returns from full episodes
and updating parameters in the direction of higher-reward trajectories.

Update rule:
    ∇θ J(θ) = Σ_t G_t ∇θ log π(a_t | s_t; θ)

Where G_t is the discounted return from time t:
    G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}

This module provides:
  - PolicyNetwork: MLP that outputs action probabilities
  - REINFORCEAgent: Full training loop with 10 hyperparameter runs
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import KigaliJobMarketEnv

# ── Constants ─────────────────────────────────────────────────────────────────
N_EPISODES     = 600      # training episodes per run
EVAL_EPISODES  = 20
SAVE_DIR       = "models/pg"
RESULTS_DIR    = "results"
PLOTS_DIR      = "plots"

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Multi-layer perceptron policy for REINFORCE.

    Architecture:
        Input(13) → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(12) → Softmax
    """

    def __init__(self, obs_dim: int = 13, action_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Initialize weights (Xavier)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

    def get_action(self, state: np.ndarray):
        """Sample action and return (action, log_prob)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        probs   = self.forward(state_t)
        dist    = Categorical(probs)
        action  = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Compute log probs and entropy for a batch."""
        probs   = self.forward(states)
        dist    = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy


# =============================================================================
# REINFORCE Agent
# =============================================================================

class REINFORCEAgent:
    """REINFORCE with optional baseline (mean return subtraction)."""

    def __init__(
        self,
        obs_dim:    int   = 13,
        action_dim: int   = 12,
        hidden_dim: int   = 128,
        lr:         float = 1e-3,
        gamma:      float = 0.99,
        use_baseline: bool = True,
        entropy_coef: float = 0.01,
    ):
        self.gamma       = gamma
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef

        self.policy    = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)

        # Episode buffers
        self.log_probs: list = []
        self.rewards:   list = []
        self.entropies: list = []

    def select_action(self, state: np.ndarray):
        action, log_prob = self.policy.get_action(state)
        # Record log prob (will be used in update)
        probs_t  = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        dist     = Categorical(self.policy(probs_t))
        entropy  = dist.entropy()
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        return action

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns G_t for each time step."""
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns).to(DEVICE)
        if self.use_baseline:
            # Subtract mean (variance reduction)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    def update(self) -> float:
        """Update policy parameters using REINFORCE loss."""
        if not self.log_probs:
            return 0.0

        returns      = self.compute_returns()
        log_probs_t  = torch.stack(self.log_probs)
        entropies_t  = torch.stack(self.entropies)

        # Policy gradient loss: -E[G_t * log π(a_t|s_t)]
        pg_loss      = -(log_probs_t * returns).mean()
        entropy_loss = -entropies_t.mean()
        loss         = pg_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

        return loss.item()

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


# =============================================================================
# 10 Hyperparameter Combinations
# =============================================================================

REINFORCE_HYPERPARAMS = [
    # Run 1 — Baseline reference
    dict(lr=1e-3,  gamma=0.99, hidden_dim=128, use_baseline=True,  entropy_coef=0.01),
    # Run 2 — Lower learning rate
    dict(lr=5e-4,  gamma=0.99, hidden_dim=128, use_baseline=True,  entropy_coef=0.01),
    # Run 3 — Smaller gamma (short horizon)
    dict(lr=1e-3,  gamma=0.95, hidden_dim=128, use_baseline=True,  entropy_coef=0.01),
    # Run 4 — Larger network
    dict(lr=1e-3,  gamma=0.99, hidden_dim=256, use_baseline=True,  entropy_coef=0.01),
    # Run 5 — Smaller network
    dict(lr=1e-3,  gamma=0.99, hidden_dim=64,  use_baseline=True,  entropy_coef=0.01),
    # Run 6 — High learning rate
    dict(lr=2e-3,  gamma=0.99, hidden_dim=128, use_baseline=True,  entropy_coef=0.01),
    # Run 7 — Very short horizon gamma
    dict(lr=1e-3,  gamma=0.90, hidden_dim=128, use_baseline=True,  entropy_coef=0.01),
    # Run 8 — No baseline (vanilla REINFORCE)
    dict(lr=5e-4,  gamma=0.95, hidden_dim=256, use_baseline=False, entropy_coef=0.01),
    # Run 9 — High entropy for exploration
    dict(lr=1e-3,  gamma=0.99, hidden_dim=128, use_baseline=True,  entropy_coef=0.05),
    # Run 10 — Very low LR + large net
    dict(lr=2e-4,  gamma=0.99, hidden_dim=256, use_baseline=True,  entropy_coef=0.02),
]


# =============================================================================
# Training function
# =============================================================================

def train_reinforce(run_id: int, params: dict, seed: int = 42) -> dict:
    print(f"\n{'─'*60}")
    print(f"  REINFORCE Run {run_id+1}/10")
    print(f"  LR={params['lr']} | γ={params['gamma']} | "
          f"hidden={params['hidden_dim']} | baseline={params['use_baseline']} | "
          f"ent_coef={params['entropy_coef']}")
    print(f"{'─'*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = KigaliJobMarketEnv(max_steps=200)
    eval_env = KigaliJobMarketEnv(max_steps=200)

    agent = REINFORCEAgent(**params)

    episode_rewards = []
    episode_losses  = []
    entropy_log     = []
    t_start = time.time()

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        ep_entropies = []

        for _ in range(200):
            action = agent.select_action(obs)
            if agent.entropies:
                ep_entropies.append(agent.entropies[-1].item())
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            total_reward += reward
            if terminated or truncated:
                break

        loss = agent.update()
        agent.scheduler.step()

        episode_rewards.append(total_reward)
        episode_losses.append(loss)
        entropy_log.append(np.mean(ep_entropies) if ep_entropies else 0.0)

        if (ep + 1) % 100 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(f"    Episode {ep+1:4d} | Avg(50): {recent:+.3f} | "
                  f"Loss: {loss:.4f}")

    elapsed = time.time() - t_start

    # Evaluate
    eval_rewards = []
    for ep in range(EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=9999 + ep)
        total_r = 0.0
        agent.policy.eval()
        with torch.no_grad():
            for _ in range(200):
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                probs   = agent.policy(state_t)
                action  = probs.argmax(dim=-1).item()  # deterministic
                obs, r, done, trunc, _ = eval_env.step(action)
                total_r += r
                if done or trunc:
                    break
        eval_rewards.append(total_r)
        agent.policy.train()

    mean_reward = float(np.mean(eval_rewards))
    std_reward  = float(np.std(eval_rewards))
    final_avg   = float(np.mean(episode_rewards[-50:]))
    best_reward = float(np.max(episode_rewards))

    result = {
        "run":             run_id + 1,
        "learning_rate":   params["lr"],
        "gamma":           params["gamma"],
        "hidden_dim":      params["hidden_dim"],
        "use_baseline":    params["use_baseline"],
        "entropy_coef":    params["entropy_coef"],
        "mean_eval_reward":round(mean_reward, 3),
        "std_eval_reward": round(std_reward, 3),
        "final_avg_reward":round(final_avg, 3),
        "best_episode_reward": round(best_reward, 3),
        "total_episodes":  N_EPISODES,
        "train_time_s":    round(elapsed, 1),
        "episode_rewards": episode_rewards,
        "episode_losses":  episode_losses,
        "entropy_log":     entropy_log,
    }

    print(f"  ✅ Mean eval reward: {mean_reward:.3f} ± {std_reward:.3f} | "
          f"Best: {best_reward:.2f} | Time: {elapsed:.1f}s")

    # Save
    save_path = os.path.join(SAVE_DIR, f"reinforce_run{run_id+1}.pt")
    agent.save(save_path)
    print(f"  💾 Saved → {save_path}")

    env.close()
    eval_env.close()
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_reinforce_results(all_results: list):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#12162A")
    fig.suptitle(
        "REINFORCE Hyperparameter Experiments — Kigali Job Market RL\n"
        "Jean Jacques JABO | ALU BSE 2026",
        color="white", fontsize=14, fontweight="bold"
    )

    colors = plt.cm.cool(np.linspace(0.1, 0.9, 10))

    # ── 1. Training curves ───────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1C2234")
    for r, res in enumerate(all_results):
        rews = res["episode_rewards"]
        smoothed = pd.Series(rews).rolling(30, min_periods=1).mean().values
        ax1.plot(smoothed, color=colors[r], label=f"R{r+1}", lw=1.5, alpha=0.85)
    ax1.set_title("REINFORCE Training Curves (smoothed, window=30)", color="white", fontsize=11)
    ax1.set_xlabel("Episode", color="#AAAAAA")
    ax1.set_ylabel("Episode Reward", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA")
    ax1.legend(facecolor="#2A2E42", labelcolor="white", fontsize=7, ncol=2)
    ax1.axhline(0, color="#555555", lw=0.8, ls="--")
    for sp in ax1.spines.values(): sp.set_color("#3A4060")

    # ── 2. Entropy curves ────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1C2234")
    for r, res in enumerate(all_results):
        ent = res["entropy_log"]
        if ent:
            smoothed = pd.Series(ent).rolling(30, min_periods=1).mean().values
            ax2.plot(smoothed, color=colors[r], label=f"R{r+1}", lw=1.5, alpha=0.85)
    ax2.set_title("Policy Entropy Over Training (exploration signal)", color="white", fontsize=11)
    ax2.set_xlabel("Episode", color="#AAAAAA")
    ax2.set_ylabel("Entropy", color="#AAAAAA")
    ax2.tick_params(colors="#AAAAAA")
    ax2.legend(facecolor="#2A2E42", labelcolor="white", fontsize=7, ncol=2)
    for sp in ax2.spines.values(): sp.set_color("#3A4060")

    # ── 3. Eval reward bar chart ─────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1C2234")
    means = [r["mean_eval_reward"] for r in all_results]
    stds  = [r["std_eval_reward"]  for r in all_results]
    runs  = [r["run"] for r in all_results]
    ax3.bar(runs, means, color=colors, alpha=0.9, yerr=stds,
            capsize=4, error_kw=dict(color="white", lw=1.5))
    best = int(np.argmax(means))
    ax3.bar(runs[best], means[best], color="#FFD700", alpha=1.0, label=f"Best: Run {runs[best]}")
    ax3.set_title("Mean Evaluation Reward per Run", color="white", fontsize=11)
    ax3.set_xlabel("Run", color="#AAAAAA")
    ax3.set_ylabel("Mean Reward", color="#AAAAAA")
    ax3.tick_params(colors="#AAAAAA")
    ax3.legend(facecolor="#2A2E42", labelcolor="white")
    for sp in ax3.spines.values(): sp.set_color("#3A4060")

    # ── 4. Baseline vs No-Baseline comparison ────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1C2234")
    baseline_runs    = [r for r in all_results if r["use_baseline"]]
    no_baseline_runs = [r for r in all_results if not r["use_baseline"]]
    for r in baseline_runs:
        rews = r["episode_rewards"]
        smoothed = pd.Series(rews).rolling(30, min_periods=1).mean().values
        ax4.plot(smoothed, color="#46C864", lw=1.2, alpha=0.7)
    for r in no_baseline_runs:
        rews = r["episode_rewards"]
        smoothed = pd.Series(rews).rolling(30, min_periods=1).mean().values
        ax4.plot(smoothed, color="#FF6060", lw=2.0, alpha=0.9, ls="--", label="No Baseline")
    ax4.plot([], [], color="#46C864", label="With Baseline (mean subtraction)")
    ax4.set_title("Effect of Baseline on Training Stability", color="white", fontsize=11)
    ax4.set_xlabel("Episode", color="#AAAAAA")
    ax4.set_ylabel("Reward (smoothed)", color="#AAAAAA")
    ax4.tick_params(colors="#AAAAAA")
    ax4.legend(facecolor="#2A2E42", labelcolor="white")
    for sp in ax4.spines.values(): sp.set_color("#3A4060")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "reinforce_hyperparameter_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ REINFORCE plots saved → {out}")
    plt.close()


def save_reinforce_table(all_results: list):
    rows = []
    for r in all_results:
        rows.append({
            "Run":               r["run"],
            "Learning Rate":     r["learning_rate"],
            "Gamma":             r["gamma"],
            "Hidden Dim":        r["hidden_dim"],
            "Use Baseline":      r["use_baseline"],
            "Entropy Coef":      r["entropy_coef"],
            "Mean Eval Reward":  r["mean_eval_reward"],
            "Std Eval Reward":   r["std_eval_reward"],
            "Final Avg Reward":  r["final_avg_reward"],
            "Best Episode":      r["best_episode_reward"],
            "Episodes":          r["total_episodes"],
            "Train Time (s)":    r["train_time_s"],
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "reinforce_hyperparameter_table.csv")
    df.to_csv(out, index=False)
    print(f"\n📊 REINFORCE table saved → {out}")
    print("\n" + df[["Run","Learning Rate","Gamma","Hidden Dim",
                      "Mean Eval Reward","Best Episode"]].to_string())
    return df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  REINFORCE Training — Kigali Job Market RL")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print(f"  Device: {DEVICE} | Episodes per run: {N_EPISODES}")
    print("=" * 65)

    all_results = []
    for i, params in enumerate(REINFORCE_HYPERPARAMS):
        result = train_reinforce(i, params.copy(), seed=42 + i)
        all_results.append(result)

    df = save_reinforce_table(all_results)
    plot_reinforce_results(all_results)

    best_idx = int(np.argmax([r["mean_eval_reward"] for r in all_results]))
    best_info = {
        "algorithm":   "REINFORCE",
        "run":         all_results[best_idx]["run"],
        "mean_reward": all_results[best_idx]["mean_eval_reward"],
        "model_path":  f"models/pg/reinforce_run{all_results[best_idx]['run']}.pt",
        "params":      REINFORCE_HYPERPARAMS[best_idx],
    }
    with open(os.path.join(RESULTS_DIR, "reinforce_best.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"\n🏆 Best REINFORCE: Run {best_info['run']} | "
          f"Mean reward = {best_info['mean_reward']:.3f}")
