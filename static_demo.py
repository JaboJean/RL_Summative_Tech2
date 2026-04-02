"""
static_demo.py — Random Agent Visualization
=============================================
Demonstrates the KigaliJobMarketEnv visualization WITHOUT any trained model.
The agent takes purely random actions, showing all components of the environment
in action: sector bars, energy depletion, market dynamics, and the action log.

This satisfies the requirement:
  "Create a static file that shows the agent taking random actions
   (not using a model) in the custom environment."

Run:
    python static_demo.py
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import KigaliJobMarketEnv


# =============================================================================
# 1. Run a random-agent episode and collect trajectory data
# =============================================================================

def run_random_episode(seed: int = 42, max_steps: int = 200) -> dict:
    """Run one full episode with a uniformly random agent."""
    env = KigaliJobMarketEnv(render_mode=None, max_steps=max_steps)
    obs, info = env.reset(seed=seed)

    trajectory = {
        "steps": [], "rewards": [], "cumulative_rewards": [],
        "skills": [], "demands": [], "energies": [],
        "actions": [], "action_names": [], "terminated_at": None,
        "employed": False,
    }

    cumulative = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()        # ← purely random
        obs, reward, terminated, truncated, info = env.step(action)

        cumulative += reward
        trajectory["steps"].append(step)
        trajectory["rewards"].append(reward)
        trajectory["cumulative_rewards"].append(cumulative)
        trajectory["skills"].append(info["skill_levels"].copy())
        trajectory["demands"].append(info["market_demand"].copy())
        trajectory["energies"].append(info["energy"])
        trajectory["actions"].append(action)
        trajectory["action_names"].append(env.action_description(action))

        if terminated or truncated:
            trajectory["terminated_at"] = step
            trajectory["employed"] = info["employed"]
            break

    env.close()
    return trajectory


# =============================================================================
# 2. Static visualization — multi-panel matplotlib figure
# =============================================================================

SECTOR_COLORS_HEX = ["#4680FF", "#FFB932", "#46C864", "#C86440", "#B45AE0"]
SECTORS = ["Tech", "Business", "Agriculture", "Construction", "Healthcare"]

ACTION_COLORS = {
    "study":   "#46C864",
    "apply":   "#4680FF",
    "network": "#FFB932",
    "rest":    "#888888",
}

def action_category(action: int) -> str:
    if action < 5:    return "study"
    elif action < 10: return "apply"
    elif action == 10: return "network"
    else:             return "rest"


def render_static_demo(trajectory: dict, output_path: str = "plots/static_random_demo.png"):
    """Produce a comprehensive static visualization of the random agent episode."""
    steps = trajectory["steps"]
    rewards = trajectory["rewards"]
    cum_rewards = trajectory["cumulative_rewards"]
    skills = np.array(trajectory["skills"])
    demands = np.array(trajectory["demands"])
    energies = trajectory["energies"]
    actions = trajectory["actions"]

    fig = plt.figure(figsize=(20, 14), facecolor="#12162A")
    fig.suptitle(
        "Kigali Job Market RL — Random Agent Demo\n"
        "Jean Jacques JABO | BSE Pre-Capstone | ALU 2026",
        fontsize=16, color="white", fontweight="bold", y=0.98
    )

    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.07, left=0.06, right=0.97)

    # ── 1. Cumulative Reward ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1C2234")
    ax1.plot(steps, cum_rewards, color="#46FF82", lw=2, label="Cumulative Reward")
    ax1.axhline(0, color="#555555", lw=0.8, ls="--")
    if trajectory["terminated_at"]:
        t = trajectory["terminated_at"]
        ax1.axvline(t, color="#FF4040", lw=1.5, ls="--", alpha=0.7, label=f"Episode End (step {t})")
    ax1.set_title("Cumulative Reward — Random Agent", color="white", fontsize=11)
    ax1.set_xlabel("Step", color="#AAAAAA")
    ax1.set_ylabel("Reward", color="#AAAAAA")
    ax1.tick_params(colors="#AAAAAA")
    ax1.legend(facecolor="#2A2E42", labelcolor="white", fontsize=9)
    for sp in ax1.spines.values():
        sp.set_color("#3A4060")

    # ── 2. Per-Step Reward ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor("#1C2234")
    bar_colors = [
        "#46C864" if r >= 0 else "#FF4040" for r in rewards
    ]
    ax2.bar(steps, rewards, color=bar_colors, alpha=0.8, width=1.0)
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_title("Per-Step Rewards", color="white", fontsize=11)
    ax2.set_xlabel("Step", color="#AAAAAA")
    ax2.set_ylabel("Reward", color="#AAAAAA")
    ax2.tick_params(colors="#AAAAAA")
    for sp in ax2.spines.values():
        sp.set_color("#3A4060")
    patches = [
        mpatches.Patch(color="#46C864", label="Positive"),
        mpatches.Patch(color="#FF4040", label="Negative"),
    ]
    ax2.legend(handles=patches, facecolor="#2A2E42", labelcolor="white", fontsize=9)

    # ── 3. Skill Evolution per Sector ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1C2234")
    for i, (sector, color) in enumerate(zip(SECTORS, SECTOR_COLORS_HEX)):
        ax3.plot(steps, skills[:, i], label=sector, color=color, lw=1.8, alpha=0.9)
        # Plot demand as dashed line
        ax3.plot(steps, demands[:, i], color=color, lw=0.8, ls="--", alpha=0.4)
    ax3.set_title("Skill Levels vs Market Demand (dashed)", color="white", fontsize=11)
    ax3.set_xlabel("Step", color="#AAAAAA")
    ax3.set_ylabel("Level (0–1)", color="#AAAAAA")
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(colors="#AAAAAA")
    ax3.legend(facecolor="#2A2E42", labelcolor="white", fontsize=8, ncol=2)
    for sp in ax3.spines.values():
        sp.set_color("#3A4060")

    # ── 4. Energy Over Time ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_facecolor("#1C2234")
    energy_colors = []
    for e in energies:
        if e > 0.6:
            energy_colors.append("#46C864")
        elif e > 0.3:
            energy_colors.append("#FFB932")
        else:
            energy_colors.append("#FF4040")
    ax4.fill_between(steps, energies, alpha=0.4, color="#4680FF")
    ax4.plot(steps, energies, color="#4680FF", lw=2)
    ax4.axhline(0.2, color="#FF4040", lw=1, ls="--", alpha=0.7, label="Low energy threshold")
    ax4.set_title("Agent Energy Over Time", color="white", fontsize=11)
    ax4.set_xlabel("Step", color="#AAAAAA")
    ax4.set_ylabel("Energy", color="#AAAAAA")
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(colors="#AAAAAA")
    ax4.legend(facecolor="#2A2E42", labelcolor="white", fontsize=9)
    for sp in ax4.spines.values():
        sp.set_color("#3A4060")

    # ── 5. Action Distribution ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor("#1C2234")
    action_counts = np.bincount(actions, minlength=12)
    action_labels = [KigaliJobMarketEnv.action_description(a) for a in range(12)]
    bar_cols = [
        ACTION_COLORS[action_category(a)] for a in range(12)
    ]
    bars = ax5.bar(range(12), action_counts, color=bar_cols, alpha=0.85)
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(action_labels, rotation=45, ha="right", fontsize=8, color="#CCCCCC")
    ax5.set_title("Action Distribution (Random Agent)", color="white", fontsize=11)
    ax5.set_ylabel("Count", color="#AAAAAA")
    ax5.tick_params(colors="#AAAAAA")
    for sp in ax5.spines.values():
        sp.set_color("#3A4060")
    # Category legend
    cat_patches = [
        mpatches.Patch(color="#46C864", label="Study"),
        mpatches.Patch(color="#4680FF", label="Apply"),
        mpatches.Patch(color="#FFB932", label="Network"),
        mpatches.Patch(color="#888888", label="Rest"),
    ]
    ax5.legend(handles=cat_patches, facecolor="#2A2E42", labelcolor="white", fontsize=9)

    # ── 6. Final State Snapshot ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.set_facecolor("#1C2234")
    if len(skills) > 0:
        final_skills  = skills[-1]
        final_demands = demands[-1]
        thresholds    = final_demands * KigaliJobMarketEnv.HIRE_THRESHOLD_RATIO
        x_pos = np.arange(len(SECTORS))
        width = 0.28
        ax6.bar(x_pos - width, final_skills, width, label="Final Skill",
                color=SECTOR_COLORS_HEX, alpha=0.9)
        ax6.bar(x_pos,         final_demands, width, label="Market Demand",
                color="#FF6060", alpha=0.75)
        ax6.bar(x_pos + width, thresholds, width, label="Hire Threshold",
                color="#FFB932", alpha=0.75)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(SECTORS, color="#CCCCCC", fontsize=9)
        ax6.set_ylim(0, 1.1)
        ax6.set_title("Final State: Skills vs Demand vs Threshold", color="white", fontsize=11)
        ax6.set_ylabel("Level", color="#AAAAAA")
        ax6.tick_params(colors="#AAAAAA")
        ax6.legend(facecolor="#2A2E42", labelcolor="white", fontsize=9)
        for sp in ax6.spines.values():
            sp.set_color("#3A4060")

        # Mark sectors where agent is hireable
        for i, (s, t) in enumerate(zip(final_skills, thresholds)):
            if s >= t:
                ax6.annotate("✓", xy=(i - width, s), ha="center",
                             fontsize=14, color="#46FF82")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✅ Static demo saved → {output_path}")
    return fig


# =============================================================================
# 3. Environment Architecture Diagram
# =============================================================================

def render_env_diagram(output_path: str = "plots/environment_diagram.png"):
    """Render a professional diagram of the environment's components."""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor="#12162A")
    ax.set_facecolor("#12162A")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, color, sub="", fontsize=11):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub,
                    ha="center", va="center", fontsize=8, color="#CCCCCC")

    def arrow(x1, y1, x2, y2, label="", color="#AAAAAA"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my, label, color=color, fontsize=8, ha="left")

    # Central Agent
    box(6.5, 4.2, 3, 1.5, "RL AGENT", "#4046CC", "Policy Network\n(DQN / PPO / REINFORCE)", fontsize=10)

    # Environment box
    box(0.5, 7.5, 15, 2, "", "#1C2234", fontsize=9)
    ax.text(8, 9.2, "KIGALI JOB MARKET ENVIRONMENT  (KigaliJobMarketEnv — gymnasium.Env)",
            ha="center", va="center", fontsize=11, color="#FFD700", fontweight="bold")

    # Observation components
    obs_items = [
        ("Skill Levels\n(5 sectors)", 0.7, 5.8, "#4680FF"),
        ("Market Demand\n(5 sectors)", 2.5, 5.8, "#FF9830"),
        ("Energy Level", 4.3, 5.8, "#46C864"),
        ("Time Elapsed", 6.1, 5.8, "#B45AE0"),
        ("Employed\nStatus", 7.9, 5.8, "#FF4040"),
    ]
    for label, x, y, color in obs_items:
        box(x, y, 1.6, 1.1, label, color, fontsize=8)
        arrow(x + 0.8, y, 8, 4.2 + 0.75, color=color)  # → agent

    # Action components
    act_items = [
        ("Study\nSkills (×5)", 10.0, 5.8, "#46C864"),
        ("Apply\nJobs (×5)", 11.7, 5.8, "#4680FF"),
        ("Network", 13.4, 5.8, "#FFB932"),
        ("Rest", 14.7, 5.8, "#888888"),
    ]
    for label, x, y, color in act_items:
        box(x, y, 1.5, 1.1, label, color, fontsize=8)
        arrow(9.5, 4.2 + 0.75, x + 0.75, y + 1.1, color=color)  # agent →

    # Reward signals
    box(5.5, 1.2, 5, 1.8, "REWARD STRUCTURE", "#2A3050", fontsize=10)
    rewards_txt = (
        "+5–10  Job Secured  |  +0.05–0.2  Study High-Demand Skill\n"
        "−0.5–1.5  Rejected  |  −2.0  Burnout  |  −0.05  Resting"
    )
    ax.text(8, 1.75, rewards_txt, ha="center", va="center",
            fontsize=8, color="#AAAAAA", style="italic")

    # Terminal conditions
    box(0.5, 1.2, 4.5, 1.8, "TERMINAL CONDITIONS", "#3A1C1C", fontsize=9)
    ax.text(2.75, 1.9, "✅ Employed (success)\n⚠️ Energy = 0 (burnout)\n⏱  max_steps reached",
            ha="center", va="center", fontsize=8, color="#FFAAAA")

    box(11, 1.2, 4.5, 1.8, "OBS SPACE", "#1C3A2A", fontsize=9)
    ax.text(13.25, 1.9, "Box(0,1, shape=(13,))\n5 skills + 5 demands\n+ time + energy + hired",
            ha="center", va="center", fontsize=8, color="#AAFFAA")

    # Title
    ax.text(8, 9.7, "Kigali Job Market RL — Environment Architecture",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")
    ax.text(8, 0.5, "Jean Jacques JABO  |  BSE Pre-Capstone  |  ALU 2026  |  Supervisor: GATERA Thadde",
            ha="center", va="center", fontsize=9, color="#888888")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Environment diagram saved → {output_path}")
    plt.close()


# =============================================================================
# 4. Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Kigali Job Market RL — Static Random Agent Demo")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print("=" * 65)

    print("\n[1/3] Running random agent episode...")
    traj = run_random_episode(seed=42)

    n_steps = len(traj["steps"])
    total_r  = traj["cumulative_rewards"][-1] if traj["cumulative_rewards"] else 0
    print(f"  → Episode length : {n_steps} steps")
    print(f"  → Total reward   : {total_r:.3f}")
    print(f"  → Employed       : {traj['employed']}")

    action_freq = {}
    for a in traj["actions"]:
        cat = action_category(a)
        action_freq[cat] = action_freq.get(cat, 0) + 1
    print("\n  Action frequency breakdown:")
    for cat, cnt in sorted(action_freq.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(n_steps, 1)
        print(f"    {cat:10s}: {cnt:4d}  ({pct:.1f}%)")

    print("\n[2/3] Rendering static visualization...")
    render_static_demo(traj, output_path="plots/static_random_demo.png")

    print("\n[3/3] Rendering environment diagram...")
    render_env_diagram(output_path="plots/environment_diagram.png")

    print("\n✅ All done. Check the plots/ directory.")
    print("   • plots/static_random_demo.png  — random agent episode")
    print("   • plots/environment_diagram.png — environment architecture")
