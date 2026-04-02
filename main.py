"""
main.py — Run Best Performing RL Agent
=========================================
Entry point for the Kigali Job Market RL project.

Loads the best-performing model (selected after hyperparameter tuning)
and runs it in the KigaliJobMarketEnv with full pygame visualization.

Usage:
    python main.py                     # Auto-select best model, with GUI
    python main.py --algo PPO          # Force specific algorithm
    python main.py --algo DQN --no-gui # Headless evaluation
    python main.py --episodes 5        # Run N episodes

Author : Jean Jacques JABO
Program: BSE Pre-Capstone, African Leadership University (ALU)
Year   : 2026
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import KigaliJobMarketEnv

RESULTS_DIR = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Load helpers
# =============================================================================

def load_sb3_model(algo: str, model_path: str):
    """Load a Stable-Baselines3 model."""
    from stable_baselines3 import DQN, PPO
    loaders = {"DQN": DQN, "PPO": PPO}
    if algo not in loaders:
        raise ValueError(f"Unknown SB3 algorithm: {algo}")
    return loaders[algo].load(model_path)


def load_reinforce_model(info: dict):
    """Load a saved REINFORCE agent."""
    from training.reinforce_algorithm import REINFORCEAgent
    params = info["params"]
    agent = REINFORCEAgent(
        lr=params["lr"],
        gamma=params["gamma"],
        hidden_dim=params["hidden_dim"],
        use_baseline=params.get("use_baseline", True),
        entropy_coef=params.get("entropy_coef", 0.01),
    )
    agent.load(info["model_path"])
    agent.policy.eval()
    return agent


def find_best_model():
    """Identify the best algorithm based on saved evaluation results."""
    final_path = os.path.join(RESULTS_DIR, "final_comparison_table.csv")
    if os.path.exists(final_path):
        import pandas as pd
        df = pd.read_csv(final_path)
        best_algo = df.loc[df["Mean Reward"].idxmax(), "Algorithm"]
        print(f"[Auto-select] Best algorithm from evaluation: {best_algo}")
        return best_algo

    # Fallback: compare best JSON files
    best_reward = -999.0
    best_algo   = None
    for algo in ["DQN", "PPO", "REINFORCE"]:
        info_path = os.path.join(RESULTS_DIR, f"{algo.lower()}_best.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            if info["mean_reward"] > best_reward:
                best_reward = info["mean_reward"]
                best_algo   = algo

    if best_algo:
        print(f"[Auto-select] Best algorithm: {best_algo} (reward={best_reward:.3f})")
    return best_algo


def load_model(algo: str):
    """Load model and info for a given algorithm."""
    info_path = os.path.join(RESULTS_DIR, f"{algo.lower()}_best.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"No trained model found for {algo}. "
            f"Run training/{'dqn_training' if algo=='DQN' else 'pg_training' if algo=='PPO' else 'reinforce_algorithm'}.py first."
        )
    with open(info_path) as f:
        info = json.load(f)

    if algo == "REINFORCE":
        model = load_reinforce_model(info)
        model_type = "reinforce"
    else:
        model = load_sb3_model(algo, info["model_path"])
        model_type = "sb3"

    return model, model_type, info


# =============================================================================
# Episode runner
# =============================================================================

def run_episode(model, model_type: str, env: KigaliJobMarketEnv,
                seed: int = 0, render: bool = True, verbose: bool = True):
    """Run one episode with the given model."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Episode seed={seed} | Render={'ON' if render else 'OFF'}")
        print(f"{'─'*50}")
        print(f"  Initial skills : {[f'{x:.2f}' for x in info['skill_levels']]}")
        print(f"  Market demand  : {[f'{x:.2f}' for x in info['market_demand']]}")

    while True:
        # Action selection
        if model_type == "sb3":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif model_type == "reinforce":
            with torch.no_grad():
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                probs   = model.policy(state_t)
                action  = probs.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose and steps % 25 == 0:
            skills = info["skill_levels"]
            demands = info["market_demand"]
            print(f"  Step {steps:3d} | Reward: {total_reward:+.2f} | "
                  f"Energy: {info['energy']:.2f} | "
                  f"Best skill: {env.SECTORS[int(np.argmax(skills))]}")

        if terminated or truncated:
            break

    if verbose:
        result_str = "✅ EMPLOYED" if info["employed"] else "❌ Not employed"
        print(f"\n  {result_str}")
        print(f"  Total reward    : {total_reward:+.3f}")
        print(f"  Steps taken     : {steps}")
        print(f"  Final skills    : {[f'{x:.2f}' for x in info['skill_levels']]}")
        print(f"  Final energy    : {info['energy']:.2f}")
        print(env.get_state_summary())

    return {
        "total_reward": total_reward,
        "steps":        steps,
        "employed":     bool(info["employed"]),
        "final_info":   info,
    }


# =============================================================================
# API serialization (bonus: JSON output for frontend integration)
# =============================================================================

def episode_result_to_json(result: dict, algo: str) -> dict:
    """Serialize episode result to JSON-safe dict for API/frontend use.
    All numpy scalar types are explicitly cast to native Python float/bool/int.
    """
    info = result["final_info"]

    def _f(x):
        """Convert any numpy scalar to a native Python float, rounded to 3dp."""
        return round(float(x), 3)

    return {
        "algorithm":    algo,
        "total_reward": _f(result["total_reward"]),
        "steps":        int(result["steps"]),
        "employed":     bool(result["employed"]),
        "final_state": {
            "skills": {
                sector: _f(skill)
                for sector, skill in zip(
                    KigaliJobMarketEnv.SECTORS, info["skill_levels"]
                )
            },
            "market_demand": {
                sector: _f(demand)
                for sector, demand in zip(
                    KigaliJobMarketEnv.SECTORS, info["market_demand"]
                )
            },
            "energy": _f(info["energy"]),
        },
        "hire_probabilities": {
            sector: _f(prob)
            for sector, prob in zip(
                KigaliJobMarketEnv.SECTORS, info["hire_probabilities"]
            )
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kigali Job Market RL — Run Best Agent"
    )
    parser.add_argument("--algo", type=str, default=None,
                        choices=["DQN", "PPO", "REINFORCE"],
                        help="Algorithm to run (default: auto-select best)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable pygame visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for first episode")
    args = parser.parse_args()

    print("=" * 60)
    print("  Kigali Job Market RL — Best Agent Runner")
    print("  Jean Jacques JABO | BSE Pre-Capstone | ALU 2026")
    print("=" * 60)

    # Select algorithm
    algo = args.algo
    if algo is None:
        algo = find_best_model()
        if algo is None:
            print("\n⚠️  No trained models found. Please run training first.")
            print("    python training/dqn_training.py")
            print("    python training/pg_training.py")
            print("    python training/reinforce_algorithm.py")
            sys.exit(1)

    print(f"\nAlgorithm : {algo}")
    print(f"Episodes  : {args.episodes}")
    print(f"GUI       : {'OFF' if args.no_gui else 'ON (pygame)'}")

    # Load model
    print(f"\nLoading {algo} model...")
    try:
        model, model_type, info = load_model(algo)
        print(f"✅ Loaded: Run {info['run']} | "
              f"Mean reward: {info['mean_reward']:.3f}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Create environment
    render_mode = None if args.no_gui else "human"
    env = KigaliJobMarketEnv(render_mode=render_mode, max_steps=200)

    # Run episodes
    all_results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n{'='*60}")
        print(f"  Episode {ep+1}/{args.episodes}")
        result = run_episode(model, model_type, env, seed=seed,
                             render=(not args.no_gui), verbose=True)
        all_results.append(result)

        # Print JSON output (for API integration demo)
        api_output = episode_result_to_json(result, algo)
        print(f"\n  📡 API JSON Output:")
        print(f"  {json.dumps(api_output, indent=4)}")

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("  MULTI-EPISODE SUMMARY")
    print("=" * 60)
    rewards   = [r["total_reward"] for r in all_results]
    employed  = [r["employed"] for r in all_results]
    print(f"  Algorithm        : {algo}")
    print(f"  Episodes         : {args.episodes}")
    print(f"  Mean Reward      : {np.mean(rewards):+.3f} ± {np.std(rewards):.3f}")
    print(f"  Employment Rate  : {np.mean(employed)*100:.1f}%")
    print(f"  Best Episode     : {max(rewards):+.3f}")
    print(f"  Worst Episode    : {min(rewards):+.3f}")


if __name__ == "__main__":
    main()
