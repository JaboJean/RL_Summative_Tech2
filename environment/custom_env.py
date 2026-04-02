"""
Kigali Job Market Forecasting Environment
==========================================
Custom Gymnasium environment based on the BSE Pre-Capstone project:
"Job Market Forecasting" by Jean Jacques JABO, African Leadership University

The agent represents a job seeker in Kigali, Rwanda, navigating the labor market
by acquiring skills and applying for jobs in high-demand sectors aligned with
Rwanda's Vision 2050 and NST1 priorities.

Sectors: Technology, Business, Agriculture, Construction, Healthcare
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class KigaliJobMarketEnv(gym.Env):
    """
    A custom Gymnasium environment simulating job market navigation in Kigali, Rwanda.

    OBSERVATION SPACE (Box, 13 features, all normalized [0,1]):
    ─────────────────────────────────────────────────────────────
    [0-4]   skill_tech, skill_business, skill_agriculture,
            skill_construction, skill_healthcare
    [5-9]   demand_tech, demand_business, demand_agriculture,
            demand_construction, demand_healthcare
    [10]    time_elapsed (step / max_steps)
    [11]    energy (0=burnt out, 1=fully energized)
    [12]    employed (0=seeking, 1=employed — terminal success flag)

    ACTION SPACE (Discrete, 12 actions):
    ──────────────────────────────────────
    0  → Study Tech Skills          (costs energy, increases tech skill)
    1  → Study Business Skills      (costs energy, increases business skill)
    2  → Study Agriculture Skills   (costs energy, increases agriculture skill)
    3  → Study Construction Skills  (costs energy, increases construction skill)
    4  → Study Healthcare Skills    (costs energy, increases healthcare skill)
    5  → Apply for Tech Job         (success if skill ≥ demand threshold)
    6  → Apply for Business Job
    7  → Apply for Agriculture Job
    8  → Apply for Construction Job
    9  → Apply for Healthcare Job
    10 → Network / Attend Job Fair  (small universal skill boost + market insight)
    11 → Rest / Recover Energy      (replenishes energy, small time penalty)

    REWARD STRUCTURE:
    ──────────────────
    - Studying high-demand skill:     +0.05 to +0.20 (demand-weighted)
    - Studying low-demand skill:      -0.02 (opportunity cost)
    - Consistent study streaks:       +0.05 bonus
    - Successful job application:     +5.0 to +10.0 (skill-demand match quality)
    - Failed job application:         -0.5 to -1.5 (skill gap penalty)
    - Networking (well-aligned):      +0.1 to +0.3
    - Resting:                        -0.05 (inactivity penalty)
    - Energy depletion (burnout):     -2.0 + terminal

    TERMINATION CONDITIONS:
    ────────────────────────
    - SUCCESS: Agent secures employment (employed=True)
    - FAILURE: Energy depleted to 0 (burnout)
    - TRUNCATION: max_steps reached without employment
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Sector definitions aligned with Rwanda's labor market data (NISR 2023)
    SECTORS = ["Tech", "Business", "Agriculture", "Construction", "Healthcare"]
    N_SECTORS = 5

    # Base demand levels based on Rwanda Development Board (2023) employment reports
    # Tech highest due to NST1 digital economy focus; Agriculture moderate due to informality
    BASE_DEMAND = np.array([0.85, 0.72, 0.55, 0.50, 0.68], dtype=np.float32)
    DEMAND_VOLATILITY = np.array([0.12, 0.10, 0.08, 0.08, 0.09], dtype=np.float32)

    # Hiring threshold: skill must reach this fraction of demand to be hired
    HIRE_THRESHOLD_RATIO = 0.75

    # Energy costs per action type
    ENERGY_COST_STUDY = 0.04
    ENERGY_COST_APPLY = 0.06
    ENERGY_COST_NETWORK = 0.03
    ENERGY_GAIN_REST = 0.18

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 200):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # === Action Space ===
        self.action_space = spaces.Discrete(12)

        # === Observation Space ===
        # 5 skills + 5 demands + time + energy + employed = 13 features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # Pygame rendering handles
        self.window = None
        self.clock = None
        self._renderer = None

        # Internal state
        self.skill_levels: np.ndarray = np.zeros(self.N_SECTORS, dtype=np.float32)
        self.market_demand: np.ndarray = self.BASE_DEMAND.copy()
        self.step_count: int = 0
        self.energy: float = 1.0
        self.employed: bool = False
        self.total_reward: float = 0.0
        self.consecutive_studies: int = 0
        self.job_applications: np.ndarray = np.zeros(self.N_SECTORS)
        self.last_action: int = -1
        self.last_action_result: str = ""
        self.action_history: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Observation & Info helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array([
            *self.skill_levels,                          # indices 0-4
            *self.market_demand,                         # indices 5-9
            self.step_count / self.max_steps,            # index 10
            float(self.energy),                          # index 11
            float(self.employed),                        # index 12
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        alignment = self.skill_levels * self.market_demand
        best_sector = int(np.argmax(alignment))
        return {
            "step": self.step_count,
            "skill_levels": self.skill_levels.copy(),
            "market_demand": self.market_demand.copy(),
            "energy": self.energy,
            "employed": self.employed,
            "total_reward": self.total_reward,
            "best_sector": self.SECTORS[best_sector],
            "skill_demand_alignment": alignment.copy(),
            "hire_probabilities": self._compute_hire_probabilities(),
            "last_action_result": self.last_action_result,
        }

    def _compute_hire_probabilities(self) -> np.ndarray:
        """Estimate probability of being hired in each sector given current skills."""
        threshold = self.market_demand * self.HIRE_THRESHOLD_RATIO
        probs = np.clip(
            (self.skill_levels - threshold) / (threshold + 1e-8) + 0.5, 0.0, 1.0
        )
        return probs.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Environment core methods
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Agent starts with very low skills (recent graduate baseline)
        self.skill_levels = self.np_random.uniform(
            0.02, 0.18, size=self.N_SECTORS
        ).astype(np.float32)

        # Market demand initialized with slight random variation around base
        noise = self.np_random.normal(0, 0.04, self.N_SECTORS)
        self.market_demand = np.clip(
            self.BASE_DEMAND + noise, 0.25, 1.0
        ).astype(np.float32)

        self.step_count = 0
        self.energy = 1.0
        self.employed = False
        self.total_reward = 0.0
        self.consecutive_studies = 0
        self.job_applications = np.zeros(self.N_SECTORS)
        self.last_action = -1
        self.last_action_result = "Episode started"
        self.action_history = []

        return self._get_obs(), self._get_info()

    def _update_market_demand(self):
        """Simulate dynamic market fluctuations (random walk with mean reversion)."""
        noise = self.np_random.normal(0.0, self.DEMAND_VOLATILITY * 0.08)
        # Mean reversion: market reverts towards base demand over time
        reversion = 0.02 * (self.BASE_DEMAND - self.market_demand)
        self.market_demand = np.clip(
            self.market_demand + noise + reversion, 0.2, 1.0
        ).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = 0.0
        terminated = False
        truncated = False
        self.last_action = action
        self._update_market_demand()

        # ── STUDY ACTIONS (0–4) ──────────────────────────────────────────────
        if action < 5:
            sector = action
            demand = self.market_demand[sector]

            # Diminishing returns on skill gain (harder to improve already-high skills)
            current_skill = self.skill_levels[sector]
            max_gain = 0.08 * (1.0 - 0.6 * current_skill)
            skill_gain = float(self.np_random.uniform(0.03, max(0.031, max_gain)))
            self.skill_levels[sector] = min(1.0, current_skill + skill_gain)

            self.energy = max(0.0, self.energy - self.ENERGY_COST_STUDY)
            self.consecutive_studies += 1

            # Reward: proportional to demand alignment
            alignment = self.skill_levels[sector] * demand
            base_reward = 0.08 * demand + 0.06 * alignment - 0.02
            streak_bonus = 0.05 if self.consecutive_studies >= 3 and demand > 0.65 else 0.0
            reward = base_reward + streak_bonus

            self.last_action_result = (
                f"Studied {self.SECTORS[sector]} | "
                f"Skill: {self.skill_levels[sector]:.2f} | "
                f"Demand: {demand:.2f} | "
                f"R: {reward:+.3f}"
            )

        # ── APPLY ACTIONS (5–9) ──────────────────────────────────────────────
        elif action < 10:
            sector = action - 5
            skill = self.skill_levels[sector]
            demand = self.market_demand[sector]
            hire_threshold = demand * self.HIRE_THRESHOLD_RATIO

            self.energy = max(0.0, self.energy - self.ENERGY_COST_APPLY)
            self.job_applications[sector] += 1
            self.consecutive_studies = 0

            if skill >= hire_threshold:
                # ✅ Job secured — episode ends successfully
                match_quality = skill / (demand + 1e-8)
                reward = 5.0 + 3.0 * match_quality + 2.0 * demand
                self.employed = True
                terminated = True
                self.last_action_result = (
                    f"✅ HIRED in {self.SECTORS[sector]}! | "
                    f"Skill: {skill:.2f} ≥ Threshold: {hire_threshold:.2f} | "
                    f"R: {reward:+.2f}"
                )
            else:
                # ❌ Rejected — penalty proportional to skill gap
                gap = hire_threshold - skill
                reward = -0.4 - gap * 1.2
                # Small experience boost from the interview
                self.skill_levels[sector] = min(
                    1.0, self.skill_levels[sector] + 0.01
                )
                self.last_action_result = (
                    f"❌ Rejected for {self.SECTORS[sector]} | "
                    f"Skill: {skill:.2f} < Threshold: {hire_threshold:.2f} | "
                    f"Gap: {gap:.2f} | R: {reward:+.3f}"
                )

        # ── NETWORK ACTION (10) ───────────────────────────────────────────────
        elif action == 10:
            self.energy = max(0.0, self.energy - self.ENERGY_COST_NETWORK)
            self.consecutive_studies = 0

            # Universal skill micro-boost (networking improves all skills slightly)
            boost = self.np_random.uniform(0.005, 0.02, self.N_SECTORS)
            self.skill_levels = np.clip(
                self.skill_levels + boost, 0.0, 1.0
            ).astype(np.float32)

            alignment_score = float(np.mean(self.skill_levels * self.market_demand))
            reward = 0.08 + 0.25 * alignment_score

            self.last_action_result = (
                f"🤝 Networked | Alignment: {alignment_score:.2f} | R: {reward:+.3f}"
            )

        # ── REST ACTION (11) ─────────────────────────────────────────────────
        elif action == 11:
            gain = float(self.np_random.uniform(
                self.ENERGY_GAIN_REST * 0.7, self.ENERGY_GAIN_REST * 1.3
            ))
            self.energy = min(1.0, self.energy + gain)
            self.consecutive_studies = 0
            reward = -0.05  # Opportunity cost of inactivity

            self.last_action_result = (
                f"😴 Rested | Energy: {self.energy:.2f} | R: {reward:+.3f}"
            )

        # ── BURNOUT CHECK ────────────────────────────────────────────────────
        if self.energy <= 0.0 and not terminated:
            reward -= 2.0
            terminated = True
            self.last_action_result += " | ⚠️ BURNOUT — episode failed"

        self.step_count += 1
        self.total_reward += reward
        self.action_history.append((action, reward))

        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        try:
            from environment.rendering import KigaliJobMarketRenderer
        except ImportError:
            from rendering import KigaliJobMarketRenderer

        if self._renderer is None:
            self._renderer = KigaliJobMarketRenderer(self)

        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Utility: human-readable action descriptions
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def action_description(action: int) -> str:
        descriptions = [
            "Study Tech Skills",
            "Study Business Skills",
            "Study Agriculture Skills",
            "Study Construction Skills",
            "Study Healthcare Skills",
            "Apply for Tech Job",
            "Apply for Business Job",
            "Apply for Agriculture Job",
            "Apply for Construction Job",
            "Apply for Healthcare Job",
            "Network / Attend Job Fair",
            "Rest / Recover Energy",
        ]
        return descriptions[action] if 0 <= action < len(descriptions) else "Unknown"

    def get_state_summary(self) -> str:
        lines = [
            f"Step: {self.step_count}/{self.max_steps}",
            f"Energy: {self.energy:.2f}",
            f"Employed: {self.employed}",
            f"Total Reward: {self.total_reward:.3f}",
            "",
            "Skills vs Demand:",
        ]
        for i, s in enumerate(self.SECTORS):
            skill = self.skill_levels[i]
            demand = self.market_demand[i]
            threshold = demand * self.HIRE_THRESHOLD_RATIO
            status = "✅ HIREABLE" if skill >= threshold else f"❌ gap:{threshold-skill:.2f}"
            lines.append(f"  {s:15s}: skill={skill:.2f} | demand={demand:.2f} | {status}")
        return "\n".join(lines)
