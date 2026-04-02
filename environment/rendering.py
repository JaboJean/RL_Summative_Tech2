"""
Kigali Job Market — Pygame Renderer
=====================================
Advanced 2D visualization for the KigaliJobMarketEnv using Pygame.

Layout (1200 × 700 px):
┌──────────────────────────────────────────────────────────────────────────┐
│  HEADER: Title bar with project info and step counter                    │
├──────────────┬────────────────────────────────────────┬──────────────────┤
│ AGENT STATUS │         KIGALI JOB MARKET              │ MARKET INTEL     │
│  • Energy    │  5 sector columns (skill vs demand)    │  • Alignment     │
│  • Steps     │  • Animated skill bars                 │  • Hire probs    │
│  • Reward    │  • Demand target lines                 │  • Best sector   │
│  • Status    │  • Color-coded readiness               │  • Action log    │
├──────────────┴────────────────────────────────────────┴──────────────────┤
│  FOOTER: Last action taken + result                                      │
└──────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.custom_env import KigaliJobMarketEnv

# ── Color Palette (Rwanda-inspired: green, gold, dark) ────────────────────
BG_DARK       = (18,  22,  36)
BG_PANEL      = (28,  34,  52)
BG_CARD       = (38,  46,  70)
ACCENT_GREEN  = (0,  200, 120)
ACCENT_GOLD   = (255, 200,  50)
ACCENT_RED    = (220,  60,  60)
ACCENT_BLUE   = (70,  140, 255)
ACCENT_PURPLE = (160,  90, 255)
TEXT_WHITE    = (240, 240, 250)
TEXT_GREY     = (150, 160, 180)
TEXT_DIM      = (90,  100, 120)
BORDER_COLOR  = (50,   65,  95)

# Sector color palette
SECTOR_COLORS = [
    (70,  140, 255),   # Tech     — Blue
    (255, 180,  50),   # Business — Gold
    (70,  200, 100),   # Agri     — Green
    (200, 100,  60),   # Constr.  — Orange
    (180,  90, 220),   # Health   — Purple
]


class KigaliJobMarketRenderer:
    """Pygame-based renderer for the Kigali Job Market environment."""

    WIDTH  = 1200
    HEIGHT = 700
    FPS    = 10

    HEADER_H = 70
    FOOTER_H = 60
    PANEL_W  = 240
    CENTER_W = WIDTH - 2 * PANEL_W

    def __init__(self, env: "KigaliJobMarketEnv"):
        import pygame
        pygame.init()
        pygame.display.set_caption(
            "Kigali Job Market RL — Jean Jacques JABO | ALU BSE 2026"
        )
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock  = pygame.time.Clock()
        self.pygame = pygame

        # Fonts
        self.font_title  = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_body   = pygame.font.SysFont("consolas", 13)
        self.font_small  = pygame.font.SysFont("consolas", 11)
        self.font_large  = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_icon   = pygame.font.SysFont("consolas", 16)

        self.frame_count = 0
        self.particle_system = []

    # ─────────────────────────────────────────────────────────────────────────

    def render(self, env: "KigaliJobMarketEnv"):
        """Main render call — draws full frame and returns surface."""
        pg = self.pygame
        screen = self.screen

        # Handle quit events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return None

        screen.fill(BG_DARK)
        self._draw_grid(screen)
        self._draw_header(screen, env)
        self._draw_left_panel(screen, env)
        self._draw_center_panel(screen, env)
        self._draw_right_panel(screen, env)
        self._draw_footer(screen, env)
        self._update_particles(screen)

        pg.display.flip()
        self.clock.tick(self.FPS)
        self.frame_count += 1

        # Return rgb_array
        return np.transpose(
            np.array(pg.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def _draw_grid(self, screen):
        """Subtle background grid."""
        pg = self.pygame
        for x in range(0, self.WIDTH, 40):
            pg.draw.line(screen, (25, 30, 48), (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pg.draw.line(screen, (25, 30, 48), (0, y), (self.WIDTH, y), 1)

    # ── HEADER ────────────────────────────────────────────────────────────────

    def _draw_header(self, screen, env):
        pg = self.pygame
        # Background bar
        pg.draw.rect(screen, BG_PANEL, (0, 0, self.WIDTH, self.HEADER_H))
        pg.draw.line(screen, BORDER_COLOR, (0, self.HEADER_H), (self.WIDTH, self.HEADER_H), 2)

        # Rwanda flag stripe (decorative)
        for i, color in enumerate([
            (0, 135, 81), (255, 210, 0), (0, 168, 198)
        ]):
            pg.draw.rect(screen, color, (0, i * 4, self.WIDTH, 4))

        # Title
        title = self.font_large.render(
            "KIGALI JOB MARKET RL — Jean Jacques JABO | ALU BSE 2026",
            True, TEXT_WHITE
        )
        screen.blit(title, (20, 22))

        # Step info (right-aligned)
        step_pct = env.step_count / env.max_steps
        step_text = self.font_title.render(
            f"Step {env.step_count}/{env.max_steps}  ({step_pct*100:.0f}%)",
            True, ACCENT_GOLD
        )
        screen.blit(step_text, (self.WIDTH - step_text.get_width() - 20, 26))

    # ── LEFT PANEL — Agent Status ─────────────────────────────────────────────

    def _draw_left_panel(self, screen, env):
        pg = self.pygame
        x, y = 0, self.HEADER_H + 2
        w, h = self.PANEL_W, self.HEIGHT - self.HEADER_H - self.FOOTER_H - 4

        pg.draw.rect(screen, BG_PANEL, (x, y, w, h))
        pg.draw.rect(screen, BORDER_COLOR, (x, y, w, h), 1)

        cy = y + 15
        self._label(screen, "◈ AGENT STATUS", x + 12, cy, ACCENT_GREEN, self.font_title)
        cy += 30

        # Energy bar
        self._label(screen, "Energy", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        self._bar(screen, x + 12, cy, w - 24, 16, env.energy,
                  _lerp_color(ACCENT_RED, ACCENT_GREEN, env.energy))
        self._label(screen, f"{env.energy*100:.0f}%", x + w - 42, cy - 1, TEXT_WHITE, self.font_small)
        cy += 26

        # Total reward
        self._label(screen, "Total Reward", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        reward_color = ACCENT_GREEN if env.total_reward >= 0 else ACCENT_RED
        rwd_surf = self.font_title.render(f"{env.total_reward:+.2f}", True, reward_color)
        screen.blit(rwd_surf, (x + 12, cy))
        cy += 28

        # Employment status
        self._label(screen, "Status", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        if env.employed:
            status_text, sc = "✅  EMPLOYED", ACCENT_GREEN
        elif env.energy < 0.2:
            status_text, sc = "⚠️  LOW ENERGY", ACCENT_RED
        else:
            status_text, sc = "🔍  SEEKING", ACCENT_GOLD
        self._label(screen, status_text, x + 12, cy, sc, self.font_body)
        cy += 30

        # Applications made
        self._label(screen, "Applications", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        for i, sector in enumerate(env.SECTORS):
            apps = int(env.job_applications[i])
            if apps > 0:
                color = SECTOR_COLORS[i]
                self._label(
                    screen, f"  {sector[:5]:5s}: {apps}",
                    x + 12, cy, color, self.font_small
                )
                cy += 14

        cy = max(cy, y + 240)
        # Reward history sparkline
        self._label(screen, "Reward History", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        self._draw_sparkline(screen, env, x + 12, cy, w - 24, 60)

    # ── CENTER PANEL — Sector Visualization ──────────────────────────────────

    def _draw_center_panel(self, screen, env):
        pg = self.pygame
        x0 = self.PANEL_W + 2
        y0 = self.HEADER_H + 2
        w  = self.CENTER_W - 4
        h  = self.HEIGHT - self.HEADER_H - self.FOOTER_H - 4

        pg.draw.rect(screen, BG_PANEL, (x0, y0, w, h))
        pg.draw.rect(screen, BORDER_COLOR, (x0, y0, w, h), 1)

        self._label(screen, "◈ KIGALI LABOR MARKET — SECTOR ANALYSIS",
                    x0 + 12, y0 + 12, ACCENT_GOLD, self.font_title)

        # Legend
        lx = x0 + w - 220
        pg.draw.rect(screen, ACCENT_BLUE, (lx, y0 + 12, 12, 12))
        self._label(screen, "Skill Level", lx + 16, y0 + 12, TEXT_GREY, self.font_small)
        pg.draw.rect(screen, ACCENT_RED, (lx + 100, y0 + 12, 12, 12))
        self._label(screen, "Demand", lx + 116, y0 + 12, TEXT_GREY, self.font_small)

        # Draw 5 sector columns
        col_margin = 12
        col_w = (w - col_margin * (env.N_SECTORS + 1)) // env.N_SECTORS
        bar_area_top = y0 + 50
        bar_area_h   = h - 90

        for i, sector in enumerate(env.SECTORS):
            col_x = x0 + col_margin + i * (col_w + col_margin)

            # Card background
            pg.draw.rect(screen, BG_CARD, (col_x, bar_area_top, col_w, bar_area_h), border_radius=6)
            pg.draw.rect(screen, SECTOR_COLORS[i], (col_x, bar_area_top, col_w, bar_area_h), 1, border_radius=6)

            skill  = float(env.skill_levels[i])
            demand = float(env.market_demand[i])
            threshold = demand * env.HIRE_THRESHOLD_RATIO

            inner_top  = bar_area_top + 10
            inner_h    = bar_area_h - 55
            inner_x    = col_x + 8
            inner_w    = col_w - 16

            # Background bar
            pg.draw.rect(screen, BG_DARK, (inner_x, inner_top, inner_w, inner_h), border_radius=4)

            # Skill fill
            skill_fill_h = int(inner_h * skill)
            skill_y = inner_top + inner_h - skill_fill_h
            skill_color = (
                ACCENT_GREEN if skill >= threshold
                else ACCENT_GOLD if skill >= threshold * 0.8
                else SECTOR_COLORS[i]
            )
            if skill_fill_h > 0:
                pg.draw.rect(
                    screen, skill_color,
                    (inner_x, skill_y, inner_w, skill_fill_h),
                    border_radius=3
                )

            # Animated shimmer on skill bar
            shimmer_pos = (self.frame_count * 3) % (inner_h + 30) - 10
            shimmer_y = skill_y + shimmer_pos
            if inner_top <= shimmer_y <= inner_top + inner_h and skill > 0.05:
                shimmer_surf = pg.Surface((inner_w, 3), pg.SRCALPHA)
                shimmer_surf.fill((255, 255, 255, 40))
                screen.blit(shimmer_surf, (inner_x, shimmer_y))

            # Demand line (horizontal, red)
            demand_y = inner_top + int(inner_h * (1.0 - demand))
            pg.draw.line(screen, ACCENT_RED,
                         (inner_x - 2, demand_y), (inner_x + inner_w + 2, demand_y), 2)

            # Threshold dashed line (orange)
            thresh_y = inner_top + int(inner_h * (1.0 - threshold))
            for dash_x in range(inner_x, inner_x + inner_w, 8):
                pg.draw.line(screen, ACCENT_GOLD,
                             (dash_x, thresh_y), (min(dash_x + 4, inner_x + inner_w), thresh_y), 1)

            # Value labels
            skill_lbl  = self.font_small.render(f"{skill*100:.0f}%", True, TEXT_WHITE)
            demand_lbl = self.font_small.render(f"D:{demand*100:.0f}%", True, ACCENT_RED)
            screen.blit(skill_lbl,  (inner_x, inner_top + inner_h + 3))
            screen.blit(demand_lbl, (inner_x, inner_top + inner_h + 16))

            # Sector name
            name_surf = self.font_body.render(sector, True, SECTOR_COLORS[i])
            screen.blit(
                name_surf,
                (col_x + (col_w - name_surf.get_width()) // 2, bar_area_top + bar_area_h - 20)
            )

            # Ready indicator
            if skill >= threshold:
                ready = self.font_small.render("READY!", True, ACCENT_GREEN)
                screen.blit(
                    ready,
                    (col_x + (col_w - ready.get_width()) // 2, bar_area_top + 2)
                )

    # ── RIGHT PANEL — Market Intelligence ────────────────────────────────────

    def _draw_right_panel(self, screen, env):
        pg = self.pygame
        x = self.PANEL_W + self.CENTER_W + 2
        y = self.HEADER_H + 2
        w = self.PANEL_W - 2
        h = self.HEIGHT - self.HEADER_H - self.FOOTER_H - 4

        pg.draw.rect(screen, BG_PANEL, (x, y, w, h))
        pg.draw.rect(screen, BORDER_COLOR, (x, y, w, h), 1)

        cy = y + 15
        self._label(screen, "◈ MARKET INTEL", x + 12, cy, ACCENT_PURPLE, self.font_title)
        cy += 28

        # Hire probabilities
        self._label(screen, "Hire Probability", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        probs = env._compute_hire_probabilities()
        for i, sector in enumerate(env.SECTORS):
            self._label(screen, f"{sector[:6]:6s}", x + 12, cy, SECTOR_COLORS[i], self.font_small)
            bar_x = x + 70
            bar_w = w - 82
            pg.draw.rect(screen, BG_DARK, (bar_x, cy + 1, bar_w, 11), border_radius=3)
            fill_w = int(bar_w * probs[i])
            if fill_w > 0:
                color = _lerp_color(ACCENT_RED, ACCENT_GREEN, probs[i])
                pg.draw.rect(screen, color, (bar_x, cy + 1, fill_w, 11), border_radius=3)
            prob_lbl = self.font_small.render(f"{probs[i]*100:.0f}%", True, TEXT_WHITE)
            screen.blit(prob_lbl, (bar_x + bar_w + 2, cy))
            cy += 16

        cy += 6
        # Best sector to study
        alignment = env.skill_levels * env.market_demand
        best_study  = int(np.argmax(env.market_demand - env.skill_levels))
        best_apply  = int(np.argmax(probs))

        self._label(screen, "Best to Study", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 16
        self._label(
            screen, f"  ▶ {env.SECTORS[best_study]}",
            x + 12, cy, SECTOR_COLORS[best_study], self.font_body
        )
        cy += 22

        self._label(screen, "Best to Apply", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 16
        self._label(
            screen, f"  ▶ {env.SECTORS[best_apply]}",
            x + 12, cy, SECTOR_COLORS[best_apply], self.font_body
        )
        cy += 28

        # Action log (last 10 actions)
        self._label(screen, "Action Log", x + 12, cy, TEXT_GREY, self.font_body)
        cy += 18
        history = env.action_history[-8:] if len(env.action_history) > 0 else []
        for j, (act, rew) in enumerate(reversed(history)):
            alpha = 255 - j * 25
            act_name = env.action_description(act)[:16]
            color_val = max(80, 240 - j * 20)
            color = (color_val, color_val, color_val)
            rew_color = ACCENT_GREEN if rew >= 0 else ACCENT_RED
            self._label(screen, f"• {act_name}", x + 12, cy, color, self.font_small)
            rew_surf = self.font_small.render(f"{rew:+.2f}", True, rew_color)
            screen.blit(rew_surf, (x + w - rew_surf.get_width() - 6, cy))
            cy += 14

    # ── FOOTER ───────────────────────────────────────────────────────────────

    def _draw_footer(self, screen, env):
        pg = self.pygame
        fy = self.HEIGHT - self.FOOTER_H
        pg.draw.rect(screen, BG_PANEL, (0, fy, self.WIDTH, self.FOOTER_H))
        pg.draw.line(screen, BORDER_COLOR, (0, fy), (self.WIDTH, fy), 2)

        action_name = (
            env.action_description(env.last_action)
            if env.last_action >= 0 else "Waiting..."
        )
        label = self.font_body.render(f"Last Action: {action_name}", True, ACCENT_BLUE)
        screen.blit(label, (20, fy + 10))

        result = env.last_action_result[:90]
        result_surf = self.font_small.render(result, True, TEXT_GREY)
        screen.blit(result_surf, (20, fy + 30))

        # Right: reward rate
        rate = env.total_reward / max(1, env.step_count)
        rate_surf = self.font_body.render(
            f"Reward/step: {rate:+.3f}", True,
            ACCENT_GREEN if rate >= 0 else ACCENT_RED
        )
        screen.blit(rate_surf, (self.WIDTH - rate_surf.get_width() - 20, fy + 20))

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _bar(self, screen, x, y, w, h, value, color):
        pg = self.pygame
        pg.draw.rect(screen, BG_DARK, (x, y, w, h), border_radius=4)
        fill = int(w * max(0.0, min(1.0, value)))
        if fill > 0:
            pg.draw.rect(screen, color, (x, y, fill, h), border_radius=4)

    def _label(self, screen, text, x, y, color, font):
        surf = font.render(str(text), True, color)
        screen.blit(surf, (x, y))

    def _draw_sparkline(self, screen, env, x, y, w, h):
        pg = self.pygame
        if len(env.action_history) < 2:
            return
        rewards = [r for _, r in env.action_history[-w:]]
        if not rewards:
            return
        min_r, max_r = min(rewards), max(rewards)
        rng = max(max_r - min_r, 0.1)
        n = len(rewards)
        pg.draw.rect(screen, BG_DARK, (x, y, w, h), border_radius=4)
        # Zero line
        zero_y = y + int(h * max(0, -min_r) / rng)
        pg.draw.line(screen, TEXT_DIM, (x, zero_y), (x + w, zero_y), 1)
        pts = []
        for j, r in enumerate(rewards):
            px = x + int(j * w / max(n - 1, 1))
            py = y + h - int((r - min_r) / rng * h)
            pts.append((px, py))
        if len(pts) >= 2:
            pg.draw.lines(screen, ACCENT_GREEN, False, pts, 1)

    def _update_particles(self, screen):
        """Simple particle system for visual polish."""
        pg = self.pygame
        # Randomly emit
        if self.frame_count % 8 == 0 and len(self.particle_system) < 20:
            import random
            self.particle_system.append({
                "x": random.randint(self.PANEL_W, self.PANEL_W + self.CENTER_W),
                "y": self.HEIGHT - self.FOOTER_H - 5,
                "vy": -random.uniform(0.5, 2.0),
                "life": random.randint(30, 80),
                "color": random.choice([ACCENT_GREEN, ACCENT_GOLD, ACCENT_BLUE]),
            })
        alive = []
        for p in self.particle_system:
            p["y"] += p["vy"]
            p["life"] -= 1
            alpha = max(0, int(255 * p["life"] / 80))
            s = pg.Surface((3, 3), pg.SRCALPHA)
            s.fill((*p["color"], alpha))
            screen.blit(s, (int(p["x"]), int(p["y"])))
            if p["life"] > 0:
                alive.append(p)
        self.particle_system = alive

    def close(self):
        self.pygame.quit()


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
