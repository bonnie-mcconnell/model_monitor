#!/usr/bin/env python3
"""Generate the README hero plot: a four-panel monitoring dashboard screenshot.

Usage:
    python scripts/demo_plot.py              # saves to docs/demo_plot.png
    python scripts/demo_plot.py --show       # opens interactive window instead

What the plot shows
-------------------
The plot simulates 60 monitoring batches with a drift event at batch 35
and a recovery retrain at batch 45.  It illustrates the full monitoring stack:

  Panel 1 - Trust Score + Decision Events
    The composite health score in [0,1] with vertical bands showing the
    decision engine's actions: reject (red), retrain (orange), promote (green).
    This is the most important panel - it's what an operator watches.

  Panel 2 - Per-Feature PSI (input drift)
    Three features with different drift profiles.  Feature 2 drifts sharply
    at batch 35 and stabilises after batch 45 (retrain on new distribution).
    Threshold bands show the 0.10 (warning) and 0.25 (critical) PSI levels.

  Panel 3 - Output Drift + Conformal Set Size
    Left axis: output PSI mirrors input drift but lags slightly (the model
    adjusts its distribution before metrics degrade).  Right axis: conformal
    prediction set size grows during the drift event (model uncertainty
    increases before coverage formally breaks).

  Panel 4 - p95 / p99 Latency
    Tail latency is unaffected by the drift event, demonstrating that the
    monitoring overhead is negligible.  A healthy-looking mean would mask
    nothing here - both p95 and p99 are stable.

Design notes
------------
- All data is synthetic but follows realistic distributions
- Numpy seed is fixed for reproducibility
- The plot uses the project's colour palette
- Saved at 200 DPI to look sharp in GitHub's README renderer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate model_monitor README hero plot")
    p.add_argument("--output", default="docs/demo_plot.png", help="Output path")
    p.add_argument("--show", action="store_true", help="Open interactive window")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _simulate(n_batches: int = 60, drift_start: int = 35,
              recovery: int = 45, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Synthesise monitoring time-series with a realistic drift-retrain-recovery arc."""
    batches = np.arange(n_batches)

    # ── Trust score ────────────────────────────────────────────────────────
    trust = np.ones(n_batches) * 0.88
    trust += rng.normal(0, 0.02, n_batches)
    # Drift degrades trust linearly from batch 35 to 43
    for i in range(drift_start, min(recovery, n_batches)):
        trust[i] = 0.88 - (i - drift_start) * 0.06 + rng.normal(0, 0.01)
    # Recovery after retrain
    for i in range(recovery, n_batches):
        trust[i] = min(0.88 + (i - recovery) * 0.01 + rng.normal(0, 0.015), 0.95)
    trust = np.clip(trust, 0.0, 1.0)

    # ── Decision events ────────────────────────────────────────────────────
    # reject from batch 36-39, retrain at 40-43, promote at 47
    decisions: dict[int, str] = {}
    for i in range(36, 40):
        decisions[i] = "reject"
    for i in range(40, 44):
        decisions[i] = "retrain"
    decisions[47] = "promote"

    # ── Per-feature PSI ────────────────────────────────────────────────────
    psi_f0 = 0.02 + rng.exponential(0.008, n_batches)
    psi_f1 = 0.03 + rng.exponential(0.010, n_batches)
    psi_f2 = 0.02 + rng.exponential(0.008, n_batches)
    # Feature 2 drifts
    for i in range(drift_start, recovery):
        psi_f2[i] += 0.05 + (i - drift_start) * 0.012
    # Feature 0 drifts mildly
    for i in range(drift_start, recovery):
        psi_f0[i] += 0.02 + (i - drift_start) * 0.004
    # Post-recovery: all settle
    psi_f0[recovery:] = 0.02 + rng.exponential(0.006, n_batches - recovery)
    psi_f1[recovery:] = 0.03 + rng.exponential(0.008, n_batches - recovery)
    psi_f2[recovery:] = 0.02 + rng.exponential(0.006, n_batches - recovery)

    # ── Output drift PSI ───────────────────────────────────────────────────
    output_psi = 0.015 + rng.exponential(0.005, n_batches)
    for i in range(drift_start + 2, recovery):   # lags input drift by 2 batches
        output_psi[i] += 0.03 + (i - drift_start - 2) * 0.009
    output_psi[recovery:] = 0.015 + rng.exponential(0.004, n_batches - recovery)

    # ── Conformal set size ─────────────────────────────────────────────────
    set_size = 1.05 + rng.normal(0, 0.04, n_batches)
    for i in range(drift_start, recovery):
        set_size[i] += (i - drift_start) * 0.07
    set_size[recovery:] = 1.05 + rng.normal(0, 0.04, n_batches - recovery)
    set_size = np.clip(set_size, 1.0, None)

    # ── Tail latency ───────────────────────────────────────────────────────
    p95 = 85.0 + rng.exponential(12.0, n_batches)
    p99 = p95 + rng.exponential(18.0, n_batches)

    return {
        "batches": batches,
        "trust": trust,
        "decisions": decisions,
        "psi_f0": psi_f0,
        "psi_f1": psi_f1,
        "psi_f2": psi_f2,
        "output_psi": output_psi,
        "set_size": set_size,
        "p95": p95,
        "p99": p99,
    }


def _action_colour(action: str) -> str:
    return {"reject": "#e74c3c", "retrain": "#f39c12", "promote": "#2ecc71"}.get(
        action, "#95a5a6"
    )


def generate(output_path: Path, *, show: bool = False, seed: int = 42) -> None:
    """Render and save (or show) the monitoring dashboard plot."""
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(seed)
    data = _simulate(rng=rng)
    batches = data["batches"]

    DARK_BG = "#1a1a2e"
    PANEL_BG = "#16213e"
    ACCENT   = "#0f3460"
    TEXT     = "#e0e0e0"
    GRID     = "#2a2a4a"

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    fig.suptitle(
        "model_monitor - Live Monitoring Dashboard",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 1, hspace=0.55, left=0.07, right=0.96,
                           top=0.94, bottom=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        ax.set_xlim(batches[0] - 0.5, batches[-1] + 0.5)

    # ── Panel 1: Trust score + decision events ─────────────────────────────
    ax = axes[0]
    ax.set_title("Trust Score  ·  Decision Events", fontsize=10, pad=4)
    ax.set_ylabel("Trust Score", fontsize=8)
    ax.set_ylim(0.0, 1.05)

    # Decision event bands
    for batch_idx, action in data["decisions"].items():
        colour = _action_colour(action)
        ax.axvline(batch_idx, color=colour, alpha=0.35, linewidth=6, zorder=1)
        ax.axvline(batch_idx, color=colour, alpha=0.9, linewidth=1.2, zorder=2)

    ax.plot(batches, data["trust"], color="#3498db", linewidth=1.8, zorder=3, label="Trust score")
    ax.axhline(0.6, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7, label="Min threshold (0.60)")
    ax.axhline(0.75, color="#f39c12", linewidth=0.8, linestyle=":", alpha=0.7, label="Warn threshold (0.75)")
    ax.fill_between(batches, data["trust"], 0.6,
                    where=data["trust"] < 0.6, alpha=0.15, color="#e74c3c", zorder=0)

    legend_patches = [
        mpatches.Patch(color=_action_colour("reject"),  label="Reject"),
        mpatches.Patch(color=_action_colour("retrain"), label="Retrain"),
        mpatches.Patch(color=_action_colour("promote"), label="Promote"),
        Line2D([0], [0], color="#3498db", linewidth=1.5, label="Trust score"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower left",
              facecolor=ACCENT, labelcolor=TEXT, framealpha=0.8)

    # ── Panel 2: Per-feature PSI ───────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Feature Drift (PSI per feature)", fontsize=10, pad=4)
    ax.set_ylabel("PSI", fontsize=8)
    ax.axhline(0.10, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0.25, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(batches[-1] + 0.3, 0.10, "warn",   color="#f39c12", fontsize=6, va="center")
    ax.text(batches[-1] + 0.3, 0.25, "severe", color="#e74c3c", fontsize=6, va="center")
    ax.fill_between(batches, 0.10, 0.25, alpha=0.08, color="#f39c12")
    ax.fill_between(batches, 0.25, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.25 else 0.6,
                    alpha=0.08, color="#e74c3c")
    colors = ["#9b59b6", "#1abc9c", "#e74c3c"]
    for psi_key, color, label in zip(
        ["psi_f0", "psi_f1", "psi_f2"], colors, ["feature_0", "feature_1", "feature_2"]
    ):
        ax.plot(batches, data[psi_key], color=color, linewidth=1.4, label=label, alpha=0.9)
    # Shade the PSI warning area dynamically
    max_psi = np.maximum.reduce([data["psi_f0"], data["psi_f1"], data["psi_f2"]])
    ax.fill_between(batches, max_psi, 0, alpha=0.06, color="#9b59b6")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left", facecolor=ACCENT, labelcolor=TEXT, framealpha=0.8)

    # ── Panel 3: Output drift + conformal set size ─────────────────────────
    ax = axes[2]
    ax.set_title("Output Drift (PSI)  ·  Conformal Prediction Set Size", fontsize=10, pad=4)
    ax.set_ylabel("Output PSI", fontsize=8, color="#e67e22")
    ax.tick_params(axis="y", colors="#e67e22")
    ax.plot(batches, data["output_psi"], color="#e67e22", linewidth=1.6, label="Output PSI")
    ax.axhline(0.10, color="#e67e22", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL_BG)
    ax2.tick_params(colors="#2ecc71", labelsize=8)
    ax2.set_ylabel("Mean set size", fontsize=8, color="#2ecc71")
    ax2.plot(batches, data["set_size"], color="#2ecc71", linewidth=1.4,
             linestyle="-.", label="Set size", alpha=0.85)
    ax2.axhline(1.5, color="#2ecc71", linewidth=0.7, linestyle=":", alpha=0.6)
    ax2.set_ylim(bottom=1.0)

    lines = [
        Line2D([0], [0], color="#e67e22", linewidth=1.5, label="Output PSI"),
        Line2D([0], [0], color="#2ecc71", linewidth=1.5, linestyle="-.", label="Set size"),
    ]
    ax.legend(handles=lines, fontsize=7, loc="upper left",
              facecolor=ACCENT, labelcolor=TEXT, framealpha=0.8)

    # ── Panel 4: Tail latency ──────────────────────────────────────────────
    ax = axes[3]
    ax.set_title("Tail Latency (p95 / p99)", fontsize=10, pad=4)
    ax.set_ylabel("Latency (ms)", fontsize=8)
    ax.set_xlabel("Batch index", fontsize=8)
    ax.fill_between(batches, data["p95"], data["p99"], alpha=0.25, color="#3498db")
    ax.plot(batches, data["p95"], color="#3498db", linewidth=1.5, label="p95")
    ax.plot(batches, data["p99"], color="#2980b9", linewidth=1.2, linestyle="--",
            alpha=0.8, label="p99")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right", facecolor=ACCENT, labelcolor=TEXT, framealpha=0.8)

    # ── Shared drift event annotation ─────────────────────────────────────
    for ax in axes:
        ax.axvspan(35, 44, alpha=0.04, color="#e74c3c", zorder=0)

    # Annotation arrow on panel 1
    axes[0].annotate(
        "Drift event →\nretrain triggered",
        xy=(40, 0.62), xytext=(28, 0.35),
        color=TEXT, fontsize=7,
        arrowprops={"arrowstyle": "->", "color": TEXT, "lw": 0.8},
    )

    if show:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor=DARK_BG, edgecolor="none")
        print(f"Saved to {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    args = _parse_args()
    generate(Path(args.output), show=args.show, seed=args.seed)
