#!/usr/bin/env python3
"""
Guide for extending alpha sweep to other parameters (beta, gamma, etc).

This file shows how to adapt the alpha_sweep_experiment.py to sweep
other parameters like beta (risk reward weight) or gamma (congestion penalty).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Callable, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from environment.layouts import build_babycare_layout
from traditional_planner.adapter import EnvAdapter
from traditional_planner.planner import GreedySweepPlanner
from traditional_planner.scoring import PlannerConfig


def run_one_episode_beta(
    build_env_fn: Callable[[], Any],
    beta_value: float,
    seed: int,
    max_steps: int = 600,
) -> Dict[str, Any]:
    """
    Run ONE episode with a given beta (risk reward weight).
    Similar structure to alpha sweep, just change which parameter.
    """
    env = build_env_fn()
    adapter = EnvAdapter(env=env, info_mode="realistic")

    cfg = PlannerConfig()
    cfg.beta = beta_value  # ← Sweep beta instead of alpha

    planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)
    snap = adapter.reset(seed=seed)

    for t in range(max_steps):
        nodes = snap["nodes"]
        frontier = [
            nid
            for nid, info in nodes.items()
            if info["type"] == "room" and not info["swept"]
        ]
        if not frontier:
            break

        actions = planner.plan_step(snap)

        for aid, ainfo in actions.items():
            act_type = ainfo["action"]
            dest = ainfo.get("dest")

            if act_type == "move" and dest is not None:
                adapter.move(aid, dest)
            elif act_type == "search":
                adapter.search(aid)

        adapter.step()
        snap = adapter.snapshot()

    stats = snap["stats"]
    return dict(
        beta=beta_value,
        seed=seed,
        rescued=stats["people_rescued"],
        final_time=snap["time"],
    )


def example_beta_sweep():
    """Example: sweep beta parameter (risk reward weight)."""
    print("\n=== EXAMPLE: BETA PARAMETER SWEEP ===")
    print("This shows the same structure as alpha sweep,")
    print("but changing the 'beta' parameter instead of 'alpha'.\n")

    build_env_fn = build_babycare_layout
    beta_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    num_seeds = 5

    print("To run this:")
    print("  1. Uncomment the code below")
    print("  2. Run: python sweep_parameter_guide.py")
    print("")
    print("Expected output:")
    print("  [beta=1.0, seed=0] rescued=..., time=...")
    print("  [beta=1.0, seed=1] rescued=..., time=...")
    print("  ... (for each beta and seed)")
    print("")
    print("Then plot the results just like we did for alpha!")


def example_multi_parameter():
    """
    Example: sweep TWO parameters simultaneously
    (more complex, but possible)
    """
    print("\n=== EXAMPLE: 2D PARAMETER SWEEP (alpha × beta) ===")
    print("This shows how to do a grid search over multiple parameters.\n")

    print("Pseudocode:")
    print("  for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:")
    print("    for beta in [1.0, 2.0, 3.0, 4.0, 5.0]:")
    print("      for seed in range(num_seeds):")
    print("        cfg.alpha = alpha")
    print("        cfg.beta = beta")
    print("        run_episode(cfg, seed)")
    print("")
    print("Results: 5 × 5 × num_seeds = 125+ episodes")
    print("Plot: heatmap of (alpha, beta) → mean_rescued or efficiency")


def guide():
    """Print comprehensive guide."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP EXTENSION GUIDE")
    print("=" * 70)

    print("\n📚 PARAMETERS IN PlannerConfig (scoring.py):")
    print("""
    Alpha-level parameters:
    ├── alpha (default=0.2)
    │   └─ Travel distance penalty weight
    │      (Higher → prefer closer targets)
    │
    ├── beta (default=4.0)
    │   └─ Risk "reward" weight (note: negated in score)
    │      (Higher → prefer high-risk rooms)
    │
    └── gamma (default=0.2)
        └─ Congestion penalty weight
           (Higher → avoid congested routes)

    Risk-level parameters (used in compute_risk):
    ├── w_intensity (default=0.5)
    │   └─ Weight of fire intensity in risk
    │
    ├── w_smoke (default=0.2)
    │   └─ Weight of smoke presence in risk
    │
    ├── w_dist_fire (default=0.2)
    │   └─ Weight of distance from fire
    │
    ├── w_neighbor_fire (default=0.1)
    │   └─ Weight of nearby fire sources
    │
    └── w_civilian_hint (default=0.3)
        └─ Weight of civilian presence
    """)

    print("\n🔄 HOW TO ADAPT alpha_sweep_experiment.py:")
    print("""
    Step 1: Change parameter name
    ────────────────────────────
    OLD: cfg.alpha = alpha_value
    NEW: cfg.beta = beta_value    # or any other parameter

    Step 2: Update sweep range
    ────────────────────────────
    alpha_values = [0.05, 0.1, 0.15, ..., 2.0]
    # Change to:
    beta_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    Step 3: Update results aggregation
    ────────────────────────────────────
    for alpha in alpha_values:
    # Change to:
    for beta in beta_values:

    Step 4: Update plot labels
    ───────────────────────────
    plt.xlabel(r"distance weight $α$")
    # Change to:
    plt.xlabel(r"risk reward weight $β$")
    """)

    print("\n💡 RECOMMENDATIONS FOR DIFFERENT PARAMETERS:")
    print("""
    α (distance weight):
      Range: [0.01, 3.0]
      Step: 0.1-0.2
      Why: Controls distance vs. utility tradeoff
      
    β (risk reward weight):
      Range: [1.0, 8.0]
      Step: 0.5-1.0
      Why: Controls how much risk prioritization matters
      
    γ (congestion penalty):
      Range: [0.0, 1.0]
      Step: 0.1-0.2
      Why: Low values favor efficient paths
      
    Risk weights (w_intensity, w_smoke, etc):
      Range: [0.0, 1.0]
      Step: 0.1-0.2
      Why: Relative importance of risk features
    """)

    print("\n🎯 ADVANCED: 2D GRID SEARCH")
    print("""
    To compare TWO parameters simultaneously:
    
    1. Create grid: alpha × beta
    2. Run all combinations
    3. Plot heatmap: imshow(results, aspect='auto')
    4. Find Pareto frontier (best tradeoff)
    
    This helps understand parameter interactions!
    """)

    example_beta_sweep()
    example_multi_parameter()

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    guide()
