#!/usr/bin/env python3
"""
Template for running alpha sweep on different layouts.

Usage:
    python sweep_all_layouts.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traditional_planner.alpha_sweep_experiment import (
    sweep_alpha_on_daycare
)


def main():
    """Run alpha sweep experiments on all layouts."""
    print("=" * 70)
    print("ALPHA PARAMETER SWEEP - COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    print("\n📊 Currently running on: Babycare layout")
    print("   Nodes: 41, Floors: 3")
    print("   Alpha values: 23 (0.05 to 2.0)")
    print("   Seeds per alpha: 10")
    print("   Total episodes: 230\n")

    sweep_alpha_on_daycare()

    print("\n" + "=" * 70)
    print("✓ Experiment completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • daycare_alpha_rescued.png")
    print("  • daycare_alpha_time.png")
    print("  • daycare_alpha_efficiency.png")
    print("  • daycare_alpha_tradeoff.png")
    print("  • daycare_alpha_combined.png")
    print("\nYou can extend this to run on office/warehouse layouts by")
    print("modifying the build_env_fn parameter in alpha_sweep_experiment.py")


if __name__ == "__main__":
    main()
