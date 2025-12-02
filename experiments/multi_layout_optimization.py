#!/usr/bin/env python3
"""
Multi-Layout Parameter Optimization

对 office、babycare 和 warehouse 分别进行参数优化，
然后找出综合最优参数。
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.environment.layouts import (
    build_standard_office_layout,
    build_babycare_layout,
    build_two_floor_warehouse,
)
from src.traditional_planner.adapter import EnvAdapter
from src.traditional_planner.planner import GreedySweepPlanner
from src.traditional_planner.scoring import PlannerConfig


def run_episode(layout_name: str, alpha: float, beta: float,
                gamma: float, seed: int, max_steps: int = 600) -> Dict:
    """运行单个 episode"""
    np.random.seed(seed)

    try:
        if layout_name == "office":
            env = build_standard_office_layout()
        elif layout_name == "babycare":
            env = build_babycare_layout()
        elif layout_name == "warehouse":
            env = build_two_floor_warehouse()
        else:
            raise ValueError(f"Unknown layout: {layout_name}")

        adapter = EnvAdapter(env=env, info_mode="realistic")

        cfg = PlannerConfig()
        cfg.alpha = alpha
        cfg.beta = beta
        cfg.gamma = gamma

        planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)
        snap = adapter.reset(seed=seed)

        for t in range(max_steps):
            nodes = snap["nodes"]
            frontier = [
                nid for nid, info in nodes.items()
                if info.get("type") == "room" and not info.get("swept")
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
        rescued = stats.get("people_rescued", 0)
        time = snap["time"]
        swept = stats.get("nodes_swept", 0)
        efficiency = rescued / time if time > 0 else 0

        return {
            "efficiency": efficiency,
            "rescued": rescued,
            "time": time,
            "swept": swept,
            "success": True,
        }

    except Exception:
        return {
            "efficiency": 0,
            "rescued": 0,
            "time": 0,
            "swept": 0,
            "success": False,
        }


def optimize_layout(layout_name: str, alphas: List[float] = None,
                    betas: List[float] = None, gammas: List[float] = None,
                    num_seeds: int = 5, verbose: bool = True) -> List[Dict]:
    """对单个 layout 进行网格搜索优化"""

    if alphas is None:
        alphas = [0.10, 0.20, 0.30, 0.50]
    if betas is None:
        betas = [1.0, 2.0, 4.0, 6.0, 8.0]
    if gammas is None:
        gammas = [0.1, 0.2, 0.4]

    results = []
    total = len(alphas) * len(betas) * len(gammas) * num_seeds

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing {layout_name.upper()}")
        print(f"{'='*70}")
        print(f"Grid: {len(alphas)}α × {len(betas)}β × {len(gammas)}γ "
              f"= {len(alphas)*len(betas)*len(gammas)} combinations")
        print(f"Total: {total} episodes\n")

    count = 0
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                episode_results = []

                for seed in range(num_seeds):
                    count += 1
                    result = run_episode(layout_name, alpha, beta,
                                         gamma, seed)

                    if result["success"]:
                        episode_results.append(result)

                    if verbose and count % 30 == 0:
                        pct = 100.0 * count / total
                        print(f"  {count}/{total} ({pct:.0f}%)")

                if episode_results:
                    results.append({
                        "layout": layout_name,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "avg_efficiency": np.mean(
                            [r["efficiency"] for r in episode_results]
                        ),
                        "avg_rescued": np.mean(
                            [r["rescued"] for r in episode_results]
                        ),
                        "avg_time": np.mean(
                            [r["time"] for r in episode_results]
                        ),
                    })

    if verbose:
        print(f"Completed!\n")

    return results


def main():
    """Main optimization pipeline"""

    print("\n" + "="*70)
    print("MULTI-LAYOUT PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Optimize each layout
    office_results = optimize_layout("office", num_seeds=5)
    babycare_results = optimize_layout("babycare", num_seeds=5)
    warehouse_results = optimize_layout("warehouse", num_seeds=5)

    # Find best for each
    best_office = max(office_results, key=lambda x: x["avg_efficiency"])
    best_babycare = max(babycare_results,
                        key=lambda x: x["avg_efficiency"])
    best_warehouse = max(warehouse_results,
                         key=lambda x: x["avg_efficiency"])

    # Find unified best
    all_results = office_results + babycare_results + warehouse_results

    # Normalize
    office_avg = np.mean([r["avg_efficiency"] for r in office_results])
    babycare_avg = np.mean([r["avg_efficiency"] for r in babycare_results])
    warehouse_avg = np.mean([r["avg_efficiency"] for r in warehouse_results])

    best_unified = None
    best_score = -1

    for alpha in set(r["alpha"] for r in all_results):
        for beta in set(r["beta"] for r in all_results):
            for gamma in set(r["gamma"] for r in all_results):
                results_for_params = [
                    r for r in all_results
                    if r["alpha"] == alpha and r["beta"] == beta
                    and r["gamma"] == gamma
                ]

                if len(results_for_params) == 3:
                    scores = []
                    for r in results_for_params:
                        if r["layout"] == "office":
                            scores.append(r["avg_efficiency"] / max(
                                office_avg, 0.001))
                        elif r["layout"] == "babycare":
                            scores.append(r["avg_efficiency"] / max(
                                babycare_avg, 0.001))
                        else:
                            scores.append(r["avg_efficiency"] / max(
                                warehouse_avg, 0.001))

                    score = np.mean(scores)
                    if score > best_score:
                        best_score = score
                        best_unified = {
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "score": score,
                        }

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__),
                              "multi_layout_results")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "best_office": {
            "alpha": best_office["alpha"],
            "beta": best_office["beta"],
            "gamma": best_office["gamma"],
            "efficiency": best_office["avg_efficiency"],
        },
        "best_babycare": {
            "alpha": best_babycare["alpha"],
            "beta": best_babycare["beta"],
            "gamma": best_babycare["gamma"],
            "efficiency": best_babycare["avg_efficiency"],
        },
        "best_warehouse": {
            "alpha": best_warehouse["alpha"],
            "beta": best_warehouse["beta"],
            "gamma": best_warehouse["gamma"],
            "efficiency": best_warehouse["avg_efficiency"],
        },
        "best_unified": {
            "alpha": best_unified["alpha"],
            "beta": best_unified["beta"],
            "gamma": best_unified["gamma"],
            "score": best_unified["score"],
        },
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("="*70)
    print("RESULTS")
    print("="*70 + "\n")

    print("📍 OFFICE BEST:")
    print(f"   α={best_office['alpha']:.2f}, β={best_office['beta']:.1f}, "
          f"γ={best_office['gamma']:.2f}")
    print(f"   Efficiency: {best_office['avg_efficiency']:.4f}\n")

    print("🏥 BABYCARE BEST:")
    print(f"   α={best_babycare['alpha']:.2f}, β={best_babycare['beta']:.1f}, "
          f"γ={best_babycare['gamma']:.2f}")
    print(f"   Efficiency: {best_babycare['avg_efficiency']:.4f}\n")

    print("🏭 WAREHOUSE BEST:")
    print(f"   α={best_warehouse['alpha']:.2f}, "
          f"β={best_warehouse['beta']:.1f}, γ={best_warehouse['gamma']:.2f}")
    print(f"   Efficiency: {best_warehouse['avg_efficiency']:.4f}\n")

    print("🎯 UNIFIED BEST:")
    print(f"   α={best_unified['alpha']:.2f}, β={best_unified['beta']:.1f}, "
          f"γ={best_unified['gamma']:.2f}")
    print(f"   Score: {best_unified['score']:.4f}\n")

    print(f"Saved to: {output_dir}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    with open(os.path.join(output_dir, "office_results.json"), "w") as f:
        json.dump(office_results, f, indent=2)
    with open(os.path.join(output_dir, "babycare_results.json"), "w") as f:
        json.dump(babycare_results, f, indent=2)
    with open(os.path.join(output_dir, "warehouse_results.json"), "w") as f:
        json.dump(warehouse_results, f, indent=2)


if __name__ == "__main__":
    main()
