#!/usr/bin/env python3
"""
Quick Multi-Layout Parameter Optimization (Simplified)

对 office、babycare 和 warehouse 分别进行参数优化。
为了快速完成，减少参数搜索空间。
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


def optimize_layout_quick(layout_name: str, num_seeds: int = 3) -> List[Dict]:
    """对单个 layout 进行快速参数优化"""

    # 缩小搜索空间以加快速度
    alphas = [0.15, 0.20, 0.25, 0.30]
    betas = [2.0, 4.0, 6.0]
    gammas = [0.1, 0.2, 0.3]

    results = []
    total = len(alphas) * len(betas) * len(gammas) * num_seeds

    print(f"\n{'='*70}")
    print(f"OPTIMIZING {layout_name.upper()}")
    print(f"{'='*70}")
    print(f"Search space: {len(alphas)}α × {len(betas)}β × {len(gammas)}γ "
          f"= {len(alphas)*len(betas)*len(gammas)} combinations")
    print(f"Seeds: {num_seeds}, Total episodes: {total}\n")

    count = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for k, gamma in enumerate(gammas):
                episode_results = []

                for seed in range(num_seeds):
                    count += 1
                    result = run_episode(layout_name, alpha, beta,
                                         gamma, seed)

                    if result["success"]:
                        episode_results.append(result)

                    if count % 12 == 0:
                        pct = 100.0 * count / total
                        print(f"  [{count:3d}/{total}] {pct:5.1f}%  "
                              f"(α={alpha:.2f}, β={beta:.1f}, γ={gamma:.1f})")

                if episode_results:
                    results.append({
                        "layout": layout_name,
                        "alpha": round(alpha, 2),
                        "beta": round(beta, 1),
                        "gamma": round(gamma, 1),
                        "avg_efficiency": round(
                            np.mean([r["efficiency"]
                                     for r in episode_results]), 4),
                        "avg_rescued": round(
                            np.mean([r["rescued"]
                                     for r in episode_results]), 2),
                        "avg_time": round(
                            np.mean([r["time"]
                                     for r in episode_results]), 1),
                    })

    print(f"\n✅ {layout_name} completed!\n")
    return results


def main():
    """Main optimization pipeline"""

    print("\n" + "="*70)
    print("QUICK MULTI-LAYOUT PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Optimize each layout (faster)
    print("⏳ Running office optimization...")
    office_results = optimize_layout_quick("office", num_seeds=3)

    print("⏳ Running babycare optimization...")
    babycare_results = optimize_layout_quick("babycare", num_seeds=3)

    print("⏳ Running warehouse optimization...")
    warehouse_results = optimize_layout_quick("warehouse", num_seeds=3)

    # Find best for each
    best_office = max(office_results, key=lambda x: x["avg_efficiency"])
    best_babycare = max(babycare_results,
                        key=lambda x: x["avg_efficiency"])
    best_warehouse = max(warehouse_results,
                         key=lambda x: x["avg_efficiency"])

    # Find unified best
    all_results = office_results + babycare_results + warehouse_results

    office_avg = np.mean([r["avg_efficiency"] for r in office_results])
    babycare_avg = np.mean([r["avg_efficiency"] for r in babycare_results])
    warehouse_avg = np.mean([r["avg_efficiency"] for r in warehouse_results])

    best_unified = None
    best_score = -1

    for alpha in sorted(set(r["alpha"] for r in all_results)):
        for beta in sorted(set(r["beta"] for r in all_results)):
            for gamma in sorted(set(r["gamma"] for r in all_results)):
                results_for_params = [
                    r for r in all_results
                    if r["alpha"] == alpha and r["beta"] == beta
                    and r["gamma"] == gamma
                ]

                if len(results_for_params) == 3:
                    scores = []
                    for r in results_for_params:
                        if r["layout"] == "office":
                            norm = r["avg_efficiency"] / max(office_avg, 0.001)
                        elif r["layout"] == "babycare":
                            norm = r["avg_efficiency"] / max(babycare_avg,
                                                             0.001)
                        else:
                            norm = r["avg_efficiency"] / max(warehouse_avg,
                                                             0.001)
                        scores.append(norm)

                    score = np.mean(scores)
                    if score > best_score:
                        best_score = score
                        best_unified = {
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "score": round(score, 4),
                        }

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__),
                              "multi_layout_results")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "best_office": best_office,
        "best_babycare": best_babycare,
        "best_warehouse": best_warehouse,
        "best_unified": best_unified,
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70 + "\n")

    print("📍 BEST OFFICE:")
    print(f"   α={best_office['alpha']}, β={best_office['beta']}, "
          f"γ={best_office['gamma']}")
    print(f"   Efficiency: {best_office['avg_efficiency']}\n")

    print("🏥 BEST BABYCARE:")
    print(f"   α={best_babycare['alpha']}, β={best_babycare['beta']}, "
          f"γ={best_babycare['gamma']}")
    print(f"   Efficiency: {best_babycare['avg_efficiency']}\n")

    print("🏭 BEST WAREHOUSE:")
    print(f"   α={best_warehouse['alpha']}, β={best_warehouse['beta']}, "
          f"γ={best_warehouse['gamma']}")
    print(f"   Efficiency: {best_warehouse['avg_efficiency']}\n")

    print("🎯 UNIFIED BEST:")
    print(f"   α={best_unified['alpha']}, β={best_unified['beta']}, "
          f"γ={best_unified['gamma']}")
    print(f"   Score: {best_unified['score']}\n")

    print(f"Saved to: {output_dir}/summary.json")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
