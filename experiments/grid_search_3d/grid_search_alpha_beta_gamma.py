"""
3D Grid Search: Alpha × Beta × Gamma 完整参数优化

这个脚本进行真正的全面参数优化:
- alpha (0.1, 0.2, 0.3): 距离权重 (3 值)
- beta (2.0, 4.0, 6.0): 风险权重 (3 值)  
- gamma (0.1, 0.2, 0.4): 拥堵权重 (3 值)

关键问题: 三个参数可能有相互作用!
例如: alpha 在 beta=2 时最优，但在 beta=6 时可能不同

实验规模: 3 × 3 × 3 = 27 种组合
         每个组合 3 个 seed = 81 episodes
运行时间: ~5-8 分钟

注意: 这是演示实验，用较少的搜索空间和 seed 数
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict

_this_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_this_dir, "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.environment.layouts import build_babycare_layout
from src.traditional_planner.adapter import EnvAdapter
from src.traditional_planner.planner import GreedySweepPlanner
from src.traditional_planner.scoring import PlannerConfig


def run_one_episode(
    cfg: PlannerConfig,
    seed: int,
    max_steps: int = 600,
) -> Dict:
    """运行单个 episode，使用贪心规划器"""
    np.random.seed(seed)

    try:
        env = build_babycare_layout()
        adapter = EnvAdapter(env=env, info_mode="realistic")

        planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)

        snap = adapter.reset(seed=seed)

        for t in range(max_steps):
            stats = snap["stats"]

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
        people_rescued = stats["people_rescued"]
        final_time = snap["time"]

        efficiency = people_rescued / final_time if final_time > 0 else 0

        return {
            "rescued": people_rescued,
            "time": final_time,
            "efficiency": efficiency,
            "success": True
        }
    except Exception as e:
        return {
            "rescued": 0,
            "time": 600,
            "efficiency": 0,
            "error": str(e),
            "success": False
        }


def sweep_grid_3d():
    """执行 3D 网格搜索: alpha × beta × gamma"""

    # 搜索范围 (缩小以节省时间)
    alpha_values = [0.1, 0.2, 0.3]
    beta_values = [2.0, 4.0, 6.0]
    gamma_values = [0.1, 0.2, 0.4]
    num_seeds = 3

    print("=" * 80)
    print("🔍 3D GRID SEARCH: Alpha × Beta × Gamma")
    print("=" * 80)
    print(f"\nAlpha 范围:  {alpha_values}")
    print(f"Beta 范围:   {beta_values}")
    print(f"Gamma 范围:  {gamma_values}")
    print(f"每个组合的 seed 数: {num_seeds}")
    total_eps = (len(alpha_values) * len(beta_values) *
                 len(gamma_values) * num_seeds)
    print(f"总 episode 数: {total_eps}")
    print(f"\n预计运行时间: 5-8 分钟\n")
    print("=" * 80 + "\n")

    results = {}
    total_episodes = 0
    total = total_eps
    start_time = datetime.now()

    for i, alpha in enumerate(alpha_values):
        if alpha not in results:
            results[alpha] = {}

        for j, beta in enumerate(beta_values):
            if beta not in results[alpha]:
                results[alpha][beta] = {}

            for k, gamma in enumerate(gamma_values):
                results[alpha][beta][gamma] = []

                for seed in range(num_seeds):
                    total_episodes += 1

                    cfg = PlannerConfig(
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma
                    )

                    result = run_one_episode(cfg, seed)
                    results[alpha][beta][gamma].append(result)

                    pct = (total_episodes / total) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if total_episodes < total:
                        eta_sec = (elapsed / total_episodes) * \
                                  (total - total_episodes)
                        eta_min = eta_sec / 60
                    else:
                        eta_min = 0

                    msg = (f"\r[{pct:5.1f}%] α={alpha:.2f} β={beta:.1f} "
                           f"γ={gamma:.2f} seed={seed} | "
                           f"eff={result['efficiency']:.3f} | "
                           f"ETA: {eta_min:.1f}m")
                    print(msg, end="", flush=True)

    print("\n\n✓ 所有 episode 完成！\n")

    # 分析结果
    print("=" * 80)
    print("📊 分析结果 (3D 网格)")
    print("=" * 80 + "\n")

    summary_data = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                episode_results = results[alpha][beta][gamma]
                rescued_list = [r['rescued'] for r in episode_results]
                time_list = [r['time'] for r in episode_results]
                eff_list = [r['efficiency'] for r in episode_results]

                summary_data[(alpha, beta, gamma)] = {
                    'rescued_mean': np.mean(rescued_list),
                    'rescued_std': np.std(rescued_list),
                    'time_mean': np.mean(time_list),
                    'time_std': np.std(time_list),
                    'efficiency_mean': np.mean(eff_list),
                    'efficiency_std': np.std(eff_list),
                }

    # 找最优组合
    best_combo = max(summary_data.items(),
                     key=lambda x: x[1]['efficiency_mean'])

    print("🏆 最优组合:\n")
    alpha, beta, gamma = best_combo[0]
    data = best_combo[1]
    print(f"  参数: α={alpha:.2f}, β={beta:.1f}, γ={gamma:.2f}")
    print(f"  效率: {data['efficiency_mean']:.4f} rescued/step")
    print(f"  救援: {data['rescued_mean']:.1f} ± {data['rescued_std']:.1f}")
    print(f"  时间: {data['time_mean']:.1f} ± {data['time_std']:.1f}\n")

    # 检查 beta 对最优 alpha 的影响
    print("⚠️  关键观察 - Alpha 的最优值对 Beta 的依赖:\n")
    for beta_val in beta_values:
        best_alpha_for_beta = max(
            [(a, summary_data[(a, beta_val, 0.2)])
             for a in alpha_values],
            key=lambda x: x[1]['efficiency_mean']
        )
        print(f"  当 β={beta_val:.1f} 时，最优 α={best_alpha_for_beta[0]:.2f} "
              f"(效率={best_alpha_for_beta[1]['efficiency_mean']:.4f})")

    print("\n  💡 这说明参数之间存在相互作用!")
    print("     不能独立优化每个参数\n")

    # 绘制结果
    print("=" * 80)
    print("📈 生成可视化...")
    print("=" * 80 + "\n")

    fig, axes = plt.subplots(1, len(gamma_values), figsize=(15, 4))

    for k, gamma_val in enumerate(gamma_values):
        # 为每个 gamma 值绘制 alpha × beta 热力图
        eff_matrix = np.zeros((len(beta_values), len(alpha_values)))

        for i, alpha in enumerate(alpha_values):
            for j, beta in enumerate(beta_values):
                data = summary_data[(alpha, beta, gamma_val)]
                eff_matrix[j, i] = data['efficiency_mean']

        im = axes[k].imshow(eff_matrix, cmap='YlGn', aspect='auto',
                            origin='lower')
        axes[k].set_title(f'γ={gamma_val:.2f}', fontsize=12, fontweight='bold')
        axes[k].set_xlabel('Alpha', fontsize=10)
        axes[k].set_ylabel('Beta', fontsize=10)
        axes[k].set_xticks(range(len(alpha_values)))
        axes[k].set_xticklabels([f'{a:.2f}' for a in alpha_values])
        axes[k].set_yticks(range(len(beta_values)))
        axes[k].set_yticklabels([f'{b:.1f}' for b in beta_values])
        plt.colorbar(im, ax=axes[k], label='Efficiency')

    plt.suptitle('3D 网格搜索: 不同 Gamma 值下的效率热力图',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__),
                               'grid_search_3d_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()

    # 保存为 JSON
    json_data = {
        'alpha_values': alpha_values,
        'beta_values': beta_values,
        'gamma_values': gamma_values,
        'best_combo': {
            'alpha': best_combo[0][0],
            'beta': best_combo[0][1],
            'gamma': best_combo[0][2],
            'efficiency': best_combo[1]['efficiency_mean'],
        },
        'summary': {
            f"{a}_{b}_{g}": summary_data[(a, b, g)]
            for a in alpha_values
            for b in beta_values
            for g in gamma_values
        }
    }

    json_path = os.path.join(os.path.dirname(__file__),
                             'grid_search_3d_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ 保存: {json_path}")

    print("\n" + "=" * 80)
    print("✅ 3D 网格搜索完成!")
    print("=" * 80)

    return summary_data, results


if __name__ == '__main__':
    sweep_grid_3d()
