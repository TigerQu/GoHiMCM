"""
2D Grid Search: Alpha × Beta 参数联合优化

这个脚本证明了为什么需要多参数优化:
- alpha (0.05 ~ 2.0): 距离权重
- beta (1.0 ~ 8.0): 风险权重

关键问题: 单独优化 alpha=0.15 时使用默认 beta=4.0
         但如果 beta 也能优化，最优组合可能完全不同!

实验规模: 10 alpha × 8 beta = 80 种组合
         每个组合 5 个 seed = 400 episodes
运行时间: ~15-20 分钟
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from typing import Dict, Tuple, List

# Add src to path
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


def sweep_grid_alpha_beta():
    """执行 2D 网格搜索: alpha × beta"""
    
    # 搜索范围
    alpha_values = [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 9 values
    beta_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]              # 8 values
    num_seeds = 5  # 每个组合 5 个 seed (降低总耗时)
    
    print("=" * 80)
    print("🔍 2D GRID SEARCH: Alpha × Beta")
    print("=" * 80)
    print(f"\nAlpha 范围: {alpha_values}")
    print(f"Beta 范围:  {beta_values}")
    print(f"每个组合的 seed 数: {num_seeds}")
    print(f"总 episode 数: {len(alpha_values)} × {len(beta_values)} × {num_seeds} = {len(alpha_values) * len(beta_values) * num_seeds}")
    print(f"\n预计运行时间: 15-20 分钟 (在 M1 macOS 上)")
    print("\n" + "=" * 80 + "\n")
    
    # 存储结果
    results = {}  # {alpha: {beta: [results_list]}}
    
    total_episodes = 0
    total = len(alpha_values) * len(beta_values) * num_seeds
    
    start_time = datetime.now()
    
    for i, alpha in enumerate(alpha_values):
        if alpha not in results:
            results[alpha] = {}
        
        for j, beta in enumerate(beta_values):
            results[alpha][beta] = []
            
            for seed in range(num_seeds):
                total_episodes += 1
                
                # 创建配置
                cfg = PlannerConfig(
                    alpha=alpha,
                    beta=beta,
                    gamma=0.2  # 固定 gamma
                )
                
                # 运行 episode
                result = run_one_episode(cfg, seed)
                results[alpha][beta].append(result)
                
                # 进度显示
                pct = (total_episodes / total) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                eta_sec = (elapsed / total_episodes) * (total - total_episodes)
                eta_min = eta_sec / 60
                
                print(f"\r[{pct:5.1f}%] α={alpha:.2f} β={beta:.1f} seed={seed} | "
                      f"rescued={result['rescued']:.0f} time={result['time']:.0f} "
                      f"eff={result['efficiency']:.3f} | ETA: {eta_min:.1f}m", end="", flush=True)
    
    print("\n\n✓ 所有 episode 完成！\n")
    
    # 分析结果
    print("=" * 80)
    print("📊 分析结果 (2D 网格)")
    print("=" * 80 + "\n")
    
    # 计算每个参数组合的平均值
    summary_data = {}  # {(alpha, beta): {mean_rescued, std_rescued, mean_time, std_time, efficiency}}
    
    for alpha in alpha_values:
        for beta in beta_values:
            episode_results = results[alpha][beta]
            rescued_list = [r['rescued'] for r in episode_results]
            time_list = [r['time'] for r in episode_results]
            eff_list = [r['efficiency'] for r in episode_results]
            
            summary_data[(alpha, beta)] = {
                'rescued_mean': np.mean(rescued_list),
                'rescued_std': np.std(rescued_list),
                'time_mean': np.mean(time_list),
                'time_std': np.std(time_list),
                'efficiency_mean': np.mean(eff_list),
                'efficiency_std': np.std(eff_list),
            }
    
    # 找到最优组合
    best_efficiency_combo = max(summary_data.items(), 
                                 key=lambda x: x[1]['efficiency_mean'])
    best_rescued_combo = max(summary_data.items(),
                             key=lambda x: x[1]['rescued_mean'])
    best_time_combo = min(summary_data.items(),
                          key=lambda x: x[1]['time_mean'])
    
    print("🏆 关键发现:\n")
    print(f"  最高效率组合: α={best_efficiency_combo[0][0]:.2f}, β={best_efficiency_combo[0][1]:.1f}")
    print(f"    → 效率 = {best_efficiency_combo[1]['efficiency_mean']:.4f} rescued/step")
    print(f"    → 救援 = {best_efficiency_combo[1]['rescued_mean']:.1f} ± {best_efficiency_combo[1]['rescued_std']:.1f}")
    print(f"    → 时间 = {best_efficiency_combo[1]['time_mean']:.1f} ± {best_efficiency_combo[1]['time_std']:.1f}")
    
    print(f"\n  最多救援组合: α={best_rescued_combo[0][0]:.2f}, β={best_rescued_combo[0][1]:.1f}")
    print(f"    → 救援 = {best_rescued_combo[1]['rescued_mean']:.1f} ± {best_rescued_combo[1]['rescued_std']:.1f}")
    print(f"    → 时间 = {best_rescued_combo[1]['time_mean']:.1f} ± {best_rescued_combo[1]['time_std']:.1f}")
    print(f"    → 效率 = {best_rescued_combo[1]['efficiency_mean']:.4f} rescued/step")
    
    print(f"\n  最快清理组合: α={best_time_combo[0][0]:.2f}, β={best_time_combo[0][1]:.1f}")
    print(f"    → 时间 = {best_time_combo[1]['time_mean']:.1f} ± {best_time_combo[1]['time_std']:.1f}")
    print(f"    → 救援 = {best_time_combo[1]['rescued_mean']:.1f} ± {best_time_combo[1]['rescued_std']:.1f}")
    print(f"    → 效率 = {best_time_combo[1]['efficiency_mean']:.4f} rescued/step")
    
    print("\n⚠️  关键观察:\n")
    if best_efficiency_combo[0] != (0.15, 4.0):
        print(f"  ❌ 之前推荐的 α=0.15, β=4.0 NOT 最优!")
        print(f"  ✅ 最优组合是 α={best_efficiency_combo[0][0]:.2f}, β={best_efficiency_combo[0][1]:.1f}")
        print(f"  📈 效率提升: {(best_efficiency_combo[1]['efficiency_mean'] / 0.544 - 1) * 100:.1f}%")
    else:
        print(f"  ✓ 之前推荐的 α=0.15, β=4.0 确实是最优!")
    
    # 生成可视化
    print("\n" + "=" * 80)
    print("📈 生成可视化...")
    print("=" * 80 + "\n")
    
    # 准备数据用于热力图
    alpha_arr = np.array(alpha_values)
    beta_arr = np.array(beta_values)
    
    efficiency_matrix = np.zeros((len(beta_values), len(alpha_values)))
    rescued_matrix = np.zeros((len(beta_values), len(alpha_values)))
    time_matrix = np.zeros((len(beta_values), len(alpha_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            data = summary_data[(alpha, beta)]
            efficiency_matrix[j, i] = data['efficiency_mean']
            rescued_matrix[j, i] = data['rescued_mean']
            time_matrix[j, i] = data['time_mean']
    
    # 绘制热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 效率热力图
    im1 = axes[0].imshow(efficiency_matrix, cmap='YlGn', aspect='auto', origin='lower')
    axes[0].set_xlabel('Alpha', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Beta', fontsize=12, fontweight='bold')
    axes[0].set_title('效率 (Rescued/Time)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(alpha_values)))
    axes[0].set_xticklabels([f'{a:.2f}' for a in alpha_values], rotation=45)
    axes[0].set_yticks(range(len(beta_values)))
    axes[0].set_yticklabels([f'{b:.1f}' for b in beta_values])
    plt.colorbar(im1, ax=axes[0])
    
    # 救援人数热力图
    im2 = axes[1].imshow(rescued_matrix, cmap='Blues', aspect='auto', origin='lower')
    axes[1].set_xlabel('Alpha', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Beta', fontsize=12, fontweight='bold')
    axes[1].set_title('平均救援人数', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(alpha_values)))
    axes[1].set_xticklabels([f'{a:.2f}' for a in alpha_values], rotation=45)
    axes[1].set_yticks(range(len(beta_values)))
    axes[1].set_yticklabels([f'{b:.1f}' for b in beta_values])
    plt.colorbar(im2, ax=axes[1])
    
    # 时间热力图
    im3 = axes[2].imshow(time_matrix, cmap='Reds_r', aspect='auto', origin='lower')
    axes[2].set_xlabel('Alpha', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Beta', fontsize=12, fontweight='bold')
    axes[2].set_title('平均清理时间', fontsize=13, fontweight='bold')
    axes[2].set_xticks(range(len(alpha_values)))
    axes[2].set_xticklabels([f'{a:.2f}' for a in alpha_values], rotation=45)
    axes[2].set_yticks(range(len(beta_values)))
    axes[2].set_yticklabels([f'{b:.1f}' for b in beta_values])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'grid_search_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {output_path}")
    plt.close()
    
    # 保存数据到 JSON
    json_data = {
        'alpha_values': alpha_values,
        'beta_values': beta_values,
        'summary': {
            f"{alpha}_{beta}": summary_data[(alpha, beta)]
            for alpha in alpha_values
            for beta in beta_values
        },
        'best_efficiency': {
            'alpha': best_efficiency_combo[0][0],
            'beta': best_efficiency_combo[0][1],
            'efficiency': best_efficiency_combo[1]['efficiency_mean'],
        },
        'best_rescued': {
            'alpha': best_rescued_combo[0][0],
            'beta': best_rescued_combo[0][1],
            'rescued': best_rescued_combo[1]['rescued_mean'],
        },
        'best_time': {
            'alpha': best_time_combo[0][0],
            'beta': best_time_combo[0][1],
            'time': best_time_combo[1]['time_mean'],
        }
    }
    
    json_path = os.path.join(os.path.dirname(__file__), 'grid_search_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ 保存: {json_path}")
    
    print("\n" + "=" * 80)
    print("✅ 2D 网格搜索完成!")
    print("=" * 80)
    
    return summary_data, results


if __name__ == '__main__':
    summary_data, results = sweep_grid_alpha_beta()
