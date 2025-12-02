# src/traditional_planner/alpha_sweep_experiment.py

from __future__ import annotations
from typing import Callable, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

# ====== imports from your project ======
try:
    from environment.layouts import (
        build_standard_office_layout,
        build_babycare_layout,
        build_two_floor_warehouse,
    )
except Exception:
    import os
    import sys

    _this_dir = os.path.dirname(__file__)
    _src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from environment.layouts import (
        build_standard_office_layout,
        build_babycare_layout,
        build_two_floor_warehouse,
    )

from traditional_planner.adapter import EnvAdapter
from traditional_planner.planner import GreedySweepPlanner
from traditional_planner.scoring import PlannerConfig


# ============================================================
# 1) 单次 episode：给定 alpha，返回 rescued / final_time 等统计
# ============================================================

def run_one_episode(
    build_env_fn: Callable[[], Any],
    alpha_value: float,
    seed: int,
    max_steps: int = 600,
    info_mode: str = "realistic",
) -> Dict[str, Any]:
    """
    Run ONE episode of the greedy planner with a given alpha.
    Returns a dict with rescued count and final sweep time.
    """
    # ---- build env & adapter ----
    env = build_env_fn()
    adapter = EnvAdapter(env=env, info_mode=info_mode)

    # ---- planner config: 这里把 alpha 写进去 ----
    cfg = PlannerConfig()
    cfg.alpha = alpha_value  # 设置距离权重

    planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)

    # ---- reset env ----
    snap = adapter.reset(seed=seed)

    # ---- main loop ----
    for t in range(max_steps):
        stats = snap["stats"]

        # 终止条件：所有 room swept
        # （如果你有更严格的终止逻辑，可以在这里补充）
        nodes = snap["nodes"]
        frontier = [
            nid
            for nid, info in nodes.items()
            if info["type"] == "room" and not info["swept"]
        ]
        if not frontier:
            break

        # 让 greedy planner 选动作
        actions = planner.plan_step(snap)

        # 应用动作
        for aid, ainfo in actions.items():
            act_type = ainfo["action"]
            dest = ainfo.get("dest")

            if act_type == "move" and dest is not None:
                adapter.move(aid, dest)
            elif act_type == "search":
                adapter.search(aid)
            # "wait" -> do nothing

        # step 环境
        adapter.step()
        snap = adapter.snapshot()

    # 结束时统计
    stats = snap["stats"]
    rescued = stats["people_rescued"]
    nodes_swept = stats["nodes_swept"]
    final_time = snap["time"]

    return dict(
        alpha=alpha_value,
        seed=seed,
        rescued=rescued,
        nodes_swept=nodes_swept,
        final_time=final_time,
    )


# ============================================================
# 2) 对一串 alpha 做 sweep + 多个 seed 取平均
# ============================================================

def sweep_alpha_on_daycare() -> None:
    """
    在 daycare layout 上，对一串 alpha 做实验，画 3 张图：
      - rescued vs alpha
      - final_time vs alpha
      - trade-off: rescued vs final_time
    """
    build_env_fn = build_babycare_layout  # 你也可以换成 office / warehouse

    # 测试更多的 alpha 值，范围从 0.05 到 2.0
    alpha_values = [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    ]

    # 每个 alpha 跑多少 seed，越多越平滑
    num_seeds = 10

    all_results: List[Dict[str, Any]] = []

    for alpha in alpha_values:
        for seed in range(num_seeds):
            res = run_one_episode(
                build_env_fn=build_env_fn,
                alpha_value=alpha,
                seed=seed,
                max_steps=600,
                info_mode="realistic",
            )
            all_results.append(res)
            print(
                f"[alpha={alpha:.2f}, seed={seed}] "
                f"rescued={res['rescued']}, time={res['final_time']}"
            )

    # ---- 把结果整理成 numpy array 方便算 mean / std ----
    alpha_arr = np.array(alpha_values, dtype=float)

    rescued_mean = []
    rescued_std = []
    time_mean = []
    time_std = []

    for alpha in alpha_values:
        sub = [r for r in all_results if abs(r["alpha"] - alpha) < 1e-9]
        rescued_vals = np.array([r["rescued"] for r in sub], dtype=float)
        time_vals = np.array([r["final_time"] for r in sub], dtype=float)

        rescued_mean.append(rescued_vals.mean())
        rescued_std.append(rescued_vals.std(ddof=0))
        time_mean.append(time_vals.mean())
        time_std.append(time_vals.std(ddof=0))

    rescued_mean = np.array(rescued_mean)
    rescued_std = np.array(rescued_std)
    time_mean = np.array(time_mean)
    time_std = np.array(time_std)

    print("\n=== summary (daycare, alpha sweep) ===")
    print(
        "alpha\trescued_mean\trescued_std\tfinal_time_mean\t"
        "final_time_std"
    )
    for a, rm, rs, tm, ts in zip(
        alpha_arr, rescued_mean, rescued_std, time_mean, time_std
    ):
        print(f"{a:.2f}\t{rm:.2f}\t\t{rs:.2f}\t\t{tm:.1f}\t\t{ts:.1f}")

    # 找最优的 alpha（基于多个指标）
    print("\n=== 详细分析 ===")
    best_rescued_idx = np.argmax(rescued_mean)
    best_time_idx = np.argmin(time_mean)
    print(f"最多救援: alpha={alpha_arr[best_rescued_idx]:.2f}, "
          f"mean_rescued={rescued_mean[best_rescued_idx]:.2f}")
    print(f"最快清理: alpha={alpha_arr[best_time_idx]:.2f}, "
          f"mean_time={time_mean[best_time_idx]:.1f}")

    # 计算权衡指标（救援数 / 时间）
    rescue_per_time = rescued_mean / (time_mean + 1e-6)
    best_efficiency_idx = np.argmax(rescue_per_time)
    print(f"最高效率: alpha={alpha_arr[best_efficiency_idx]:.2f}, "
          f"rescued/time={rescue_per_time[best_efficiency_idx]:.4f}")

    # ========================================================
    # 3) 画图：rescued vs alpha
    # ========================================================

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        alpha_arr,
        rescued_mean,
        yerr=rescued_std,
        marker="o",
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    plt.xlabel(r"distance weight $\alpha$", fontsize=12)
    plt.ylabel("rescued (mean over seeds)", fontsize=12)
    plt.title("Babycare: Rescued People vs Distance Weight", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("daycare_alpha_rescued.png", dpi=200)
    print("\n✓ Saved: daycare_alpha_rescued.png")

    # ========================================================
    # 4) 画图：final_time vs alpha
    # ========================================================

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        alpha_arr,
        time_mean,
        yerr=time_std,
        marker="s",
        capsize=4,
        linewidth=2,
        markersize=6,
        color="orange",
    )
    plt.xlabel(r"distance weight $\alpha$", fontsize=12)
    plt.ylabel("all-clear time (timesteps, mean)", fontsize=12)
    plt.title("Babycare: Sweep Time vs Distance Weight", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("daycare_alpha_time.png", dpi=200)
    print("✓ Saved: daycare_alpha_time.png")

    # ========================================================
    # 5) 画 trade-off 图：rescued vs final_time
    # ========================================================

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        time_mean, rescued_mean, c=alpha_arr, cmap="viridis", s=100,
        alpha=0.7, edgecolors="black", linewidth=1
    )
    for a, x, y in zip(alpha_arr, time_mean, rescued_mean):
        plt.text(
            x, y, f"{a:.2f}", fontsize=7, ha="center", va="bottom"
        )
    cbar = plt.colorbar(scatter)
    cbar.set_label(r"$\alpha$", fontsize=12)
    plt.xlabel("all-clear time (mean, timesteps)", fontsize=12)
    plt.ylabel("rescued (mean)", fontsize=12)
    plt.title(
        "Babycare Trade-off: Time vs Rescued (colored by alpha)",
        fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("daycare_alpha_tradeoff.png", dpi=200)
    print("✓ Saved: daycare_alpha_tradeoff.png")

    # ========================================================
    # 6) 新增：Efficiency 图（rescued / time）
    # ========================================================

    plt.figure(figsize=(10, 6))
    plt.plot(
        alpha_arr, rescue_per_time, marker="D", linewidth=2, markersize=6,
        color="green"
    )
    plt.xlabel(r"distance weight $\alpha$", fontsize=12)
    plt.ylabel("Efficiency (rescued / time)", fontsize=12)
    plt.title(
        "Babycare: Rescue Efficiency vs Distance Weight", fontsize=14
    )
    plt.grid(True, alpha=0.3)
    best_idx = np.argmax(rescue_per_time)
    plt.plot(
        alpha_arr[best_idx], rescue_per_time[best_idx], "r*",
        markersize=20, label=f"Best: α={alpha_arr[best_idx]:.2f}"
    )
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("daycare_alpha_efficiency.png", dpi=200)
    print("✓ Saved: daycare_alpha_efficiency.png")

    # ========================================================
    # 7) 新增：Combined plot (子图)
    # ========================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: rescued
    axes[0, 0].errorbar(
        alpha_arr, rescued_mean, yerr=rescued_std, marker="o", capsize=4
    )
    axes[0, 0].set_xlabel(r"$\alpha$")
    axes[0, 0].set_ylabel("Rescued (mean)")
    axes[0, 0].set_title("Rescued People")
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: time
    axes[0, 1].errorbar(
        alpha_arr, time_mean, yerr=time_std, marker="s", capsize=4,
        color="orange"
    )
    axes[0, 1].set_xlabel(r"$\alpha$")
    axes[0, 1].set_ylabel("Time (mean)")
    axes[0, 1].set_title("Sweep Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: efficiency
    axes[1, 0].plot(alpha_arr, rescue_per_time, marker="D", color="green")
    axes[1, 0].set_xlabel(r"$\alpha$")
    axes[1, 0].set_ylabel("Efficiency")
    axes[1, 0].set_title("Rescue Efficiency")
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: trade-off scatter
    scatter = axes[1, 1].scatter(
        time_mean, rescued_mean, c=alpha_arr, cmap="viridis", s=50,
        alpha=0.7
    )
    axes[1, 1].set_xlabel("Time (mean)")
    axes[1, 1].set_ylabel("Rescued (mean)")
    axes[1, 1].set_title("Trade-off")
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label(r"$\alpha$")

    plt.tight_layout()
    plt.savefig("daycare_alpha_combined.png", dpi=200)
    print("✓ Saved: daycare_alpha_combined.png")


if __name__ == "__main__":
    sweep_alpha_on_daycare()
