"""
End-to-end planner test driver for GreedySweepPlanner.

This script runs the greedy planner on three built-in layouts and prints
per-step and final statistics. It also saves a sweep visualization PNG for
each layout using the local `plot_sweep` helper.
"""

from __future__ import annotations

from typing import Callable, Dict, Any

try:
    from environment.layouts import (
        build_standard_office_layout,
        build_babycare_layout,
        build_two_floor_warehouse,
    )
except Exception:
    # Fallback when running file directly and src/ isn't on sys.path.
    import os, sys

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

# Use the plotting helper in this package (if present)
try:
    from traditional_planner.plot_sweep import plot_sweep_with_risk
except Exception:
    plot_sweep_with_risk = None


def run_greedy_episode_on_layout(
    layout_name: str,
    build_env_fn: Callable[[], Any],
    info_mode: str = "realistic",
    seed: int = 0,
    max_steps: int = 600,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = build_env_fn()
    adapter = EnvAdapter(env=env, info_mode=info_mode)

    cfg = PlannerConfig()
    planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)

    snap = adapter.reset(seed=seed)

    # logging for visualization
    agent_paths: Dict[int, list[str]] = {aid: [ainfo.get("node")] for aid, ainfo in snap["agents"].items()}
    node_clear_time: Dict[str, float] = {}

    if verbose:
        print("=" * 70)
        print(f"Layout      : {layout_name}")
        print(f"Info mode   : {info_mode}")
        print(f"Seed        : {seed}")
        num_rooms = sum(1 for _, info in snap["nodes"].items() if info.get("type") == "room")
        print(f"#rooms      : {num_rooms}")
        print(f"max_steps   : {max_steps}")
        print("-" * 70)

    for t in range(max_steps):
        nodes = snap["nodes"]
        frontier = [nid for nid, info in nodes.items() if info.get("type") == "room" and not info.get("swept")]

        if not frontier:
            if verbose:
                print(f"\nAll rooms swept at step {t}. Terminating episode.")
            break

        actions = planner.plan_step(snap)

        if verbose:
            print(f"t={t:3d} planned actions: {actions}")

        for aid, ainfo in actions.items():
            act_type = ainfo["action"]
            dest = ainfo.get("dest")
            if act_type == "move" and dest is not None:
                adapter.move(aid, dest)
            elif act_type == "search":
                adapter.search(aid)

        adapter.step()
        snap = adapter.snapshot()

        # update agent paths
        for aid, ainfo in snap["agents"].items():
            node_id = ainfo.get("node")
            agent_paths.setdefault(aid, []).append(node_id)

        # update clear times
        current_time = snap["time"]
        for nid, info in snap["nodes"].items():
            if info.get("type") == "room" and info.get("swept"):
                node_clear_time.setdefault(nid, current_time)

        if verbose and (t < 5 or t % 20 == 0):
            stats = snap["stats"]
            print(
                f"t={snap['time']:4d} | "
                f"swept={stats['nodes_swept']:3d} | "
                f"found={stats['people_found']:3d} | "
                f"rescued={stats['people_rescued']:3d}"
            )

    stats = snap["stats"]

    rescued_hps = [p.hp for p in env.people.values() if getattr(p, "rescued", False)]
    if rescued_hps:
        min_hp = min(rescued_hps)
        max_hp = max(rescued_hps)
        mean_hp = sum(rescued_hps) / len(rescued_hps)
    else:
        min_hp = max_hp = mean_hp = 0.0

    agent_hps = {aid: a.hp for aid, a in env.agents.items()}

    if verbose:
        print("\n--- Final stats ---")
        print(f"time_step        : {snap['time']}")
        print(f"nodes_swept      : {stats['nodes_swept']}")
        print(f"people_found     : {stats['people_found']}")
        print(f"people_rescued   : {stats['people_rescued']}")
        print(f"#rescued_people  : {len(rescued_hps)}")
        print(
            f"rescued_HP_stats : min={min_hp:.1f}, max={max_hp:.1f}, mean={mean_hp:.1f}"
        )
        print("agent_HP         : ", end="")
        print(", ".join(f"agent{aid}={hp:.1f}" for aid, hp in sorted(agent_hps.items())))
        print()

    # visualization
    try:
        if plot_sweep_with_risk is not None:
            edges = list(env.G.edges())
            # Build rich title with episode stats
            num_swept = stats["nodes_swept"]
            num_people_found = stats["people_found"]
            num_people_rescued = stats["people_rescued"]
            total_time = snap["time"]
            total_people = len(env.people)
            title = (f"{layout_name.upper()} Sweep Trajectory\n"
                     f"Time: {total_time} | Swept: {num_swept} | "
                     f"Found: {num_people_found} | "
                     f"Rescued: {num_people_rescued}/{total_people}")
            save_path = f"{layout_name}_greedy_sweep.png"
            plot_sweep_with_risk(
                node_positions=None,
                edges=edges,
                agent_paths=agent_paths,
                node_clear_time=node_clear_time,
                risk_at_time=None,
                title=title,
                save_path=save_path,
                env=env,
                show_people=True,
            )
            if verbose:
                print(f"[INFO] Saved sweep visualization to {save_path}")
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to create sweep visualization: {e}")

    return dict(
        layout=layout_name,
        final_time=snap["time"],
        nodes_swept=stats["nodes_swept"],
        people_found=stats["people_found"],
        people_rescued=stats["people_rescued"],
    )


def main() -> None:
    layouts = [
        ("office", build_standard_office_layout),
        ("babycare", build_babycare_layout),
        ("warehouse", build_two_floor_warehouse),
    ]

    for name, fn in layouts:
        run_greedy_episode_on_layout(
            layout_name=name,
            build_env_fn=fn,
            info_mode="realistic",
            seed=0,
            max_steps=600,
            verbose=True,
        )


if __name__ == "__main__":
    main()
