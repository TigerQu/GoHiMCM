from __future__ import annotations

"""
End-to-end test for GreedySweepPlanner on three layouts:

  - Standard office
  - Babycare center (multi-floor, high-risk rooms)
  - Two-floor warehouse

For each layout, we run one episode with the greedy planner in
"realistic" information mode (aligned with RL observation space),
and then print:

  - Final time_step
  - Nodes swept
  - People rescued
  - HP statistics of rescued people (min / max / mean)
  - Final HP of each agent
"""

from typing import Callable, Dict, Any

# When running this file directly (e.g. python src/traditional_planner/
# planner_test.py) the repository's `src` directory may not be on sys.path
# which causes
# `from environment.layouts` (and other package imports) to fail with
# ModuleNotFoundError. Add a small fallback that prepends the repo's src
# directory to sys.path and retries the imports.
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
    # src/traditional_planner/.. -> src
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


def run_greedy_episode_on_layout(
    layout_name: str,
    build_env_fn: Callable[[], Any],
    info_mode: str = "realistic",
    seed: int = 0,
    max_steps: int = 600,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode of the greedy planner on a given layout.

    Args:
        layout_name: Name used for printing/logging.
        build_env_fn: Function that constructs and returns a new env instance.
        info_mode: "realistic" (RL-aligned) or "oracle" (full-info).
        seed: Random seed passed to env.reset.
        max_steps: Hard cap on the number of time steps.
        verbose: If True, print progress to stdout.

    Returns:
        final_snapshot: Snapshot dict after the episode terminates.
    """
    # 1) Build environment and wrap in adapter
    env = build_env_fn()
    adapter = EnvAdapter(env=env, info_mode=info_mode)

    cfg = PlannerConfig()
    planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)

    # 2) Reset environment
    snap = adapter.reset(seed=seed)
    nodes = snap["nodes"]
    num_rooms = sum(
        1 for _, info in nodes.items() if info["type"] == "room"
    )

    if verbose:
        print("=" * 70)
        print(f"Layout      : {layout_name}")
        print(f"Info mode   : {info_mode}")
        print(f"Seed        : {seed}")
        print(f"#rooms      : {num_rooms}")
        print(f"max_steps   : {max_steps}")
        print("-" * 70)

    # 3) Main simulation loop
    for t in range(max_steps):
        nodes = snap["nodes"]

        # Frontier = all unswept rooms
        frontier = [
            nid
            for nid, info in nodes.items()
            if info["type"] == "room" and not info["swept"]
        ]

        if not frontier:
            if verbose:
                print(f"\nAll rooms swept at step {t}. Terminating episode.")
            break

        # Plan actions for this step
        actions = planner.plan_step(snap)

        # Apply actions
        for aid, ainfo in actions.items():
            act_type = ainfo["action"]
            dest = ainfo.get("dest", None)

            if act_type == "move" and dest is not None:
                adapter.move(aid, dest)
            elif act_type == "search":
                adapter.search(aid)
            # "wait" does nothing

        # Advance environment dynamics
        adapter.step()
        snap = adapter.snapshot()

        if verbose and (t < 5 or t % 20 == 0):
            stats = snap["stats"]
            print(
                f"t={snap['time']:4d} | "
                f"swept={stats['nodes_swept']:3d} | "
                f"found={stats['people_found']:3d} | "
                f"rescued={stats['people_rescued']:3d}"
            )

    # 4) Final statistics
    stats = snap["stats"]

    # Collect rescued people HP from the underlying env
    rescued_hps = [
        person.hp
        for person in env.people.values()
        if getattr(person, "rescued", False)
    ]

    if rescued_hps:
        min_hp = min(rescued_hps)
        max_hp = max(rescued_hps)
        mean_hp = sum(rescued_hps) / len(rescued_hps)
    else:
        min_hp = max_hp = mean_hp = 0.0

    # Collect final agent HPs
    agent_hps = {aid: agent.hp for aid, agent in env.agents.items()}

    if verbose:
        print("\n--- Final stats ---")
        print(f"time_step        : {snap['time']}")
        print(f"nodes_swept      : {stats['nodes_swept']}")
        print(f"people_found     : {stats['people_found']}")
        print(f"people_rescued   : {stats['people_rescued']}")
        print(f"#rescued_people  : {len(rescued_hps)}")
        print(
            f"rescued_HP_stats : min={min_hp:.1f}, "
            f"max={max_hp:.1f}, mean={mean_hp:.1f}"
        )
        print("agent_HP         : ", end="")
        print(", ".join(
            f"agent{aid}={hp:.1f}"
            for aid, hp in sorted(agent_hps.items())
        ))
        print()

    # You can also return extra info if needed later (for logging/plots)
    result = dict(
        layout=layout_name,
        info_mode=info_mode,
        seed=seed,
        final_time=snap["time"],
        nodes_swept=stats["nodes_swept"],
        people_found=stats["people_found"],
        people_rescued=stats["people_rescued"],
        rescued_hps=rescued_hps,
        agent_hps=agent_hps,
    )
    return result


def main() -> None:
    # Define the layouts we want to test
    layouts = [
        ("office", build_standard_office_layout),
        ("babycare", build_babycare_layout),
        ("warehouse", build_two_floor_warehouse),
    ]

    for name, fn in layouts:
        # Use realistic mode to align with RL observation space
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
