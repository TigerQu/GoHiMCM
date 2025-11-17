from __future__ import annotations

"""
Step 3: end-to-end test for GreedySweepPlanner on the OFFICE layout.

This script:
  - Builds the standard office layout environment.
  - Wraps it with EnvAdapter.
  - Runs a single episode with the greedy planner.
  - Prints final stats and a few intermediate logs.

Usage (from GoHiMCM/src):

    python -m traditional_planner.test_planner
"""

from typing import Dict, Any

try:
    from environment.layouts import build_standard_office_layout
    from environment.layouts import build_babycare_layout
    from environment.layouts import build_two_floor_warehouse
    
except Exception:
    # When running this file directly, `src` may not be on sys.path.
    # Prepend the repo's src directory so the `environment` package can be imported.
    import os, sys

    _this_dir = os.path.dirname(__file__)
    _src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from environment.layouts import build_standard_office_layout
    from environment.layouts import build_babycare_layout
    from environment.layouts import build_two_floor_warehouse

from traditional_planner.adapter import EnvAdapter
from traditional_planner.planner import GreedySweepPlanner
from traditional_planner.scoring import PlannerConfig


def make_office_env():
    """
    Build the standard office layout env.

    The layout function:
      - creates nodes/edges (hallway + 3x3 rooms + exits, etc.)
      - spawns civilians in rooms
      - places agents at exits
    """
    env = build_standard_office_layout()
    return env

def make_babycare_env():
    env = build_babycare_layout()
    return env

def make_warehouse_env():
    env = build_two_floor_warehouse()
    return env


def run_greedy_episode(
    seed: int = 0,
    max_steps: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode of the greedy planner on the office layout.

    Args:
        seed: Random seed for env.reset
        max_steps: Safety cap on the number of time steps
        verbose: If True, print some logs during the run

    Returns:
        final_snapshot: snapshot dict after the run terminates
    """
    # 1) Build the OFFICE layout env and wrap it in an adapter
    env = make_warehouse_env()
    adapter = EnvAdapter(env=env)

    cfg = PlannerConfig()
    planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)

    snap = adapter.reset(seed=seed)
    initial_nodes = snap["nodes"]
    num_rooms = sum(1 for _, info in initial_nodes.items() if info["type"] == "room")
    # Total civilians in map (sum of people_count)
    total_people = sum(
        info.get("people_count", 0)
        for info in initial_nodes.values()
    )

    if verbose:
        print("=== GreedySweepPlanner demo ===")
        print(f"seed        : {seed}")
        print(f"#rooms      : {num_rooms}")
        print(f"#people     : {total_people}")
        print(f"max_steps   : {max_steps}")

    for t in range(max_steps):
        nodes = snap["nodes"]

        # Check frontier: unswept rooms
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

        # Verbose: show planned actions so we can trace routes
        if verbose:
            print(f"t={t:3d} planned actions: {actions}")

        # Apply actions to the environment
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

        if verbose and (t % 10 == 0 or t < 5):
            stats = snap["stats"]
            print(
                f"t={snap['time']:4d} | "
                f"swept={stats['nodes_swept']:3d} | "
                f"found={stats['people_found']:3d} | "
                f"rescued={stats['people_rescued']:3d}"
            )

    if verbose:
        stats = snap["stats"]
        print("\n=== Final stats ===")
        print(f"time_step      : {snap['time']}")
        print(f"nodes_swept    : {stats['nodes_swept']}")
        print(f"people_found   : {stats['people_found']}")
        print(f"people_rescued : {stats['people_rescued']}")
        # Also show the map's total people (ground-truth from nodes)
        final_total = sum(
            info.get("people_count", 0)
            for info in snap["nodes"].values()
        )
        print(f"total_people   : {final_total}")

    return snap


def main() -> None:
    # Run with verbose output so we see actions and final stats
    run_greedy_episode(seed=0, max_steps=500, verbose=True)


if __name__ == "__main__":
    main()
