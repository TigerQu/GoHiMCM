from __future__ import annotations

"""
Step 1: smoke tests for EnvAdapter.

This file checks that:
  - The adapter can construct and reset the environment.
  - snapshot() returns the expected high-level structure.
  - step() advances time and updates stats.

You can run it either as a module (recommended):

    cd GoHiMCM/src
    python -m traditional_planner.test

or directly as a script (the adapter has a fallback import path):

    cd GoHiMCM/src
    python traditional_planner/test.py
"""

from typing import Any, Dict

# When running this file directly (python traditional_planner/test.py),
# Python's import search path may not include the repository `src` folder,
# which prevents importing the `traditional_planner` package. Ensure the
# repo `src` directory is on sys.path so the package import works either
# when run as a module or as a script.
import os
import sys

_this_dir = os.path.dirname(__file__)
_src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from traditional_planner.adapter import EnvAdapter


def _build_minimal_layout(adapter: EnvAdapter) -> None:
    """
    Create a tiny layout so the environment has nodes/edges/people/agent
    before calling reset(). We populate _initial_people_state and
    _initial_agent_positions by using spawn_person/place_agent so reset()
    will restore them correctly.
    """
    env = adapter.env
    # Add two nodes and a connecting edge
    env.add_node("room1", "room", area=10.0, length=5.0)
    env.add_node("exit1", "exit", area=5.0, length=3.0)
    env.add_edge("room1", "exit1", width=1.0, length=3.0, slope=0.0, door=True, fire_door=False)

    # Spawn one civilian in the room and place one agent at the exit
    env.spawn_person("room1", age=30, mobility="adult", hp=100.0)
    env.place_agent(0, "exit1")


def smoke_test_reset_and_snapshot() -> None:
    """
    Basic sanity check:

    - Create an EnvAdapter
    - Call reset()
    - Inspect the returned snapshot structure
    """
    adapter = EnvAdapter()
    # Build a minimal layout so to_pytorch_geometric has nodes
    _build_minimal_layout(adapter)
    snap = adapter.reset(seed=0)

    # Very lightweight structural checks (asserts will raise if something is wrong)
    assert isinstance(snap, dict), "snapshot() must return a dict"
    for key in ["time", "nodes", "agents", "graph", "stats"]:
        assert key in snap, f"snapshot is missing key: {key}"

    assert isinstance(snap["nodes"], dict), "nodes must be a dict"
    assert isinstance(snap["agents"], dict), "agents must be a dict"
    assert isinstance(snap["graph"], dict), "graph must be a dict"
    assert isinstance(snap["stats"], dict), "stats must be a dict"

    print("=== Snapshot after reset() ===")
    print(f"time_step      : {snap['time']}")
    print(f"#nodes         : {snap['graph'].get('N')}")
    print(f"#edges         : {snap['graph'].get('E')}")
    print(f"nodes_swept    : {snap['stats'].get('nodes_swept')}")
    print(f"people_found   : {snap['stats'].get('people_found')}")
    print(f"people_rescued : {snap['stats'].get('people_rescued')}")

    # Show a few nodes
    print("\nSample nodes:")
    for i, (nid, info) in enumerate(snap["nodes"].items()):
        print(
            f"  {nid}: "
            f"type={info['type']}, "
            f"swept={info['swept']}, "
            f"on_fire={info['on_fire']}, "
            f"smoky={info['smoky']}, "
            f"intensity={info['intensity']:.3f}, "
            f"people={info['people_count']}"
        )
        if i >= 4:  # only print first 5 nodes
            break

    # Show agents
    print("\nAgents:")
    if not snap["agents"]:
        print("  (no agents found)")
    for aid, ainfo in snap["agents"].items():
        print(
            f"  agent {aid}: "
            f"node={ainfo['node']}, "
            f"searching={ainfo['searching']}, "
            f"search_timer={ainfo['search_timer']}, "
            f"hp={ainfo['hp']}, "
            f"exposure={ainfo['exposure']}"
        )


def smoke_test_step_progression(steps: int = 5) -> None:
    """
    Second sanity check:

    - Reset the environment
    - Call step() a few times
    - Verify that time and stats change over time
    """
    adapter = EnvAdapter()
    _build_minimal_layout(adapter)
    snap0 = adapter.reset(seed=1)

    t0 = snap0["time"]
    stats0 = snap0["stats"]

    for _ in range(steps):
        adapter.step()

    snap1 = adapter.snapshot()
    t1 = snap1["time"]
    stats1 = snap1["stats"]

    print(f"\n=== After {steps} steps ===")
    print(f"time_step      : {t0} -> {t1}")
    print(
        "nodes_swept    : "
        f"{stats0.get('nodes_swept')} -> {stats1.get('nodes_swept')}"
    )
    print(
        "people_found   : "
        f"{stats0.get('people_found')} -> {stats1.get('people_found')}"
    )
    print(
        "people_rescued : "
        f"{stats0.get('people_rescued')} -> {stats1.get('people_rescued')}"
    )


def main() -> None:
    """
    Entry point for manual smoke testing.

    In later steps, this file will grow into a full experiment runner
    (demo runs, grid search, plotting). For now, we only care that the
    adapter works and snapshots look sane.
    """
    print("Running EnvAdapter smoke tests...\n")
    smoke_test_reset_and_snapshot()
    smoke_test_step_progression(steps=5)
    print("\nAll smoke tests finished (no assertion failures).")


if __name__ == "__main__":
    main()
