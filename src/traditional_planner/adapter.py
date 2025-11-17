from __future__ import annotations

from typing import Dict, Any, Optional, Iterable

# Primary import used when the package root (the `src` directory) is
# available on sys.path. Some scripts are executed in ways where the
# script's directory becomes sys.path[0], which prevents the plain
# `environment` package import from resolving. Fall back to the
# alternate import which some scripts in this repo already use.
try:
    from environment.env import BuildingFireEnvironment
except ModuleNotFoundError:
    # Fallback when running this file directly and `src` isn't on PYTHONPATH.
    # Compute the path to the repository's `src` directory relative to
    # this file and prepend it to sys.path so `environment` can be found.
    import os
    import sys

    _this_dir = os.path.dirname(__file__)
    _src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    # Retry the import now that src/ has been added to sys.path.
    from environment.env import BuildingFireEnvironment


class EnvAdapter:
    """
    Thin wrapper around BuildingFireEnvironment.

    The traditional planner only talks to this adapter, not to the raw env.
    That keeps the planning code clean and decoupled from RL-specific APIs.
    """

    def __init__(self, env: Optional[BuildingFireEnvironment] = None) -> None:
        """
        Initialize the adapter.

        Args:
            env: Optional pre-constructed environment. If None, a new
                 BuildingFireEnvironment is created with default config.
        """
        self.env: BuildingFireEnvironment = env or BuildingFireEnvironment()

    # ========= Basic control API =========

    def reset(self, seed: Optional[int] = None,
              fire_node: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset the underlying environment and return the first snapshot.

        Env signature (from env.py):
            reset(fire_node: Optional[str] = None, seed: Optional[int] = None) -> Data

        Args:
            seed: Optional random seed passed through to env.reset
            fire_node: Optional node id to ignite (None = random, "none" = no fire)

        Returns:
            Snapshot dictionary (see snapshot() for format).
        """
        kwargs: Dict[str, Any] = {}
        if fire_node is not None:
            kwargs["fire_node"] = fire_node
        if seed is not None:
            kwargs["seed"] = seed

        self.env.reset(**kwargs)
        return self.snapshot()

    def neighbors(self, nid: str) -> list[str]:
        """
        Return the list of neighbor node ids of a given node.

        Used by graph utilities (BFS distances, shortest path, etc.).
        """
        return list(self.env.G.neighbors(nid))

    def move(self, aid: int, dest: str) -> bool:
        """
        Move an agent to an adjacent node.

        Args:
            aid: Agent id
            dest: Destination node id (must be adjacent in self.env.G)

        Returns:
            True if the move was accepted, False otherwise.
        """
        return self.env.move_agent(aid, dest)

    def search(self, aid: int) -> bool:
        """
        Start a search at the agent's current node.

        Args:
            aid: Agent id

        Returns:
            True if a search started, False otherwise.
        """
        return self.env.start_search(aid)

    def step(self) -> None:
        """
        Advance the environment by one time step.

        This triggers:
          - processing of ongoing searches
          - fire/smoke spread
          - civilian movement and health degradation
          - time_step increment
        """
        self.env.step()

    # ========= Helpers =========

    def _people_per_node(self) -> Dict[str, int]:
        """
        Count how many civilians are physically present at each node.

        Uses the ground-truth env.people dict (Person.node_id).
        """
        counts: Dict[str, int] = {}
        people = getattr(self.env, "people", {})

        if isinstance(people, dict):
            iterable: Iterable[Any] = people.values()
        else:
            iterable = people

        for p in iterable:
            nid = getattr(p, "node_id", None)
            if nid is None:
                continue
            counts[nid] = counts.get(nid, 0) + 1

        return counts

    # ========= Snapshot API =========

    def snapshot(self) -> Dict[str, Any]:
        """
        Build a planner-friendly snapshot of the current environment state.

        Returns:
            {
              "time": int,
              "nodes": {
                  nid: {
                      "type": str,          # "room" | "hall" | "exit"
                      "swept": bool,        # True if required_sweeps reached
                      "on_fire": bool,
                      "smoky": bool,
                      "intensity": float,   # fire_intensity proxy
                      "people_count": int,  # ground-truth number of people
                  }
              },
              "agents": {
                  aid: {
                      "node": str | None,   # current node id
                      "searching": bool,
                      "search_timer": int,
                      "hp": float,
                      "exposure": float,
                  }
              },
              "graph": {
                  "N": int,                # number of nodes
                  "E": int,                # number of edges
              },
              "stats": {
                  "nodes_swept": int,
                  "people_found": int,
                  "people_rescued": int,
              },
            }
        """
        G = self.env.G
        nodes_meta = self.env.nodes        # Dict[str, NodeMeta]
        agents_meta = self.env.agents      # Dict[int, Agent]
        stats_env = getattr(self.env, "stats", {})

        people_per_node = self._people_per_node()

        # ----- nodes -----
        nodes: Dict[str, Dict[str, Any]] = {}
        for nid, node in nodes_meta.items():
            ntype = getattr(node, "ntype", "room")

            sweep_count = getattr(node, "sweep_count", 0)
            required_sweeps = getattr(node, "required_sweeps", 1)
            swept = sweep_count >= required_sweeps

            on_fire = getattr(node, "on_fire", False)
            smoky = getattr(node, "smoky", False)
            intensity = getattr(node, "fire_intensity", 1.0 if on_fire else 0.0)

            nodes[nid] = {
                "type": ntype,
                "swept": bool(swept),
                "on_fire": bool(on_fire),
                "smoky": bool(smoky),
                "intensity": float(intensity),
                "people_count": int(people_per_node.get(nid, 0)),
            }

        # ----- agents -----
        agents: Dict[int, Dict[str, Any]] = {}
        for aid, agent in agents_meta.items():
            agents[aid] = {
                "node": getattr(agent, "node_id", None),
                "searching": bool(getattr(agent, "searching", False)),
                "search_timer": int(getattr(agent, "search_timer", 0)),
                "hp": float(getattr(agent, "hp", 100.0)),
                "exposure": float(getattr(agent, "exposure", 0.0)),
            }

        # ----- stats -----
        nodes_swept = int(stats_env.get("nodes_swept", 0))
        people_found = int(stats_env.get("people_found", 0))
        people_rescued = int(stats_env.get("people_rescued", 0))

        stats = {
            "nodes_swept": nodes_swept,
            "people_found": people_found,
            "people_rescued": people_rescued,
        }

        graph_meta = {
            "N": G.number_of_nodes(),
            "E": G.number_of_edges(),
        }

        time_step = int(getattr(self.env, "time_step", 0))

        return {
            "time": time_step,
            "nodes": nodes,
            "agents": agents,
            "graph": graph_meta,
            "stats": stats,
        }
