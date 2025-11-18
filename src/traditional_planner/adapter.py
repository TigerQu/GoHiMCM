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

    There are two information modes:

      - "realistic": only uses observable civilian information
                     (obs_people_count, obs_avg_hp, dist_to_fire_norm, agent_here),
                     aligned with the RL observation space.

      - "oracle"   : uses full ground-truth civilian locations (env.people),
                     acting as a full-information upper-bound baseline.
    """

    def __init__(
        self,
        env: Optional[BuildingFireEnvironment] = None,
        info_mode: str = "realistic",
    ) -> None:
        """
        Initialize the adapter.

        Args:
            env: Optional pre-constructed environment. If None, a new
                 BuildingFireEnvironment is created with default config.
            info_mode: "realistic" (RL-aligned partial info) or "oracle"
                       (full ground-truth civilians).
        """
        self.env: BuildingFireEnvironment = env or BuildingFireEnvironment()
        if info_mode not in ("realistic", "oracle"):
            raise ValueError(f"Invalid info_mode: {info_mode}")
        self.info_mode: str = info_mode

    # ========= Mode control =========

    def set_mode(self, info_mode: str) -> None:
        """
        Change information mode at runtime.

        Args:
            info_mode: "realistic" or "oracle"
        """
        if info_mode not in ("realistic", "oracle"):
            raise ValueError(f"Invalid info_mode: {info_mode}")
        self.info_mode = info_mode

    # ========= Basic control API =========

    def reset(
        self,
        seed: Optional[int] = None,
        fire_node: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def _people_per_node_full(self) -> Dict[str, int]:
        """
        Count how many civilians are physically present at each node (ground truth).

        This uses env.people[pid].node_id and ignores observability.
        Only used in 'oracle' information mode.
        """
        counts: Dict[str, int] = {}
        people = getattr(self.env, "people", {})

        if isinstance(people, dict):
            iterable: Iterable[Any] = people.values()
        else:
            iterable = people

        for p in iterable:
            nid = getattr(p, "node_id", None)
            alive = bool(getattr(p, "is_alive", True))
            if nid is None or not alive:
                continue
            counts[nid] = counts.get(nid, 0) + 1

        return counts

    # ========= Snapshot API =========

    def snapshot(self, info_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a planner-friendly snapshot of the current environment state.

        There are two ways to choose information mode:
          - Use the adapter's default self.info_mode.
          - Override per call by passing info_mode="realistic" or "oracle".

        Modes:
          - realistic:
              people_count = node.obs_people_count
          - oracle:
              people_count = ground-truth count from env.people

        In both modes we also expose:
          - people_obs  : observable count (obs_people_count)
          - people_true : ground-truth count (for analysis / debugging)
          - avg_hp_obs  : average HP of observable civilians [0,1]
          - dist_fire   : normalized distance to nearest fire [0,1]
          - agent_here  : True if an agent is currently at this node

        Returns:
            {
              "time": int,
              "nodes": {
                  nid: {
                      "type": str,          # "room" | "hall" | "exit" | ...
                      "swept": bool,        # True if required_sweeps reached
                      "on_fire": bool,
                      "smoky": bool,
                      "intensity": float,   # fire_intensity proxy

                      # Civilian info:
                      "people_count": int,  # depends on info_mode
                      "people_obs": int,    # observable count
                      "people_true": int,   # ground-truth count (if available)
                      "avg_hp_obs": float,  # average HP of observable civilians [0,1]

                      # Fire-distance and agent presence:
                      "dist_fire": float,   # normalized distance to nearest fire [0,1]
                      "agent_here": bool,
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
        # Decide which mode to use for this snapshot
        mode = info_mode or self.info_mode
        if mode not in ("realistic", "oracle"):
            raise ValueError(f"Invalid info_mode: {mode}")

        # Make sure observable fields (obs_people_count, dist_to_fire_norm, etc.)
        # are up to date. We call get_observation() but ignore its return value.
        get_obs = getattr(self.env, "get_observation", None)
        if callable(get_obs):
            _ = get_obs()

        G = self.env.G
        nodes_meta = self.env.nodes        # Dict[str, NodeMeta]
        agents_meta = self.env.agents      # Dict[int, Agent]
        stats_env = getattr(self.env, "stats", {})

        # Ground-truth people per node (only used in oracle mode or for logging)
        people_true_map: Dict[str, int] = self._people_per_node_full()

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

            # Observable civilian info (partial observability)
            people_obs = int(getattr(node, "obs_people_count", 0))
            avg_hp_obs = float(getattr(node, "obs_avg_hp", 0.0))

            # Ground-truth civilians at this node (full info)
            people_true = int(people_true_map.get(nid, 0))

            # Fire distance and agent presence (fully observable)
            dist_fire = float(getattr(node, "dist_to_fire_norm", 1.0))
            agent_here = bool(getattr(node, "agent_here", False))

            # Decide which people_count the planner will see
            if mode == "oracle":
                people_count = people_true
            else:  # "realistic"
                people_count = people_obs

            nodes[nid] = {
                "type": ntype,
                "swept": bool(swept),
                "on_fire": bool(on_fire),
                "smoky": bool(smoky),
                "intensity": float(intensity),

                # civilans (mode-dependent + both views exposed)
                "people_count": int(people_count),
                "people_obs": people_obs,
                "people_true": people_true,
                "avg_hp_obs": avg_hp_obs,

                # fire distance + agent presence
                "dist_fire": dist_fire,
                "agent_here": agent_here,
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
