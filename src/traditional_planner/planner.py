from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable

# Allow direct execution of this file (script mode) by ensuring the
# repository `src` directory is on sys.path so absolute imports succeed.
try:
    from traditional_planner.adapter import EnvAdapter
    from traditional_planner.graphutils import (
        bfs_distances,
        distance_to_fire,
        shortest_path_next_hop,
    )
    from traditional_planner.scoring import (
        PlannerConfig,
        compute_risk,
        congestion_proxy,
        compute_score,
    )
except Exception:
    import os, sys

    _this_dir = os.path.dirname(__file__)
    _src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from traditional_planner.adapter import EnvAdapter
    from traditional_planner.graphutils import (
        bfs_distances,
        distance_to_fire,
        shortest_path_next_hop,
    )
    from traditional_planner.scoring import (
        PlannerConfig,
        compute_risk,
        congestion_proxy,
        compute_score,
    )


ActionDict = Dict[int, Dict[str, Any]]  # aid -> {"action": str, "dest": str|None, "target": str|None, ...}


@dataclass
class GreedySweepPlanner:
    """
    Risk-aware greedy planner for multi-agent building sweeps.

    At each time step:
      1. Build a frontier of unswept rooms.
      2. For each agent, compute distance to all nodes (BFS).
      3. For each room, compute a mixed risk score.
      4. For each agent, build a preference list over rooms using
         Score(a, R) = alpha * d - beta * risk + gamma * congestion + epsilon.
      5. Resolve conflicts (multiple agents wanting the same room).
      6. Turn final assignments into move/search/wait actions.
    """

    adapter: EnvAdapter
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    # Per-agent target room id (or None)
    current_targets: Dict[int, Optional[str]] = field(default_factory=dict)

    # ========= Public API =========

    def plan_step(self, snapshot: Dict[str, Any]) -> ActionDict:
        """
        Given the current snapshot, compute one step of actions for all agents.

        This method is PURE with respect to the environment: it does NOT call
        adapter.move/search/step. It only decides what to do.
        The caller (e.g., a runner script) is responsible for applying the actions.

        Returns:
            actions: dict[aid] = {
                "action": "wait" | "move" | "search",
                "dest": Optional[str],   # for "move"
                "target": Optional[str], # assigned room for this agent
                "score": Optional[float] # score for assigned target (if any)
            }
        """
        nodes = snapshot["nodes"]
        agents = snapshot["agents"]

        # Build frontier: all unswept rooms
        frontier: List[str] = [
            nid
            for nid, info in nodes.items()
            if info["type"] == "room" and not info["swept"]
        ]

        actions: ActionDict = {}

        if not frontier:
            # Nothing left to sweep: everyone waits.
            for aid in agents.keys():
                actions[aid] = {
                    "action": "wait",
                    "dest": None,
                    "target": None,
                    "score": None,
                }
            # Also clear current targets
            self.current_targets = {aid: None for aid in agents.keys()}
            return actions

        # Precompute fire distances and risk for each frontier room
        dist_fire_norm = distance_to_fire(nodes, self.adapter.neighbors)

        room_risk: Dict[str, float] = {}
        room_congestion: Dict[str, float] = {}
        for room_id in frontier:
            room_risk[room_id] = compute_risk(
                room_id, nodes, dist_fire_norm, self.adapter.neighbors, self.cfg
            )
            room_congestion[room_id] = congestion_proxy(room_id, nodes)

        # For each agent: BFS distances, scores, and a preference list over rooms
        preferences: Dict[int, List[str]] = {}
        scores_per_agent: Dict[int, Dict[str, float]] = {}
        dists_per_agent: Dict[int, Dict[str, int]] = {}

        for aid, ainfo in agents.items():
            node = ainfo["node"]
            hp = ainfo["hp"]

            # Dead / invalid agents simply do nothing
            if node is None or hp <= 0:
                preferences[aid] = []
                scores_per_agent[aid] = {}
                dists_per_agent[aid] = {}
                continue

            # BFS distances from this agent's node
            dists = bfs_distances(node, self.adapter.neighbors)
            dists_per_agent[aid] = dists

            # Compute scores for all rooms in the frontier
            room_scores: Dict[str, float] = {}
            for room_id in frontier:
                dist_hops = dists.get(room_id, None)
                risk = room_risk[room_id]
                cong = room_congestion[room_id]
                score = compute_score(dist_hops, risk, cong, room_id, self.cfg)
                room_scores[room_id] = score

            scores_per_agent[aid] = room_scores

            # Sort rooms by increasing score (lower is better)
            sorted_rooms = sorted(frontier, key=lambda r: room_scores[r])

            # Apply target stickiness: if the previous target is still in frontier
            # and not much worse than the best, keep it at the front.
            prev_target = self.current_targets.get(aid)
            if prev_target is not None and prev_target in room_scores:
                prev_score = room_scores[prev_target]
                best_room = sorted_rooms[0]
                best_score = room_scores[best_room]
                if prev_score <= best_score + self.cfg.stickiness_delta:
                    # Move prev_target to the front if it's not already there
                    if sorted_rooms[0] != prev_target:
                        sorted_rooms = [prev_target] + [
                            r for r in sorted_rooms if r != prev_target
                        ]

            preferences[aid] = sorted_rooms

        # Resolve conflicts: multiple agents may want the same room as first choice.
        targets = self._resolve_conflicts(
            preferences, scores_per_agent, dists_per_agent
        )
        self.current_targets = targets

        # Turn final targets into concrete actions (move/search/wait)
        for aid, ainfo in agents.items():
            node = ainfo["node"]
            searching = ainfo["searching"]
            hp = ainfo["hp"]

            target_room = targets.get(aid)
            assigned_score = None
            if target_room is not None:
                assigned_score = scores_per_agent.get(aid, {}).get(target_room)

            if node is None or hp <= 0:
                actions[aid] = {
                    "action": "wait",
                    "dest": None,
                    "target": None,
                    "score": None,
                }
                continue

            # If agent is currently searching, we do not interrupt it.
            if searching:
                actions[aid] = {
                    "action": "wait",
                    "dest": None,
                    "target": target_room,
                    "score": assigned_score,
                }
                continue

            # No target assigned -> wait (next step we can replan again)
            if target_room is None:
                actions[aid] = {
                    "action": "wait",
                    "dest": None,
                    "target": None,
                    "score": None,
                }
                continue

            # If already at target room and it is not swept, start searching.
            node_info = nodes[node]
            if node == target_room and node_info["type"] == "room" and not node_info["swept"]:
                actions[aid] = {
                    "action": "search",
                    "dest": None,
                    "target": target_room,
                    "score": assigned_score,
                }
                continue

            # Otherwise, move one hop along a shortest path towards the target.
            next_hop = shortest_path_next_hop(
                start=node, goal=target_room, neighbors_fn=self.adapter.neighbors
            )

            if next_hop is None or next_hop == node:
                # Path is blocked or target is unreachable: wait this step.
                actions[aid] = {
                    "action": "wait",
                    "dest": None,
                    "target": target_room,
                    "score": assigned_score,
                }
            else:
                actions[aid] = {
                    "action": "move",
                    "dest": next_hop,
                    "target": target_room,
                    "score": assigned_score,
                }

        return actions

    # ========= Internal helpers =========

    def _resolve_conflicts(
        self,
        preferences: Dict[int, List[str]],
        scores_per_agent: Dict[int, Dict[str, float]],
        dists_per_agent: Dict[int, Dict[str, int]],
    ) -> Dict[int, Optional[str]]:
        """
        Resolve conflicts when multiple agents want the same room.

        Algorithm:
          - Each agent has a preference list (sorted by score).
          - In rounds, each remaining agent proposes to its current top choice.
          - If a room has multiple claimants, pick the one with:
              (1) smaller distance to that room, then
              (2) smaller agent id (tie-break).
          - Losers drop that room from their list and try again next round.
          - Repeat until no more assignments can be made.

        Returns:
            targets: dict[aid] -> assigned room id or None
        """
        remaining = set(preferences.keys())
        targets: Dict[int, Optional[str]] = {aid: None for aid in preferences.keys()}

        while remaining:
            room_to_claimants: Dict[str, List[int]] = {}
            # Build proposals: each agent picks its current top choice
            for aid in list(remaining):
                prefs = preferences[aid]
                # If no rooms left in the preference list, give up
                if not prefs:
                    remaining.remove(aid)
                    continue
                top_room = prefs[0]
                room_to_claimants.setdefault(top_room, []).append(aid)

            if not room_to_claimants:
                break

            progress = False

            for room_id, claimants in room_to_claimants.items():
                if len(claimants) == 1:
                    # Only one agent wants this room -> assign directly
                    aid = claimants[0]
                    targets[aid] = room_id
                    remaining.discard(aid)
                    progress = True
                else:
                    # Multiple agents want the same room.
                    # Choose the one with smallest graph distance, then by agent id.
                    def _dist_key(a: int) -> tuple[float, int]:
                        dmap = dists_per_agent.get(a, {})
                        d = dmap.get(room_id, self.cfg.unreachable_cost)
                        return (float(d), a)

                    best_aid = min(claimants, key=_dist_key)
                    targets[best_aid] = room_id
                    remaining.discard(best_aid)
                    progress = True

                    # Other claimants drop this room from their preference list.
                    for a in claimants:
                        if a == best_aid:
                            continue
                        prefs = preferences[a]
                        if prefs and prefs[0] == room_id:
                            prefs.pop(0)

            if not progress:
                # No new assignments were made in this round -> stop.
                break

        return targets
