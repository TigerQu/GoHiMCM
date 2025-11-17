from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Any

# Type alias: a function that returns neighbors for a given node id.
NeighborsFn = Callable[[str], Iterable[str]]


def bfs_distances(start: Optional[str],
                  neighbors_fn: NeighborsFn) -> Dict[str, int]:
    """
    Unweighted BFS distances from a single start node.

    Args:
        start: Starting node id. If None, returns an empty dict.
        neighbors_fn: Function that returns neighbors of a node id.

    Returns:
        dist: dict mapping node_id -> distance in hops
    """
    if start is None:
        return {}

    dist: Dict[str, int] = {start: 0}
    queue: deque[str] = deque([start])

    while queue:
        u = queue.popleft()
        for v in neighbors_fn(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist


def shortest_path_next_hop(start: Optional[str],
                           goal: Optional[str],
                           neighbors_fn: NeighborsFn) -> Optional[str]:
    """
    Return the next node on a shortest path from start to goal using BFS.

    Args:
        start: Start node id
        goal: Goal node id
        neighbors_fn: Function that returns neighbors for BFS

    Returns:
        - If start == goal, returns goal.
        - If there is a path, returns the first hop after start.
        - If no path exists or start/goal is None, returns None.
    """
    if start is None or goal is None:
        return None
    if start == goal:
        return goal

    parent: Dict[str, Optional[str]] = {start: None}
    queue: deque[str] = deque([start])
    found = False

    while queue and not found:
        u = queue.popleft()
        for v in neighbors_fn(u):
            if v not in parent:
                parent[v] = u
                if v == goal:
                    found = True
                    break
                queue.append(v)

    if goal not in parent:
        # No path found
        return None

    # Backtrack from goal to find the first hop after start
    cur = goal
    prev = parent[cur]
    while prev is not None and prev != start:
        cur = prev
        prev = parent[cur]

    if prev is None:
        # Somehow goal is disconnected from start
        return None
    return cur


def distance_to_fire(nodes: Dict[str, Dict[str, Any]],
                     neighbors_fn: NeighborsFn) -> Dict[str, float]:
    """
    Compute normalized graph distance from each node to the nearest fire node.

    Multi-source BFS: treats all nodes with on_fire=True as distance 0,
    then grows outward.

    Args:
        nodes: snapshot["nodes"] dict (node_id -> info dict)
        neighbors_fn: function to get neighbors

    Returns:
        dist_norm: dict mapping node_id -> normalized distance in [0, 1],
                   where 0 means "on fire", and 1 means "very far / no fire".
    """
    # Collect all fire sources
    fire_sources: List[str] = [
        nid for nid, info in nodes.items() if info.get("on_fire", False)
    ]

    # If there is no fire, treat all nodes as distance 1.0 (low fire risk).
    if not fire_sources:
        return {nid: 1.0 for nid in nodes.keys()}

    # Initialize distances with a large sentinel value
    INF = 10**9
    dist: Dict[str, int] = {nid: INF for nid in nodes.keys()}
    queue: deque[str] = deque()

    for src in fire_sources:
        dist[src] = 0
        queue.append(src)

    # Multi-source BFS
    while queue:
        u = queue.popleft()
        for v in neighbors_fn(u):
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                queue.append(v)

    # Normalize finite distances to [0, 1]
    finite_dists = [d for d in dist.values() if d < INF]
    max_d = max(finite_dists) if finite_dists else 1

    dist_norm: Dict[str, float] = {}
    for nid, d in dist.items():
        if d >= INF:
            dist_norm[nid] = 1.0
        else:
            dist_norm[nid] = d / max_d if max_d > 0 else 0.0

    return dist_norm
