# src/planner/graph_utils.py
from typing import List, Optional
import networkx as nx

def shortest_path_nodes(env, start: str, goal: str) -> Optional[List[str]]:
    """
    returns the list of nodes from start to goal (inclusive). None if unreachable.
    v0: uses unweighted shortest_path (min number of edges).
    """
    if start == goal:
        return [start]
    try:
        return nx.shortest_path(env.G, start, goal)  
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def next_hop(env, start: str, goal: str) -> Optional[str]:
    """
    gives the next neighbor from start towards goal; None if unreachable.
    """
    path = shortest_path_nodes(env, start, goal)
    if not path or len(path) < 2:
        return None
    # path: [start, next, ..., goal]
    return path[1]

def graph_distance_hops(env, a: str, b: str) -> Optional[int]:
    """
    returns the graph distance in number of hops between nodes a and b; None if unreachable.
    """
    path = shortest_path_nodes(env, a, b)
    if not path:
        return None
    return len(path) - 1
