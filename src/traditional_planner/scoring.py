from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable

# Support running this file directly (script mode) by ensuring the
# `src` directory is on sys.path so relative imports resolve. When this
# module is imported as part of the package (preferred), the try/except
# does nothing.
try:
    from .graphutils import NeighborsFn
except Exception:
    import os, sys

    _this_dir = os.path.dirname(__file__)
    _src_dir = os.path.abspath(os.path.join(_this_dir, ".."))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    from traditional_planner.graphutils import NeighborsFn


@dataclass
class PlannerConfig:
    """
    Hyperparameters and weights for the greedy sweep planner.
    """

    # Distance-risk scoring weights
    alpha: float = 1.0   # travel distance penalty
    beta: float = 2.0    # risk "reward" (note the minus sign in score)
    gamma: float = 0.3   # congestion penalty

    # Risk feature weights
    w_intensity: float = 0.5
    w_smoke: float = 0.2
    w_dist_fire: float = 0.2   # used as (1 - dist_fire_norm)
    w_neighbor_fire: float = 0.1
    w_civilian_hint: float = 0.3

    # Target stickiness: how much better a new target must be
    # before we switch away from the previous target.
    stickiness_delta: float = 0.5

    # Large cost used when a room is unreachable from an agent.
    unreachable_cost: float = 1e6


def compute_risk(
    node_id: str,
    nodes: Dict[str, Dict[str, Any]],
    dist_fire_norm: Dict[str, float],
    neighbors_fn: NeighborsFn,
    cfg: PlannerConfig,
) -> float:
    """
    Compute a mixed risk score for a room node.

    Intuition:
      - Higher fire intensity -> higher risk
      - Smoky nodes -> higher risk
      - Closer to fire sources -> higher risk (1 - dist_fire_norm)
      - Neighbors with high intensity -> higher risk (threat of spread)
      - More civilians observed at this node -> higher risk

    This risk is then plugged into the score as:
        Score(a, R) = alpha * d(a,R) - beta * risk(R) + gamma * congestion(R) + epsilon
    so that higher risk => lower score => higher priority.
    """
    info = nodes[node_id]

    intensity = float(info.get("intensity", 0.0))
    smoke = 1.0 if info.get("smoky", False) else 0.0
    dist_norm = float(dist_fire_norm.get(node_id, 1.0))
    fire_proximity = 1.0 - dist_norm

    # Max neighbor intensity
    max_neighbor_intensity = 0.0
    for nb in neighbors_fn(node_id):
        nb_int = float(nodes[nb].get("intensity", 0.0))
        if nb_int > max_neighbor_intensity:
            max_neighbor_intensity = nb_int

    civilian_hint = float(info.get("people_count", 0.0))

    risk = (
        cfg.w_intensity * intensity +
        cfg.w_smoke * smoke +
        cfg.w_dist_fire * fire_proximity +
        cfg.w_neighbor_fire * max_neighbor_intensity +
        cfg.w_civilian_hint * civilian_hint
    )
    return risk


def congestion_proxy(node_id: str,
                     nodes: Dict[str, Dict[str, Any]]) -> float:
    """
    Simple congestion proxy.

    Currently uses the number of civilians at this node.
    Later, this could be extended to incorporate hallway density, etc.
    """
    return float(nodes[node_id].get("people_count", 0.0))


def compute_score(
    dist_hops: int | None,
    risk: float,
    congestion: float,
    room_id: str,
    cfg: PlannerConfig,
) -> float:
    """
    Compute the distance-risk-congestion score for assigning an agent to a room.

    We use:
        Score = alpha * d - beta * risk + gamma * congestion + epsilon

    where:
      - alpha * d      : travel cost (higher distance = worse)
      - -beta * risk   : "reward" for high risk (higher risk => lower score => better)
      - gamma * cong.  : small penalty for congestion
      - epsilon        : tiny deterministic noise for tie-breaking

    Args:
        dist_hops: shortest path distance in hops (None if unreachable)
        risk: risk(R)
        congestion: congestion(R)
        room_id: room id (only used for deterministic tie-breaking)
        cfg: PlannerConfig

    Returns:
        Scalar score (lower is better).
    """
    if dist_hops is None:
        # Treat unreachable rooms as extremely expensive.
        dist_val = cfg.unreachable_cost
    else:
        dist_val = float(dist_hops)

    eps = 1e-3 * (hash(room_id) % 1000)

    score = (
        cfg.alpha * dist_val
        - cfg.beta * risk
        + cfg.gamma * congestion
        + eps
    )
    return score
