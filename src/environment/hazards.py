import random
from typing import Dict
import networkx as nx
from .entities import NodeMeta, EdgeMeta, Person

# fire and smoke set up

def ignite_node(nodes: Dict[str, NodeMeta], node_id: str) -> None:
    """
    Start a fire at the specified node.
    
    Args:
        node_id: Node to ignite
    """
    if node_id not in nodes:
        raise ValueError(f"Node {node_id} does not exist")
    nodes[node_id].on_fire = True
    nodes[node_id].smoky = True  # Fire implies smoke


def spread_hazards(
    G: nx.Graph,
    nodes: Dict[str, NodeMeta],
    fire_spread_prob: float,
    fire_spread_counter: int,
    fire_spread_delay_cfg: int
) -> int:
    """
    Spread fire and smoke to adjacent nodes.

    Logic:
    1. Fire spreads probabilistically to neighbors (slower through fire doors)
    2. Smoke spreads more easily than fire
    3. Fire always produces smoke

    Returns:
        Updated counter (0 if actual spread happened, otherwise incremented)
    """
    # Only spread fire periodically (not every step)
    fire_spread_counter += 1
    if fire_spread_counter < fire_spread_delay_cfg:
        return fire_spread_counter
    fire_spread_counter = 0

    # Identify currently burning nodes
    burning_nodes = [n for n in nodes.values() if n.on_fire]

    # Spread from each burning node
    for burning_node in burning_nodes:
        for neighbor_id in G.neighbors(burning_node.nid):
            neighbor = nodes[neighbor_id]
            edge_data = G.edges[burning_node.nid, neighbor_id]
            edge_meta: EdgeMeta = edge_data["meta"]
            
            # Fire doors reduce spread probability
            spread_prob = fire_spread_prob
            if edge_meta.fire_door:
                spread_prob *= 0.3  # Fire doors are 70% more effective
            # Attempt to spread fire
            if not neighbor.on_fire and random.random() < spread_prob:
                neighbor.on_fire = True
            
            # Smoke spreads more easily (always to adjacent nodes of fire)
            neighbor.smoky = True
    return fire_spread_counter


def degrade_health(
    nodes: Dict[str, NodeMeta],
    people: Dict[int, Person],
    hp_loss_fire: float,
    hp_loss_smoke: float,
    time_step: int,
    verbose: bool = False
) -> None:
    """
    Reduce HP of people in hazardous conditions.

    Logic:
    - People in fire lose HP faster than people in smoke
    - HP cannot go below 0
    - Dead people (HP=0) are tracked but no longer degrade
    - Skip people with invalid node_id (robustness guard)
    """
    for person in people.values():
        if not person.is_alive:
            continue  # Skip dead people
        
        # Guard: check that node_id is valid
        if person.node_id not in nodes:
            if verbose:
                print(f"[T={time_step}] Warning: Person {person.pid} has invalid node_id: {person.node_id}")
            continue

        node = nodes[person.node_id]
        hp_loss = 0.0
        if node.on_fire:
            hp_loss = hp_loss_fire
        elif node.smoky:
            hp_loss = hp_loss_smoke

        if hp_loss > 0:
            person.hp = max(0.0, person.hp - hp_loss)
            if verbose and not person.is_alive:
                print(f"[T={time_step}] Person {person.pid} died at {person.node_id}")
