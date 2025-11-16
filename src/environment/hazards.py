import random
from typing import Dict
import networkx as nx
from .entities import NodeMeta, EdgeMeta, Person

# fire and smoke set up

# Add intensity parameters to ignite_node 
def ignite_node(nodes: Dict[str, NodeMeta], node_id: str,
                fire_intensity_init: float = 0.4,
                smoke_density_base: float = 0.3) -> None:
    """
    Start a fire at the specified node with initial intensity.
    
    Args:
        nodes: Dictionary of node metadata
        node_id: Node to ignite
        fire_intensity_init: Initial fire intensity (0-1)
        smoke_density_base: Initial smoke density (0-1)
    """
    if node_id not in nodes:
        raise ValueError(f"Node {node_id} does not exist")
    
    node = nodes[node_id]
    node.on_fire = True
    node.smoky = True
    
    # Set initial intensities (take max to avoid reducing existing intensity)
    node.fire_intensity = max(node.fire_intensity, fire_intensity_init)
    node.smoke_density = max(node.smoke_density, smoke_density_base)

# Add intensity parameters and growth logic to spread_hazards
def spread_hazards(
    G: nx.Graph,
    nodes: Dict[str, NodeMeta],
    fire_spread_prob: float,
    fire_spread_counter: int,
    fire_spread_delay_cfg: int,
    fire_intensity_init: float = 0.4,
    fire_intensity_growth: float = 0.1,
    smoke_density_base: float = 0.3,
    smoke_density_growth: float = 0.1
) -> int:
    """
    Spread fire and smoke to adjacent nodes with intensity modeling.

    Logic:
    1. Fire intensity grows over time in burning nodes
    2. Fire spreads probabilistically to neighbors (slower through fire doors)
    3. Smoke spreads more easily than fire
    4. Both fire and smoke have continuous intensity values

    Args:
        G: Building graph
        nodes: Dictionary of node metadata
        fire_spread_prob: Base probability of fire spreading
        fire_spread_counter: Current counter value
        fire_spread_delay_cfg: Steps between spread attempts
        fire_intensity_init: Initial intensity when fire spreads to new node
        fire_intensity_growth: Intensity growth per spread cycle
        smoke_density_base: Base smoke density when smoke appears
        smoke_density_growth: Smoke density increase per cycle

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
        # NEW: Grow intensity in-place (fire gets more intense over time)
        burning_node.fire_intensity = min(1.0, 
            burning_node.fire_intensity + fire_intensity_growth)
        
        # Ensure boolean flag matches intensity state
        burning_node.on_fire = burning_node.fire_intensity > 0.0
        
        for neighbor_id in G.neighbors(burning_node.nid):
            neighbor = nodes[neighbor_id]
            edge_data = G.edges[burning_node.nid, neighbor_id]
            edge_meta: EdgeMeta = edge_data["meta"]
            
            # Fire doors reduce spread probability
            spread_prob = fire_spread_prob
            if edge_meta.fire_door:
                spread_prob *= 0.3  # Fire doors are 70% more effective
            
            # Attempt to spread fire with intensity
            if not neighbor.on_fire and random.random() < spread_prob:
                neighbor.on_fire = True
                # NEW: Set initial intensity when fire spreads to new node
                neighbor.fire_intensity = max(
                    neighbor.fire_intensity,
                    fire_intensity_init
                )
            
            # Smoke spreads more easily (always to adjacent nodes of fire)
            neighbor.smoky = True
            # NEW: Grow smoke density over time
            neighbor.smoke_density = min(
                1.0,
                max(neighbor.smoke_density, smoke_density_base) + smoke_density_growth
            )
    
    return fire_spread_counter


def degrade_health(
    nodes: Dict[str, NodeMeta],
    people: Dict[int, Person],
    hp_loss_fire: float,
    hp_loss_smoke: float,
    hp_mult_child: float,
    hp_mult_adult: float,
    hp_mult_limited: float,
    time_step: int,
    verbose: bool = False
) -> None:
    """
    Reduce HP of people in hazardous conditions with mobility-specific vulnerability.

    Logic:
    - HP loss scales with fire intensity and smoke density (not just 0/1)
    - Children and limited mobility people are more vulnerable (higher multipliers)
    - People in fire lose HP faster than people in smoke
    - HP cannot go below 0
    - Dead people (HP=0) are tracked but no longer degrade
    
    Args:
        nodes: Dictionary of node metadata
        people: Dictionary of person objects
        hp_loss_fire: Base HP loss per timestep in fire
        hp_loss_smoke: Base HP loss per timestep in smoke
        hp_mult_child: Vulnerability multiplier for children
        hp_mult_adult: Vulnerability multiplier for adults (baseline)
        hp_mult_limited: Vulnerability multiplier for limited mobility
        time_step: Current simulation timestep
        verbose: If True, print death messages
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
        
        # NEW: Determine vulnerability multiplier based on mobility
        mult = hp_mult_adult  # Default
        if person.mobility == "child":
            mult = hp_mult_child
        elif person.mobility == "limited":
            mult = hp_mult_limited
        
        # NEW: Calculate HP loss using intensity and multiplier
        hp_loss = 0.0
        if node.on_fire:
            # Use fire intensity (defaults to 1.0 if field doesn't exist)
            intensity = getattr(node, "fire_intensity", 1.0)
            hp_loss = hp_loss_fire * intensity * mult
        elif node.smoky:
            # Use smoke density (defaults to 1.0 if field doesn't exist)
            density = getattr(node, "smoke_density", 1.0)
            hp_loss = hp_loss_smoke * density * mult

        if hp_loss > 0:
            person.hp = max(0.0, person.hp - hp_loss)
            if verbose and not person.is_alive:
                print(f"[T={time_step}] Person {person.pid} (mobility={person.mobility}) died at {person.node_id}")
