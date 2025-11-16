import math
import heapq
from typing import Optional, TYPE_CHECKING
from .entities import Person, EdgeMeta
import networkx as nx

if TYPE_CHECKING:
    from .env import BuildingFireEnvironment


def move_civilians(env: 'BuildingFireEnvironment') -> None:
    """
    Move civilians toward exits using dynamic pathfinding.
    
    REVISED LOGIC:
    1. People don't move until they become aware (awareness_timer expires)
    2. Once aware, they move cautiously (panicked = True, slower speed)
    3. When assisted by an agent, they move faster and more confidently
    4. Assisted people are escorted together with the agent
    """
    # Progress people currently on edges
    for person in env.people.values():
        if not person.is_alive or person.rescued:
            continue
        
        if person.on_edge:
            person.edge_eta -= 1.0
            
            if person.edge_eta <= 0:
                # Arrival at destination node
                env._edge_dec(person.edge_u, person.edge_v)
                
                # Remove from old node's people list
                if person.node_id in env.nodes:
                    old_people = env.nodes[person.node_id].people
                    if person.pid in old_people:
                        old_people.remove(person.pid)
                
                # Update position
                person.on_edge = False
                person.node_id = person.edge_v
                person.edge_u = None
                person.edge_v = None
                person.edge_eta = 0.0
                
                # Add to new node's people list
                env.nodes[person.node_id].people.append(person.pid)
                
                # Check if reached exit
                if env.nodes[person.node_id].ntype == 'exit':
                    if not person.rescued:
                        person.rescued = True
                        env.stats['people_rescued'] += 1
                        # Release agent if being assisted
                        if person.being_assisted and person.assisting_agent_id is not None:
                            agent = env.agents.get(person.assisting_agent_id)
                            if agent:
                                agent.assisting_person_id = None
                            person.being_assisted = False
                            person.assisting_agent_id = None
    
    # Start movement for idle, aware people
    for person in env.people.values():
        if not person.is_alive or person.rescued or person.on_edge:
            continue
        
        # Awareness delay - people don't immediately react
        if person.awareness_timer > 0:
            person.awareness_timer -= 1
            continue
        
        # People only move if they're aware (seen by agent or alarm triggered)
        # In this model, we assume people only move after being discovered
        if not person.seen:
            continue
        
        #Check tenability - people below threshold cannot self-evacuate =====
        # This models physiological incapacitation from smoke inhalation, heat stress, etc.
        # Children and limited mobility people have HIGHER thresholds (become incapacitated earlier):
        #   - Child threshold: 60 HP (more vulnerable respiratory/cardiovascular systems)
        #   - Limited threshold: 50 HP (pre-existing health conditions)
        #   - Adult threshold: 40 HP (baseline healthy adult)
        # Below threshold, person needs assistance from responder to evacuate
        if not person.is_tenable:
            continue  # Person is incapacitated, cannot move without assistance

        # Update panic state - people are panicked if they see hazards but aren't assisted
        current_node = env.nodes[person.node_id]
        if (current_node.on_fire or current_node.smoky) and not person.being_assisted:
            person.panicked = True
        
        # If being assisted, follow agent's movement (handled in agent movement)
        if person.being_assisted:
            continue
        
        # Get next node on path to exit
        next_node = _person_route_next(env, person)
        
        if next_node is None:
            continue
        
        # Check if blocked by fire (people won't enter fire)
        next_node_meta = env.nodes[next_node]
        if next_node_meta.on_fire:
            continue  # Don't enter fire
        
        # Start traversing edge
        if env.G.has_edge(person.node_id, next_node):
            edge_meta: EdgeMeta = env.G.edges[person.node_id, next_node]['meta']
            cost = _edge_cost_for_person(env, person, person.node_id, next_node, edge_meta)
            
            if math.isinf(cost):
                continue  # Blocked
            
            # Start movement
            person.on_edge = True
            person.edge_u = person.node_id
            person.edge_v = next_node
            person.edge_eta = max(1.0, cost)
            env._edge_inc(person.edge_u, person.edge_v)


def _person_route_next(env: 'BuildingFireEnvironment', person: Person) -> Optional[str]:
    """
    Get next node for person using dynamic Dijkstra.
    
    Returns next node ID on shortest path to nearest exit.
    """
    exits = [nid for nid, meta in env.nodes.items() if meta.ntype == 'exit']
    if not exits:
        return None
    
    # Dijkstra from person's location
    dist = {n: float('inf') for n in env.G.nodes}
    prev = {n: None for n in env.G.nodes}
    dist[person.node_id] = 0.0
    
    pq = [(0.0, person.node_id)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        # Early exit if reached an exit
        if env.nodes[u].ntype == 'exit':
            break
        
        for v in env.G.neighbors(u):
            edge_meta: EdgeMeta = env.G.edges[u, v]['meta']
            cost = _edge_cost_for_person(env, person, u, v, edge_meta)
            
            if math.isinf(cost):
                continue
            
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    
    # Find nearest exit
    nearest_exit = min(exits, key=lambda e: dist[e], default=None)
    
    if nearest_exit is None or math.isinf(dist[nearest_exit]):
        return None
    
    # Backtrack to find next step
    current = nearest_exit
    while prev[current] is not None:
        if prev[current] == person.node_id:
            return current
        current = prev[current]
    
    return None


def _edge_cost_for_person(env: 'BuildingFireEnvironment', person: Person, 
                          u: str, v: str, edge_meta: EdgeMeta) -> float:
    """
    Compute time-dependent edge cost for person (in timesteps).
    
    UNITS: Returns cost in timesteps (1 timestep = ~5 seconds).
    
    FORMULA: 
        base_time = length / v_effective (seconds)
        congestion_factor = 1 + θ·density
        base_cost = (base_time * congestion_factor) / 5.0  (convert to timesteps)
        final_cost = base_cost * hazard_multiplier
    
    ===== NEW: Hazard penalties scale with continuous fire intensity and smoke density =====
    Instead of binary hazard flags (0 or 1), we now use:
    - Fire intensity: 0.0 (no fire) to 1.0 (fully involved)
    - Smoke density: 0.0 (clear) to 1.0 (dense smoke)
    
    Hazard multiplier calculation:
    - Fire: increases cost by up to 50% at full intensity (1.0 + 0.5 * intensity)
    - Smoke: increases cost by up to 10% at full density (1.0 + 0.1 * density)
    
    This provides:
    1. More realistic representation of hazard severity
    2. Better gradient information for path planning
    3. Allows people to make nuanced decisions (weak smoke vs. heavy smoke)
    """
    # Get effective speed (accounts for assistance and panic)
    # Use v_class as fallback if effective_speed property doesn't exist
    effective_speed = getattr(person, 'effective_speed', person.v_class)
    
    # Base traversal time in seconds
    base_time_seconds = edge_meta.length / max(effective_speed, 1e-6)
    
    # Congestion factor (more people = slower movement)
    density = _edge_density(env, u, v)
    theta = env.config['theta_density']
    congestion_factor = 1.0 + theta * density
    
    # Apply congestion and convert to timesteps (1 timestep = 5 seconds)
    base_time_timesteps = (base_time_seconds * congestion_factor) / 5.0
    
    # ===== NEW: Hazard penalties scale with continuous intensity =====
    u_node = env.nodes[u]
    v_node = env.nodes[v]
    
    # Get maximum fire intensity and smoke density along the edge
    # Use getattr with fallback to handle nodes without intensity fields
    # (backward compatibility for older code or nodes that haven't been updated)
    max_fire = max(
        getattr(u_node, "fire_intensity", 1.0 if u_node.on_fire else 0.0),
        getattr(v_node, "fire_intensity", 1.0 if v_node.on_fire else 0.0)
    )
    max_smoke = max(
        getattr(u_node, "smoke_density", 1.0 if u_node.smoky else 0.0),
        getattr(v_node, "smoke_density", 1.0 if v_node.smoky else 0.0)
    )
    
    # Calculate hazard multiplier based on intensity
    # Fire and smoke are mutually exclusive (fire dominates)
    hazard_multiplier = 1.0
    if max_fire > 0.0:
        # Fire increases cost by up to 50% at full intensity
        # Examples:
        #   - fire_intensity=0.2 → 1.0 + 0.5*0.2 = 1.1 (10% slower)
        #   - fire_intensity=0.5 → 1.0 + 0.5*0.5 = 1.25 (25% slower)
        #   - fire_intensity=1.0 → 1.0 + 0.5*1.0 = 1.5 (50% slower)
        hazard_multiplier *= (1.0 + 0.5 * max_fire)
    elif max_smoke > 0.0:
        # Smoke increases cost by up to 10% at full density
        # Examples:
        #   - smoke_density=0.3 → 1.0 + 0.1*0.3 = 1.03 (3% slower)
        #   - smoke_density=1.0 → 1.0 + 0.1*1.0 = 1.1 (10% slower)
        hazard_multiplier *= (1.0 + 0.1 * max_smoke)
    # =================================================================
    
    # Return total cost in timesteps
    return base_time_timesteps * hazard_multiplier

def _edge_density(env: 'BuildingFireEnvironment', u: str, v: str) -> float:
    """Compute edge density (people per meter)."""
    count = env._edge_load.get((u, v), 0)
    edge_meta: EdgeMeta = env.G.edges[u, v]['meta']
    length = max(edge_meta.length, 1e-6)
    return count / length


def _shortest_distance_to_any_exit(env: 'BuildingFireEnvironment', start_node: str) -> Optional[float]:
    """Compute shortest physical distance to any exit."""
    exits = [nid for nid, meta in env.nodes.items() if meta.ntype == 'exit']
    if not exits:
        return None
    
    def weight(u, v, edata):
        meta = edata.get("meta", None)
        return getattr(meta, "length", 1.0) if meta is not None else 1.0
    
    best = None
    for ex in exits:
        try:
            dist = nx.dijkstra_path_length(env.G, start_node, ex, weight=weight)
            if best is None or dist < best:
                best = dist
        except nx.NetworkXNoPath:
            continue
    return best