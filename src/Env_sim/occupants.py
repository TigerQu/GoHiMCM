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
    Compute time-dependent edge cost for person.
    
    Formula: c_e(t) = length/v_effective * (1 + θ·density) + hazard_penalty
    """
    # Get effective speed (accounts for assistance and panic)
    effective_speed = person.effective_speed
    
    # Base traversal time
    base_time = edge_meta.length / max(effective_speed, 1e-6)
    
    # Congestion factor
    density = _edge_density(env, u, v)
    theta = env.config['theta_density']
    congestion_factor = 1.0 + theta * density
    
    # Hazard penalties
    hazard_penalty = 0.0
    u_node = env.nodes[u]
    v_node = env.nodes[v]
    
    if u_node.on_fire or v_node.on_fire:
        hazard_penalty = env.config['hazard_penalty_fire']
    elif u_node.smoky or v_node.smoky:
        hazard_penalty = env.config['hazard_penalty_smoke']
    
    return base_time * congestion_factor + hazard_penalty


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