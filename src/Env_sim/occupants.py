import math, heapq
from typing import Optional
from .entities import Person, EdgeMeta
import networkx as nx

def move_civilians(self) -> None:
        """
        Move civilians toward exits using dynamic pathfinding.
        
        Logic per image requirements:
        1. Awareness delay: some wait before moving
        2. Dynamic shortest path: minimize time-dependent cost
           c_e(t) = length/v_class * (1 + θ·density) + hazard_penalty
        3. Re-route if blocked or visibility drops
        """
        # Progress people on edges
        for person in self.people.values():
            if not person.is_alive or person.rescued:
                continue
            
            if person.on_edge:
                person.edge_eta -= 1.0
                
                if person.edge_eta <= 0:
                    # Arrival
                    self._edge_dec(person.edge_u, person.edge_v)
                    person.on_edge = False
                    person.node_id = person.edge_v
                    person.edge_u = None
                    person.edge_v = None
                    person.edge_eta = 0.0
                    
                    # Check if reached exit
                    if self.nodes[person.node_id].ntype == 'exit':
                        person.rescued = True
                        self.stats['people_rescued'] += 1
        
                # Start movement for idle, aware people
        for person in self.people.values():
            if not person.is_alive or person.rescued or person.on_edge:
                continue
            
            # Awareness delay
            if person.awareness_timer > 0:
                person.awareness_timer -= 1
                continue
            
            # Get next node on path
            next_node = self._person_route_next(person)
            
            if next_node is None:
                continue
            
            # Check if blocked by fire
            next_node_meta = self.nodes[next_node]
            if next_node_meta.on_fire:
                continue  # Don't enter fire
            
            # Start traversing edge
            edge_meta: EdgeMeta = self.G.edges[person.node_id, next_node]['meta']
            cost = self._edge_cost_for_person(person, person.node_id, next_node, edge_meta)
            
            if math.isinf(cost):
                continue  # Blocked
            
            # Start movement
            person.on_edge = True
            person.edge_u = person.node_id
            person.edge_v = next_node
            person.edge_eta = max(1.0, cost)
            self._edge_inc(person.edge_u, person.edge_v)
    
def _person_route_next(self, person: Person) -> Optional[str]:
        """
        Get next node for person using dynamic Dijkstra.
        
        Returns next node ID on shortest path to nearest exit.
        """
        exits = [nid for nid, meta in self.nodes.items() if meta.ntype == 'exit']
        if not exits:
            return None
        
        # Dijkstra from person's location
        dist = {n: float('inf') for n in self.G.nodes}
        prev = {n: None for n in self.G.nodes}
        dist[person.node_id] = 0.0
        
        pq = [(0.0, person.node_id)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            
            # Early exit if reached an exit
            if self.nodes[u].ntype == 'exit':
                break
            
            for v in self.G.neighbors(u):
                edge_meta: EdgeMeta = self.G.edges[u, v]['meta']
                cost = self._edge_cost_for_person(person, u, v, edge_meta)
                
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
    
def _edge_cost_for_person(self, person: Person, u: str, v: str, edge_meta: EdgeMeta) -> float:
        """
        Compute time-dependent edge cost for person.
        
        Formula: c_e(t) = length/v_class * (1 + θ·density) + hazard_penalty
        """
        # Base traversal time
        base_time = edge_meta.length / max(person.v_class, 1e-6)
        
        # Congestion factor
        density = self._edge_density(u, v)
        theta = self.config['theta_density']
        congestion_factor = 1.0 + theta * density
        
        # Hazard penalties
        hazard_penalty = 0.0
        u_node = self.nodes[u]
        v_node = self.nodes[v]
        
        if u_node.on_fire or v_node.on_fire:
            hazard_penalty = self.config['hazard_penalty_fire']
        elif u_node.smoky or v_node.smoky:
            hazard_penalty = self.config['hazard_penalty_smoke']
        
        return base_time * congestion_factor + hazard_penalty
    
def _edge_density(self, u: str, v: str) -> float:
        """Compute edge density (people per meter)."""
        count = self._edge_load.get((u, v), 0)
        edge_meta: EdgeMeta = self.G.edges[u, v]['meta']
        length = max(edge_meta.length, 1e-6)
        return count / length
    
def _edge_inc(self, u: str, v: str) -> None:
        """Increment edge load."""
        key = (u, v)
        self._edge_load[key] = self._edge_load.get(key, 0) + 1
    
def _edge_dec(self, u: str, v: str) -> None:
        """Decrement edge load."""
        key = (u, v)
        if key in self._edge_load:
            self._edge_load[key] = max(0, self._edge_load[key] - 1)
    
def _shortest_distance_to_any_exit(self, start_node: str) -> Optional[float]:
        """Compute shortest distance to any exit."""
        exits = [nid for nid, meta in self.nodes.items() if meta.ntype == 'exit']
        if not exits:
            return None
        
        def weight(u, v, edata):
            meta = edata.get("meta", None)
            return getattr(meta, "length", 1.0) if meta is not None else 1.0
        
        best = None
        for ex in exits:
            try:
                dist = nx.dijkstra_path_length(self.G, start_node, ex, weight=weight)
                if best is None or dist < best:
                    best = dist
            except nx.NetworkXNoPath:
                continue
        return best