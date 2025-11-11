from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# Node type encoding for one-hot vectors 
NODE_TYPES = {
    "room": 0,    # Regular rooms where people may be located
    "hall": 1,    # Hallways connecting rooms
    "exit": 2     # Exit points of the building
}

# Feature dimension: F = 10 as described in the specification
FEATURE_DIM = 10

# Default configuration values
DEFAULT_CONFIG = {
    # Hazard spreading parameters
    "fire_spread_prob": 0.3,        # Probability fire spreads to adjacent node
    "fire_spread_delay": 2,          # Time steps between spread attempts
    
    # Health degradation parameters
    "hp_loss_fire": 5.0,             # HP lost per step in fire
    "hp_loss_smoke": 2.0,            # HP lost per step in smoke
    
    # Agent parameters
    "agent_speed": 1.0,              # Nodes per time step (simplified)
    "search_time": 1,                # Time steps to search a node
    
    # People parameters (for future evacuation modeling)
    "child_speed": 0.9,              # m/s
    "adult_speed": 1.2,              # m/s
    "limited_speed": 0.7,            # m/s (elderly/disabled)
}


# DATA CLASSES

@dataclass
class Person:
    """
    Represents an occupant in the building.
    
    Attributes:
        pid: Unique person ID
        age: Age in years
        mobility: Mobility category ('child', 'adult', 'limited')
        hp: Health points (0-100), decreases in hazardous conditions
        seen: Whether an agent has discovered this person
        rescued: Whether person has been evacuated
        node_id: Current location (node ID in graph)
    """
    pid: int
    age: int
    mobility: str  # 'child' | 'adult' | 'limited'
    hp: float = 100.0
    seen: bool = False
    rescued: bool = False
    node_id: Optional[str] = None
    awareness_timer: int = 0
    on_edge: bool = False
    edge_u: Optional[str] = None
    edge_v: Optional[str] = None
    edge_eta: float = 0.0
    v_class: float = 1.2
    evac_distance: Optional[float] = None  # shortest path length to any exit at discovery

    
    @property
    def is_alive(self) -> bool:
        """Person is alive if HP > 0"""
        return self.hp > 0.0


@dataclass
class NodeMeta:
    """
    Metadata for each node in the building graph.
    
    Contains both ground-truth state (actual fire/smoke/people) and 
    observed state (what agents can see through sensors or direct observation).
    """
    nid: str                          # Node ID (unique identifier)
    ntype: str                        # Node type: 'room', 'hall', or 'exit'
    
    # Physical properties
    area: float = 10.0                # Square meters
    length: float = 5.0               # Characteristic length for traversal time
    floor: int = 0                    # Floor number (for multi-story buildings)
 
    # Ground truth hazard state (actual conditions)
    on_fire: bool = False             # True if node is currently burning
    smoky: bool = False               # True if node has smoke
    
    # Ground truth occupancy (actual people present)
    people: List[int] = field(default_factory=list)  # List of person IDs
    
    # Agent presence
    agent_here: bool = False          # True if any agent is at this node
    swept: bool = False               # True if node has been searched by agent
    
    # Observable state (what the policy can see)
    obs_people_count: int = 0         # Observed number of people (0-3 range)
    obs_avg_hp: float = 0.0           # Observed average HP (0-1 normalized)
    
    # Precomputed features
    dist_to_fire_norm: float = 1.0    # Normalized distance to nearest fire


@dataclass
class EdgeMeta:
    """
    Metadata for edges (connections between nodes).
    
    Represents doors, passages, or other connections between spaces.
    """
    u: str                            # Source node ID
    v: str                            # Target node ID
    
    # Physical properties
    width: float = 1.0                # Width in meters (affects flow capacity)
    length: float = 1.0               # Length in meters (affects traversal time)
    slope: float = 0.0                # Slope in degrees (for stairs)
    
    # Door properties
    door: bool = True                 # True if this is a doorway
    fire_door: bool = False           # True if fire-rated door (slows fire spread)


@dataclass
class Agent:
    """
    Represents a firefighter agent sweeping the building.
    
    Attributes:
        agent_id: Unique agent ID
        node_id: Current location (node ID)
        path: Planned path (list of node IDs to visit)
        searching: True if currently searching a node
        search_timer: Steps remaining for current search
    """
    agent_id: int
    node_id: str
    path: List[str] = field(default_factory=list)
    searching: bool = False
    search_timer: int = 0


#main environment class

class BuildingFireEnvironment:
    """
    Main simulation environment for fire evacuation.
    
    This class manages:
    - Building topology (graph of nodes and edges)
    - People and their states
    - Firefighter agents and their actions
    - Hazard dynamics (fire and smoke spread)
    - Observation generation for learning algorithms
    
    The environment operates in discrete time steps.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the environment.
        
        Args:
            config: Optional configuration dictionary (uses defaults if not provided)
        """
        # Configuration
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Building topology
        self.G = nx.Graph()                    # NetworkX graph for topology
        self.nodes: Dict[str, NodeMeta] = {}   # Node metadata by ID
        
        # Entities
        self.people: Dict[int, Person] = {}    # People by ID
        self.agents: Dict[int, Agent] = {}     # Agents by ID
        
        # Simulation state
        self.time_step = 0                     # Current simulation time
        self.fire_spread_counter = 0           # Counter for fire spread timing
        
        # Statistics tracking
        self.stats = {
            "people_rescued": 0,
            "people_found": 0,
            "nodes_swept": 0,
            "total_search_time": 0,
        }

    def _shortest_distance_to_any_exit(self, start_node: str) -> Optional[float]:
        exits = [nid for nid, meta in self.nodes.items() if meta.ntype == 'exit']
        if not exits:
            return None

        # Weight = physical length (fallback to 1.0 if meta missing)
        def w(u, v, edata):
            meta = edata.get("meta", None)
            return getattr(meta, "length", 1.0) if meta is not None else 1.0
        best = None
        for ex in exits:
            try:
                dist = nx.dijkstra_path_length(self.G, start_node, ex, weight=w)
                if best is None or dist < best:
                    best = dist
            except nx.NetworkXNoPath:
                continue
        return best
      

    # construct building
    
    def add_node(self, nid: str, ntype: str, **kwargs) -> None:
        """
        Add a node (room/hallway/exit) to the building.
        
        Args:
            nid: Node ID (unique string identifier)
            ntype: Node type ('room', 'hall', or 'exit')
            **kwargs: Additional node properties (area, length, has_sensor, etc.)
        """
        if ntype not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {ntype}. Must be one of {list(NODE_TYPES.keys())}")
        
        meta = NodeMeta(nid=nid, ntype=ntype, **kwargs)
        self.nodes[nid] = meta
        self.G.add_node(nid)
    
    
    def add_edge(self, u: str, v: str, **kwargs) -> None:
        """
        Add an edge (connection) between two nodes.
        
        Args:
            u: Source node ID
            v: Target node ID
            **kwargs: Edge properties (width, length, door, fire_door, etc.)
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {u}, {v}")
        
        edge_meta = EdgeMeta(u=u, v=v, **kwargs)
        self.G.add_edge(u, v, meta=edge_meta)
    
    
    def spawn_person(self, node_id: str, age: int, mobility: str, hp: float = 100.0) -> int:
        """
        Add a person to the building at the specified location.
        
        Args:
            node_id: Node where person starts
            age: Age in years
            mobility: Mobility category ('child', 'adult', 'limited')
            hp: Initial health points (default 100)
        
        Returns:
            Person ID (integer)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        pid = len(self.people)
        person = Person(pid=pid, age=age, mobility=mobility, hp=hp, node_id=node_id)
        self.people[pid] = person
        self.nodes[node_id].people.append(pid)
        return pid
    
    
    def place_agent(self, agent_id: int, node_id: str) -> None:
        """
        Place a firefighter agent at a specific location.
        
        Args:
            agent_id: Agent ID (integer)
            node_id: Node where agent starts
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        agent = Agent(agent_id=agent_id, node_id=node_id)
        self.agents[agent_id] = agent
        
        # Update node states
        self._update_agent_positions()
    

    # fire and smoke set up
    
    def ignite_node(self, node_id: str) -> None:
        """
        Start a fire at the specified node.
        
        Args:
            node_id: Node to ignite
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        self.nodes[node_id].on_fire = True
        self.nodes[node_id].smoky = True  # Fire implies smoke
        print(f"[T={self.time_step}] Fire ignited at {node_id}")
    
    
    def spread_hazards(self) -> None:
        """
        Spread fire and smoke to adjacent nodes.
        
        Logic:
        1. Fire spreads probabilistically to neighbors (slower through fire doors)
        2. Smoke spreads more easily than fire
        3. Fire always produces smoke
        """
        # Only spread fire periodically (not every step)
        self.fire_spread_counter += 1
        if self.fire_spread_counter < self.config["fire_spread_delay"]:
            return
        self.fire_spread_counter = 0
        
        # Identify currently burning nodes
        burning_nodes = [n for n in self.nodes.values() if n.on_fire]
        
        # Spread from each burning node
        for burning_node in burning_nodes:
            for neighbor_id in self.G.neighbors(burning_node.nid):
                neighbor = self.nodes[neighbor_id]
                edge_data = self.G.edges[burning_node.nid, neighbor_id]
                edge_meta: EdgeMeta = edge_data["meta"]
                
                # Fire doors reduce spread probability
                spread_prob = self.config["fire_spread_prob"]
                if edge_meta.fire_door:
                    spread_prob *= 0.3  # Fire doors are 70% more effective
                # Attempt to spread fire
                if not neighbor.on_fire and random.random() < spread_prob:
                    neighbor.on_fire = True
                    print(f"[T={self.time_step}] Fire spread to {neighbor_id}")
                
                # Smoke spreads more easily (always to adjacent nodes of fire)
                neighbor.smoky = True
    
    
    def degrade_health(self) -> None:
        """
        Reduce HP of people in hazardous conditions.
        
        Logic:
        - People in fire lose HP faster than people in smoke
        - HP cannot go below 0
        - Dead people (HP=0) are tracked but no longer degrade
        """
        for person in self.people.values():
            if not person.is_alive:
                continue  # Skip dead people
            
            node = self.nodes[person.node_id]
            
            # Calculate HP loss based on hazards
            hp_loss = 0.0
            if node.on_fire:
                hp_loss = self.config["hp_loss_fire"]
            elif node.smoky:
                hp_loss = self.config["hp_loss_smoke"]
            
            # Apply damage
            if hp_loss > 0:
                person.hp = max(0.0, person.hp - hp_loss)
                if not person.is_alive:
                    print(f"[T={self.time_step}] Person {person.pid} died at {person.node_id}")
    

    # Agent Actions
    
    def move_agent(self, agent_id: int, target_node_id: str) -> bool:
        """
        Move an agent to an adjacent node.
        
        Args:
            agent_id: Agent to move
            target_node_id: Destination node (must be adjacent)
        
        Returns:
            True if move successful, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        current_node = agent.node_id
        
        # Check if target is adjacent
        if target_node_id not in self.G.neighbors(current_node):
            print(f"[Warning] Agent {agent_id} cannot move from {current_node} to {target_node_id} (not adjacent)")
            return False
        
        # Move agent
        agent.node_id = target_node_id
        agent.searching = False  # Stop any ongoing search
        agent.search_timer = 0
        
        self._update_agent_positions()
        return True
    
    
    def start_search(self, agent_id: int) -> bool:
        """
        Begin searching the current node for people.
        
        Args:
            agent_id: Agent who will search
        
        Returns:
            True if search started, False if already searching or invalid
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Can't search if already searching
        if agent.searching:
            return False
        
        # Start search
        agent.searching = True
        agent.search_timer = self.config["search_time"]
        print(f"[T={self.time_step}] Agent {agent_id} started searching {agent.node_id}")
        return True
    
    
    def process_searches(self) -> None:
        """
        Process ongoing searches and complete them when timers expire.
        
        Logic:
        - Decrement search timers
        - When timer reaches 0:
          1. Mark node as swept
          2. Reveal all people in node
          3. Update statistics
        """
        for agent in self.agents.values():
            if not agent.searching:
               continue

            agent.search_timer -= 1
            self.stats["total_search_time"] += 1

            if agent.search_timer <= 0:
               agent.searching = False
               node = self.nodes[agent.node_id]
               
               if not node.swept:
                node.swept = True
                self.stats["nodes_swept"] += 1

            # --- DISCOVERY LOGIC + EVAC DISTANCE ---
            for pid in node.people:
                person = self.people[pid]

                first_time_seen = not person.seen
                if first_time_seen:
                    person.seen = True
                    self.stats["people_found"] += 1

                    # Compute shortest distance to any exit at discovery time
                    person.evac_distance = self._shortest_distance_to_any_exit(person.node_id)

                    # If still alive at discovery, count as evacuated success immediately
                    if person.is_alive and not person.rescued:
                        person.rescued = True
                        self.stats["people_rescued"] += 1

            if getattr(self, "debug", False):
                print(f"[T={self.time_step}] Agent {agent.agent_id} completed search of {agent.node_id}")


    
    # Oberservation and feature extraction
    
    def _update_agent_positions(self) -> None:
        """
        Update agent_here flags for all nodes based on current agent positions.
        """
        # Clear all flags
        for node in self.nodes.values():
            node.agent_here = False
        
        # Set flags for occupied nodes
        for agent in self.agents.values():
            self.nodes[agent.node_id].agent_here = True
    
    
    def _update_observations(self) -> None:
        """
        Update observable people counts based on sensors and agent presence.
        
        Logic (Partial Observability):
        - By default, people are NOT observable (obs_people_count = 0)
        - Agent presence reveals full details (and marks people as seen)
        - This creates the exploration incentive for agents
        """
        # Reset observations
        for node in self.nodes.values():
            node.obs_people_count = 0
            node.obs_avg_hp = 0.0
        
        # Agent observations (full details, marks people as seen)
        for agent in self.agents.values():
            node = self.nodes[agent.node_id]
            alive_people = [self.people[pid] for pid in node.people 
                           if self.people[pid].is_alive]
            count = len(alive_people)
            
            if count > 0:
                node.obs_people_count = min(count, 3)
                avg_hp = sum(p.hp for p in alive_people) / count
                node.obs_avg_hp = avg_hp / 100.0
                
                # Mark people as seen (ground truth updated)
                for pid in node.people:
                    self.people[pid].seen = True
    
    
    def _compute_fire_distances(self) -> None:
        """
        Compute shortest path distance from each node to nearest fire.
        Uses NetworkX's multi-source Dijkstra for efficiency.
        Distances are normalized by dividing by 10.0 and capping at 1.0.
        """
        # Find all nodes currently on fire
        fire_nodes = [n.nid for n in self.nodes.values() if n.on_fire]
        
        if not fire_nodes:
            # No fire yet - all nodes have max distance
            for node in self.nodes.values():
                node.dist_to_fire_norm = 1.0
            return
        
        # Multi-source shortest path from all fire nodes
        # Weight by edge length for physical distance
        distances = nx.multi_source_dijkstra_path_length(
            self.G, 
            fire_nodes,
            weight=lambda u, v, d: d["meta"].length
        )
        
        # Normalize distances
        for node in self.nodes.values():
            raw_dist = distances.get(node.nid, 10.0)  # Default 10m if unreachable
            node.dist_to_fire_norm = min(raw_dist / 10.0, 1.0)  # Normalize and cap
    
    
    def get_node_features(self, node: NodeMeta) -> np.ndarray:
        """
        Construct the 10-dimensional feature vector for a node.
        
        Feature breakdown (F = 10):
        [0:3]   One-hot node type (room/hall/exit)
        [3]     Fire indicator (0 or 1)
        [4]     Smoke indicator (0 or 1)
        [5]     Length normalized by 10
        [6]     People count normalized (capped at 3, divided by 3)
        [7]     Average HP normalized (divided by 100)
        [8]     Agent presence indicator (0 or 1)
        [9]     Distance to fire normalized (divided by 10, capped at 1)
        
        Args:
            node: NodeMeta object
        
        Returns:
            Numpy array of shape (10,)
        """
        # Features 1-3: One-hot encoding of node type
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[NODE_TYPES[node.ntype]] = 1.0
        
        # Feature 4: Fire indicator
        fire = 1.0 if node.on_fire else 0.0
        
        # Feature 5: Smoke indicator
        smoke = 1.0 if node.smoky else 0.0
        
        # Feature 6: Normalized length
        length_norm = node.length / 10.0
        
        # Feature 7: Normalized people count (observable, capped at 3)
        people_count_norm = min(node.obs_people_count, 3) / 3.0
        
        # Feature 8: Normalized average HP (observable)
        avg_hp_norm = node.obs_avg_hp  # Already normalized in update
        
        # Feature 9: Agent presence
        agent_here = 1.0 if node.agent_here else 0.0
        
        # Feature 10: Normalized distance to fire
        dist_fire = node.dist_to_fire_norm
        
        # Assemble feature vector
        features = np.array([
            *one_hot,           # [0:3]
            fire,               # [3]
            smoke,              # [4]
            length_norm,        # [5]
            people_count_norm,  # [6]
            avg_hp_norm,        # [7]
            agent_here,         # [8]
            dist_fire           # [9]
        ], dtype=np.float32)
        
        assert features.shape[0] == FEATURE_DIM, f"Feature dimension mismatch: {features.shape[0]} != {FEATURE_DIM}"
        return features
    
    
    def to_pytorch_geometric(self) -> Tuple[Data, Dict]:
        """
        Convert current environment state to PyTorch Geometric Data object.     
        This is used for GNN-based policies that operate on graph structures.
        Returns:
            data: PyG Data object with:
                - x: Node features [N, 10]
                - edge_index: Edge connectivity [2, E]
                - edge_attr: Edge features [E, 5]
                - nid: List of node IDs (for debugging)
            env_state: Dictionary with ground truth state (for evaluation)
        """
        # Update observations before creating features
        self._update_observations()
        self._compute_fire_distances()
        
        # Create node index mapping
        node_ids = list(self.nodes.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Construct node feature matrix X: [N, F]
        node_features = []
        for nid in node_ids:
            node = self.nodes[nid]
            features = self.get_node_features(node)
            node_features.append(features)
        
        X = np.stack(node_features, axis=0)  # Shape: [N, 10]
        
        # Construct edge_index and edge_attr
        edge_src, edge_dst, edge_features = [], [], []
        
        for u, v, data in self.G.edges(data=True):
            edge_meta: EdgeMeta = data["meta"]
            
            # PyG expects directed edges, so add both directions
            i, j = node_to_idx[u], node_to_idx[v]
            
            # Edge features: [width, length, slope, door, fire_door]
            edge_feat = [
                edge_meta.width,
                edge_meta.length,
                edge_meta.slope,
                float(edge_meta.door),
                float(edge_meta.fire_door)
            ]
    
            # Add edge u->v
            edge_src.append(i)
            edge_dst.append(j)
            edge_features.append(edge_feat)
            
            # Add edge v->u (undirected graph)
            edge_src.append(j)
            edge_dst.append(i)
            edge_features.append(edge_feat)
        
        # Convert to tensors
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        x = torch.tensor(X, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data.nid = node_ids  # Store for debugging
        
        # Ground truth state for evaluation
        env_state = {
            "nodes": self.nodes,
            "people": self.people,
            "agents": self.agents,
            "time_step": self.time_step,
            "stats": self.stats
        }
        
        return data, env_state
    

    # Every simulation step

    
    def step(self) -> None:
        """
        After each time step, update the environment state.
        
        Order of operations:
        1. Process agent searches (complete any ongoing searches)
        2. Spread hazards (fire and smoke)
        3. Degrade health (people take damage from hazards)
        4. Increment time
        
        Note: Agent movement is handled separately via move_agent() calls
        """
        # Process searches
        self.process_searches()
        
        # Spread hazards
        self.spread_hazards()
        
        # Health degradation
        self.degrade_health()
        
        # Advance time
        self.time_step += 1
    
    
    def is_sweep_complete(self) -> bool:
        """
        Check if the building sweep is complete.
        
        A sweep is complete when all nodes have been searched (swept).
        
        Returns:
            True if all nodes are swept, False otherwise
        """
        return all(node.swept for node in self.nodes.values())
    
    
    def get_statistics(self) -> Dict:
        """
        Get current simulation statistics.
        
        Returns:
            Dictionary with statistics:
            - people_rescued: Number of people evacuated
            - people_found: Number of people discovered
            - nodes_swept: Number of nodes searched
            - total_search_time: Total time spent searching
            - time_step: Current time step
            - sweep_complete: Whether sweep is done
        """
        stats = {
            **self.stats,
            "time_step": self.time_step,
            "sweep_complete": self.is_sweep_complete(),
            "total_people": len(self.people),
            "people_alive": sum(1 for p in self.people.values() if p.is_alive),
        }
        return stats
    
    
    def print_status(self) -> None:
        """Print current environment status for debugging."""
        print(f"\n{'='*60}")
        print(f"Time Step: {self.time_step}")
        print(f"{'='*60}")
        
        # Agent status
        print("\nAgents:")
        for agent in self.agents.values():
            status = "SEARCHING" if agent.searching else "IDLE"
            print(f"  Agent {agent.agent_id}: {agent.node_id} ({status})")
        
        # Node status
        print("\nNodes:")
        for node in self.nodes.values():
            hazards = []
            if node.on_fire: hazards.append("FIRE")
            if node.smoky: hazards.append("SMOKE")
            hazard_str = ", ".join(hazards) if hazards else "clear"
            
            swept_str = "SWEPT" if node.swept else "unswept"
            people_str = f"{len(node.people)} people" if node.people else "empty"
            
            print(f"  {node.nid:6s} [{node.ntype:4s}]: {hazard_str:12s} | {swept_str:8s} | {people_str}")
        
        # Statistics
        print("\nStatistics:")
        for key, value in self.get_statistics().items():
            print(f"  {key}: {value}")
        print()


# Build the layout


def build_standard_office_layout() -> BuildingFireEnvironment:
    """
    Build the standard one-floor office layout as specified:
    - Central hallway divided into 3 segments (for granularity)
    - 3 rooms on top side of hallway
    - 3 rooms on bottom side of hallway
    - 2 exits (left and right ends of hallway)
    
    Returns:
        Initialized BuildingFireEnvironment
    """
    env = BuildingFireEnvironment()
    
    # Central Hallway (3 segments)
  
    for i in range(3):
        env.add_node(
            nid=f"H{i}",
            ntype="hall",
            length=6.0,        # 6 meters per segment
            area=12.0,         # 2m wide × 6m long
            floor=0
        )
    
    # Connect hallway segments
    env.add_edge("H0", "H1", length=6.0, width=2.0, door=False)  # Open hallway
    env.add_edge("H1", "H2", length=6.0, width=2.0, door=False)
    
    # ROOMS - TOP ROW (3 rooms along top of hallway)

    for i in range(3):
        env.add_node(
            nid=f"RT{i}",
            ntype="room",
            length=4.0,        # 4m × 4m room
            area=16.0,
            floor=0,
        )

        # Connect to corresponding hallway segment

        env.add_edge(
            f"RT{i}", f"H{i}",
            length=1.0,        # 1m doorway depth
            width=0.9,         # Standard door width
            door=True
        )
    

    #  3 rooms along bottom of hallway

    for i in range(3):
        env.add_node(
            nid=f"RB{i}",
            ntype="room",
            length=4.0,
            area=16.0,
            floor=0,   
        )
        # Connect to corresponding hallway segment
        env.add_edge(
            f"RB{i}", f"H{i}",
            length=1.0,
            width=0.9,
            door=True
        )
    

    # exists - left and right ends of hallway
    env.add_node(
        nid="EXIT_LEFT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0
    )
    env.add_node(
        nid="EXIT_RIGHT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0
    )
    
    # Connect exits to hallway ends
    env.add_edge("EXIT_LEFT", "H0", length=1.0, width=1.2, door=True)
    env.add_edge("EXIT_RIGHT", "H2", length=1.0, width=1.2, door=True)
    

    # populated with people (Hidden until agents search)
    
    env.spawn_person("RT0", age=34, mobility="adult", hp=95.0)
    env.spawn_person("RT2", age=55, mobility="limited", hp=90.0)
    env.spawn_person("RB0", age=28, mobility="adult", hp=100.0)
    env.spawn_person("RB2", age=6, mobility="child", hp=100.0) 
    env.spawn_person("RB1", age=32, mobility="adult", hp=100.0)
    env.spawn_person("RB1", age=41, mobility="adult", hp=97.0)
    

    # place agents at exits

    env.place_agent(agent_id=0, node_id="EXIT_LEFT")
    env.place_agent(agent_id=1, node_id="EXIT_RIGHT")
    
    return env



"""
if __name__ == "__main__":
    print("Building Fire Evacuation Simulation")
    print("=" * 60)
    
    # Create environment
    env = build_standard_office_layout()
    
    # Start fire in one room
    env.ignite_node("RT1")
    
    # Print initial status
    env.print_status()
    
    # Get PyTorch Geometric representation
    pyg_data, state = env.to_pytorch_geometric()
    print("\nPyTorch Geometric Data:")
    print(f"  Nodes: {pyg_data.x.shape}")
    print(f"  Edges: {pyg_data.edge_index.shape}")
    print(f"  Edge features: {pyg_data.edge_attr.shape}")
    print(f"\nNode features (first 3 nodes):")
    print(pyg_data.x[:3])
"""