from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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
    #movement state
    on_edge: bool = False
    edge_u: Optional[str] = None
    edge_v: Optional[str] = None
    edge_eta: float = 0.0
    #person properties
    v_class: float = 1.2
    evac_distance: Optional[float] = None  # shortest path length to any exit at discovery
    evac_path: Optional[List[str]] = None

    tenability_threshold: float = 40.0  # HP threshold below which person cannot self-evacuate
    # Assistance and panic state
    being_assisted: bool = False
    assisting_agent_id: Optional[int] = None
    assisted_multiplier: float = 1.8
    panic_multiplier: float = 0.7
    panicked: bool = False

    
    @property
    def is_alive(self) -> bool:
        """Person is alive if HP > 0"""
        return self.hp > 0.0
    
    @property
    def effective_speed(self) -> float:
        """Compute effective speed based on current state"""
        base_speed = self.v_class
        if self.being_assisted:
            return base_speed * self.assisted_multiplier
        elif self.panicked:
            return base_speed * self.panic_multiplier
        else:
            return base_speed
    
    @property
    def is_tenable(self) -> bool:
        """Person can self-evacuate if HP >= tenability threshold"""
        return self.hp >= self.tenability_threshold


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

    role: str = "generic"             # e.g. "classroom", "lab", "office", "storage"
    risk_level: str = "normal"        # "high" or "normal" for metrics
    required_sweeps: int = 1          # Per-node redundancy requirement r_v
 
    # Ground truth hazard state (actual conditions)
    on_fire: bool = False             # True if node is currently burning
    smoky: bool = False               # True if node has smoke
    
    # Ground truth occupancy (actual people present)
    people: List[int] = field(default_factory=list)  # List of person IDs
    
    # Agent presence
    agent_here: bool = False          # True if any agent is at this node
    sweep_count: int = 0              # Number of times node has been searched (target: 2)
    
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
        searched_nodes: Set of node IDs this agent has personally searched
        known_swept_nodes: Set of node IDs known to be searched (shared via communication)
        message_buffer: Messages received from other agents
        exposure: Cumulative exposure index (0-1+, where >1 indicates danger)
        hp: Health points (0-100), decreases in hazardous conditions
    """
    agent_id: int
    node_id: str
    path: List[str] = field(default_factory=list)
    searching: bool = False
    search_timer: int = 0
    searched_nodes: set = field(default_factory=set)  # Nodes this agent searched
    known_swept_nodes: set = field(default_factory=set)  # Nodes known via comm
    message_buffer: List[Dict] = field(default_factory=list)  # Incoming messages
    exposure: float = 0.0  # Cumulative exposure tracking for agent safety
    hp: float = 100.0  # Agent health points
 
    @property
    def is_active(self) -> bool:
        """Agent is active if not over-exposed and still has HP"""
        return self.hp > 0.0
