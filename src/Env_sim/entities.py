from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Import default config for fallback speed multipliers
from .config import DEFAULT_CONFIG

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
    # Assistance / panic state
    being_assisted: bool = False
    assisting_agent_id: Optional[int] = None
    panicked: bool = False

    @property
    def effective_speed(self) -> float:
        """Compute effective walking speed (m/s) for the person.

        Priority: assisted > panicked > normal
        Uses DEFAULT_CONFIG as a fallback; environment config may vary.
        """
        if self.being_assisted:
            return self.v_class * DEFAULT_CONFIG.get('assisted_speed_multiplier', 1.0)
        if self.panicked:
            return self.v_class * DEFAULT_CONFIG.get('panic_speed_reduction', 1.0)
        return self.v_class

    
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
    # movement state (in-transit along an edge)
    on_edge: bool = False
    edge_u: Optional[str] = None
    edge_v: Optional[str] = None
    edge_eta: float = 0.0
    # If agent is escorting a person, store that person's id
    assisting_person_id: Optional[int] = None
