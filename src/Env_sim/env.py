"""
Building Fire Environment - Main Simulation Class
==================================================

This module implements the core RL environment for firefighter building sweeps.

ENVIRONMENT LOGIC:
------------------

1. INITIALIZATION:
   - Building graph created (nodes = rooms/halls/exits, edges = doors/passages)
   - People spawned in rooms with awareness delays
   - Firefighters placed at entry points with preparation delay
   - Fire started in random room (or specified location)

2. STATE REPRESENTATION:
   - Graph structure with node features (10D vectors)
   - Partial observability: people only visible in searched rooms
   - Hazard information: fire/smoke locations fully observable
   
3. TIMESTEP DYNAMICS (each step = ~5 seconds of real time):
   
   A. AGENT ACTIONS (processed first):
      - move_<node>: Start traversing edge to adjacent node
      - search: Begin thorough search of current room
      - assist_<pid>: Start helping discovered person evacuate
      - wait: Do nothing this timestep
   
   B. AGENT PROCESSING:
      - Complete ongoing searches (reveal people, mark swept)
      - Progress movement along edges (decrement ETA)
      - Arrive at destinations (update positions)
   
   C. HAZARD DYNAMICS:
      - Fire spreads to adjacent rooms probabilistically
      - Fire intensity grows in burning rooms
      - Smoke fills connected spaces
      - Doors slow fire spread
   
   D. CIVILIAN BEHAVIOR:
      - Wait during awareness delay (don't react immediately)
      - Once aware, pathfind to nearest exit
      - Move cautiously if panicked (unassisted)
      - Move quickly if assisted by firefighter
      - Take damage from fire/smoke
      - Rescued when reaching exit
   
   E. HEALTH DEGRADATION:
      - People in fire: lose 1 HP/step (die in ~100 steps = 8 min)
      - People in smoke: lose 0.3 HP/step (die in ~330 steps = 27 min)
      - Children/elderly more vulnerable
   
   F. TERMINATION:
      - Success: All rooms swept (people found)
      - Failure: Max timesteps exceeded (timeout)
      - Early exit: All people dead

4. OBSERVATIONS:
   - PyTorch Geometric graph with node features
   - Partial observability enforced
   - Features: node type, hazards, length, people count, agent presence

5. REWARDS (placeholder - customize for specific objectives):
   - Currently set to 0.0 (environment ready for reward shaping)
   - Suggested: +reward for finding people, sweeping rooms
   - Suggested: -penalty for time, people dying
"""

from typing import Dict, List, Tuple, Optional

import random

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from .config import DEFAULT_CONFIG, FEATURE_DIM, NODE_TYPES
from .entities import Person, NodeMeta, EdgeMeta, Agent
from .hazards import ignite_node as _ignite_node
from .hazards import spread_hazards as _spread_hazards
from .hazards import degrade_health as _degrade_health
from .occupants import move_civilians as _move_civilians
from .occupants import _shortest_distance_to_any_exit


class BuildingFireEnvironment:
    """
    Main RL environment for fire evacuation and building sweep simulation.
    
    This environment models realistic firefighter operations including:
    - Staged entry (firefighters wait before entering)
    - Search and rescue operations
    - Fire and smoke dynamics
    - Civilian panic and evacuation
    - Agent-civilian assistance mechanics
    
    The environment operates in discrete timesteps (~5 seconds each) and
    provides a graph-based observation space suitable for GNN policies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the building fire environment.
        
        Args:
            config: Optional configuration dict to override defaults.
                   See config.py for available parameters.
        """
        # ========================================
        # CONFIGURATION
        # ========================================
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Ensure sweep configuration exists
        self.config.setdefault("sweep_node_types", {"room"})  # Only rooms need search
        self.config.setdefault("auto_sweep_types", {"hall", "exit"})  # Auto-sweep on entry
        self.config.setdefault("auto_sweep_on_visit", True)
        
        # NEW: Staging delay - firefighters prepare before entering
        self.config.setdefault("staging_delay", 10)  # ~50 seconds to gear up
        
        # ========================================
        # BUILDING TOPOLOGY
        # ========================================
        self.G = nx.Graph()  # NetworkX graph for topology
        self.nodes: Dict[str, NodeMeta] = {}  # Node metadata by ID
        
        # ========================================
        # ENTITIES
        # ========================================
        self.people: Dict[int, Person] = {}  # Civilians by ID
        self.agents: Dict[int, Agent] = {}  # Firefighters by ID
        
        # ========================================
        # SIMULATION STATE
        # ========================================
        self.time_step = 0  # Current timestep (0 = start of fire)
        self.fire_spread_counter = 0  # Counter for fire spread timing
        
        # ========================================
        # STATISTICS TRACKING
        # ========================================
        self.stats = {
            "people_rescued": 0,      # Number of people who reached exits
            "people_found": 0,        # Number of people discovered
            "nodes_swept": 0,         # Number of rooms searched
            "total_search_time": 0,   # Total timesteps spent searching
        }
        
        # ========================================
        # REWARD TRACKING (for RL)
        # ========================================
        self._last_people_found = 0
        self._last_nodes_swept = 0
        self._last_people_alive = 0
        self._last_people_rescued = 0
        
        # ========================================
        # RESET MEMORY
        # ========================================
        # Store initial conditions so we can reset environment
        self._initial_agent_positions: Dict[int, str] = {}
        self._initial_people_state: List[Tuple[str, int, str, float]] = []
        
        # ========================================
        # EDGE CONGESTION TRACKING
        # ========================================
        # Track how many people are on each edge (for pathfinding)
        self._edge_load: Dict[Tuple[str, str], int] = {}
        
        # ========================================
        # RANDOM NUMBER GENERATORS
        # ========================================
        self._rng = random.Random()
        self._np_rng = np.random.RandomState()
    
    # ============================================================================
    # BUILDING CONSTRUCTION METHODS
    # ============================================================================
    
    def add_node(self, nid: str, ntype: str, **kwargs) -> None:
        """
        Add a node (room/hallway/exit) to the building.
        
        Args:
            nid: Unique node identifier (e.g., "RT0", "H1", "EXIT_LEFT")
            ntype: Node type - must be "room", "hall", "exit", or "floor"
            **kwargs: Additional properties (area, length, floor, etc.)
        
        Raises:
            ValueError: If ntype is not valid
        """
        if ntype not in NODE_TYPES:
            raise ValueError(
                f"Invalid node type: {ntype}. Must be one of {list(NODE_TYPES.keys())}"
            )
        
        # Create node metadata
        meta = NodeMeta(nid=nid, ntype=ntype, **kwargs)
        self.nodes[nid] = meta
        
        # Add to NetworkX graph
        self.G.add_node(nid)
    
    def add_edge(self, u: str, v: str, **kwargs) -> None:
        """
        Add an edge (connection) between two nodes.
        
        Represents doors, passages, or stairs connecting spaces.
        
        Args:
            u: Source node ID
            v: Target node ID
            **kwargs: Edge properties (width, length, door, fire_door, etc.)
        
        Raises:
            ValueError: If either node doesn't exist
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {u}, {v}")
        
        # Create edge metadata
        edge_meta = EdgeMeta(u=u, v=v, **kwargs)
        
        # Add to NetworkX graph (undirected)
        self.G.add_edge(u, v, meta=edge_meta)
    
    def spawn_person(self, node_id: str, age: int, mobility: str, hp: float = 100.0) -> int:
        """
        Spawn a civilian at the specified location.
        
        Args:
            node_id: Where to place person (must be valid node)
            age: Age in years (affects vulnerability)
            mobility: "adult", "child", or "limited" (affects speed)
            hp: Initial health points (0-100)
        
        Returns:
            Person ID (integer)
        
        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        # Generate unique person ID
        pid = len(self.people)
        
        # Get mobility-based speed
        speed_map = {
            'adult': self.config['adult_speed'],
            'child': self.config['child_speed'],
            'limited': self.config['limited_speed']
        }
        v_class = speed_map.get(mobility, self.config['adult_speed'])
        
        # Sample awareness delay (how long before they react to alarm)
        # This models realistic human behavior - people don't immediately evacuate
        delay = max(0, int(self._np_rng.normal(
            self.config['awareness_delay_mean'],
            self.config['awareness_delay_std']
        )))
        
        # Create person object
        person = Person(
            pid=pid,
            age=age,
            mobility=mobility,
            hp=hp,
            node_id=node_id,
            v_class=v_class,
            awareness_timer=delay,
            assisted_multiplier=self.config.get('assisted_speed_multiplier', 1.8),
            panic_multiplier=self.config.get('panic_speed_reduction', 0.7)
        )
        
        # Register person
        self.people[pid] = person
        self.nodes[node_id].people.append(pid)
        
        # Store initial state for reset
        self._initial_people_state.append((node_id, age, mobility, hp))
        
        return pid
    
    def place_agent(self, agent_id: int, node_id: str) -> None:
        """
        Place a firefighter agent at a specific location.
        
        NEW: Agents start with a staging delay - they don't enter immediately.
        This models realistic operations where firefighters gear up and
        assess the situation before entry.
        
        Args:
            agent_id: Unique agent identifier (integer)
            node_id: Starting location (typically an exit/entry point)
        
        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        # Create agent object
        agent = Agent(agent_id=agent_id, node_id=node_id)
        
        # NEW: Set staging delay - agent must wait before acting
        # This represents gear-up time, briefing, size-up, etc.
        agent.search_timer = self.config.get('staging_delay', 10)
        agent.searching = True  # Use search_timer as staging timer
        
        # Register agent
        self.agents[agent_id] = agent
        self._initial_agent_positions[agent_id] = node_id
        
        # Update node states
        self._update_agent_positions()
    
    # ============================================================================
    # SEEDING AND RESET
    # ============================================================================
    
    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for deterministic behavior.
        
        Args:
            seed: Integer seed value (None for random)
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.RandomState(seed)
            random.seed(seed)
            np.random.seed(seed)
    
    def reset(self, fire_node: Optional[str] = None, seed: Optional[int] = None) -> Data:
        """
        Reset environment for a new episode.
        
        This is a required method for RL environments (Gym/Gymnasium interface).
        
        Args:
            fire_node: Node to ignite (None = random room, "none" = no fire)
            seed: Random seed for reproducibility
        
        Returns:
            Initial observation (PyTorch Geometric Data object)
        """
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
        
        # ========================================
        # RESET TIME
        # ========================================
        self.time_step = 0
        self.fire_spread_counter = 0
        
        # ========================================
        # RESET STATISTICS
        # ========================================
        self.stats = {
            "people_rescued": 0,
            "people_found": 0,
            "nodes_swept": 0,
            "total_search_time": 0,
        }
        
        # ========================================
        # RESET REWARD TRACKING
        # ========================================
        self._last_people_found = 0
        self._last_nodes_swept = 0
        self._last_people_alive = len(self.people)
        self._last_people_rescued = 0
        
        # ========================================
        # RESET EDGE CONGESTION
        # ========================================
        self._edge_load.clear()
        
        # ========================================
        # RESET ALL NODES TO CLEAN STATE
        # ========================================
        for node in self.nodes.values():
            node.on_fire = False
            node.smoky = False
            node.fire_intensity = 0.0
            node.swept = False
            node.agent_here = False
            node.obs_people_count = 0
            node.obs_avg_hp = 0.0
            node.people = []  # Clear people lists (will be repopulated)
        
        # ========================================
        # RESET ALL PEOPLE TO INITIAL STATE
        # ========================================
        for pid, (node_id, age, mobility, hp) in enumerate(self._initial_people_state):
            # If a Person object for this pid is missing (possible if state
            # was constructed elsewhere), create it here so reset is robust.
            if pid not in self.people:
                # Determine base speed class from mobility
                speed_map = {
                    'adult': self.config['adult_speed'],
                    'child': self.config['child_speed'],
                    'limited': self.config['limited_speed']
                }
                v_class = speed_map.get(mobility, self.config['adult_speed'])

                # Create and register new Person
                person = Person(
                    pid=pid,
                    age=age,
                    mobility=mobility,
                    hp=hp,
                    node_id=node_id,
                    v_class=v_class,
                    assisted_multiplier=self.config.get('assisted_speed_multiplier', 1.8),
                    panic_multiplier=self.config.get('panic_speed_reduction', 0.7)
                )
                self.people[pid] = person
            else:
                person = self.people[pid]

            # Reset health and status
            person.hp = hp
            person.seen = False
            person.rescued = False
            person.node_id = node_id
            person.being_assisted = False
            person.assisting_agent_id = None
            person.panicked = False

            # Re-sample awareness delay (realistic variation)
            delay = max(0, int(self._np_rng.normal(
                self.config['awareness_delay_mean'],
                self.config['awareness_delay_std']
            )))
            person.awareness_timer = delay

            # Reset movement state
            person.on_edge = False
            person.edge_u = None
            person.edge_v = None
            person.edge_eta = 0.0
            person.evac_distance = None
            person.evac_path = None

            # Re-add to node's people list
            self.nodes[node_id].people.append(pid)
        
        # ========================================
        # RESET ALL AGENTS TO INITIAL POSITIONS
        # ========================================
        for agent_id, start_pos in self._initial_agent_positions.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Reset position
                agent.node_id = start_pos
                
                # Reset search/movement state
                agent.searching = True  # Staging delay active
                agent.search_timer = self.config.get('staging_delay', 10)
                agent.on_edge = False
                agent.edge_u = None
                agent.edge_v = None
                agent.edge_eta = 0.0
                
                # Reset assistance state
                agent.assisting_person_id = None
        
        # Update agent position flags
        self._update_agent_positions()
        
        # ========================================
        # START FIRE
        # ========================================
        if fire_node is None:
            # Random room
            rooms = [n.nid for n in self.nodes.values() if n.ntype == "room"]
            if rooms:
                _ignite_node(self.nodes, self._rng.choice(rooms))
        elif fire_node != "none":
            # Specific node
            _ignite_node(self.nodes, fire_node)
        # If fire_node == "none", no fire is started (for testing)
        
        # ========================================
        # RETURN INITIAL OBSERVATION
        # ========================================
        return self.get_observation()
    
    # ============================================================================
    # OBSERVATION AND STATE ACCESS
    # ============================================================================
    
    def get_observation(self) -> Data:
        """
        Get current observation for RL agent.
        
        Returns partial observability view - people only visible in
        searched/occupied rooms.
        
        Returns:
            PyTorch Geometric Data object with graph structure
        """
        observation, _ = self.to_pytorch_geometric()
        return observation
    
    def get_state(self) -> Dict:
        """
        Get complete environment state (for evaluation/logging).
        
        Unlike get_observation(), this includes full ground truth
        (all people locations, even if not discovered).
        
        Returns:
            Dictionary with complete environment state
        """
        _, env_state = self.to_pytorch_geometric()
        return env_state
    
    # ============================================================================
    # ACTION SPACE AND EXECUTION
    # ============================================================================
    
    def get_valid_actions(self, agent_id: int) -> List[str]:
        """
        Get list of valid actions for a specific agent.
        
        Used for action masking in RL (prevents invalid actions).
        
        Args:
            agent_id: Agent to query
        
        Returns:
            List of action strings (e.g., ["search", "move_H1", "wait"])
        """
        if agent_id not in self.agents:
            return ["wait"]
        
        agent = self.agents[agent_id]
        
        # Can't act while searching/staging or moving
        if agent.searching or agent.on_edge:
            return ["wait"]
        
        valid_actions = []
        current = agent.node_id
        
        # Can search current room (if it's a room type)
        if self.nodes[current].ntype in self.config["sweep_node_types"]:
            valid_actions.append("search")
        
        # Can move to adjacent nodes
        for neighbor in self.G.neighbors(current):
            valid_actions.append(f"move_{neighbor}")
        
        # Can assist discovered people at current location
        for pid in self.nodes[current].people:
            person = self.people[pid]
            # Only assist if: alive, discovered, not already being helped
            if person.is_alive and person.seen and not person.being_assisted:
                valid_actions.append(f"assist_{pid}")
        
        # Can always wait
        valid_actions.append("wait")
        
        return valid_actions
    
    def get_agent_node_index(self, agent_id: int) -> Optional[int]:
        """
        Get the node index where agent is currently located.
        
        Used for indexing into graph observations.
        
        Args:
            agent_id: Agent to query
        
        Returns:
            Node index (integer) or None if agent doesn't exist
        """
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        node_ids = list(self.nodes.keys())
        
        try:
            return node_ids.index(agent.node_id)
        except ValueError:
            return None
    
    def do_action(self, actions: Dict[int, str]) -> Tuple[Data, float, bool, Dict]:
        """
        Execute agent actions and advance environment by one timestep.
        
        This is the main step function for the RL environment.
        
        EXECUTION ORDER:
        1. Parse and initiate agent actions
        2. Process agent searches (discoveries)
        3. Process agent movement (traversal)
        4. Spread fire and smoke
        5. Move civilians
        6. Degrade health
        7. Compute reward and check termination
        
        Args:
            actions: Dict mapping agent_id -> action_string
                    e.g., {0: "move_H1", 1: "search"}
        
        Returns:
            observation: Next state (PyG Data)
            reward: Scalar reward (currently 0.0 placeholder)
            done: Whether episode has terminated
            info: Dict with statistics and metadata
        """
        # ========================================
        # PHASE 1: INITIATE AGENT ACTIONS
        # ========================================
        for agent_id, action_str in actions.items():
            if agent_id not in self.agents:
                continue
            
            # Parse action and execute
            if action_str.startswith("move_"):
                # Move to adjacent node
                target = action_str[5:]  # Strip "move_" prefix
                self.move_agent(agent_id, target)
                
            elif action_str == "search":
                # Start searching current room
                self.start_search(agent_id)
                
            elif action_str.startswith("assist_"):
                # Start helping a person
                person_id = int(action_str[7:])  # Strip "assist_"
                self.assist_person(agent_id, person_id)
            
            # "wait" does nothing (agent idles)
        
        # ========================================
        # PHASE 2: ADVANCE ENVIRONMENT ONE TIMESTEP
        # ========================================
        self.step()
        
        # ========================================
        # PHASE 3: COMPUTE REWARD
        # ========================================
        # TODO: Implement reward function here
        # Example: reward = (people_found * 10) - (time_step * 0.1)
        reward = 0.0  # Placeholder
        
        # ========================================
        # PHASE 4: CHECK TERMINATION
        # ========================================
        done = self._is_done()
        
        # ========================================
        # PHASE 5: GET NEXT OBSERVATION
        # ========================================
        observation = self.get_observation()
        
        # ========================================
        # PHASE 6: COLLECT INFO
        # ========================================
        info = {
            "stats": self.get_statistics(),
            "is_success": self.is_sweep_complete(),
            "time_step": self.time_step,
        }
        
        return observation, reward, done, info
    
    def _is_done(self) -> bool:
        """
        Check if episode should terminate.
        
        Termination conditions:
        1. SUCCESS: All target rooms have been swept
        2. TIMEOUT: Maximum timesteps exceeded
        3. FAILURE: All people dead (optional early termination)
        
        Returns:
            True if episode should end, False otherwise
        """
        # Success condition: all rooms swept
        if self.is_sweep_complete():
            return True
        
        # Timeout condition
        if self.time_step >= self.config["max_steps"]:
            return True
        
        # Failure condition: everyone dead (optional)
        if self.people and all(not p.is_alive for p in self.people.values()):
            return True
        
        return False
    
    # ============================================================================
    # AGENT ACTION IMPLEMENTATIONS
    # ============================================================================
    
    def move_agent(self, agent_id: int, target_node_id: str) -> bool:
        """
        Move an agent to an adjacent node.
        
        REALISTIC MOVEMENT:
        - Agents take time to traverse edges (no teleportation)
        - Traversal time = edge_length / agent_speed
        - Agents in transit cannot act
        - If agent is assisting someone, person moves with them
        
        Args:
            agent_id: Agent to move
            target_node_id: Destination (must be adjacent to current location)
        
        Returns:
            True if movement initiated successfully, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Cannot move if already in transit or searching/staging
        if agent.on_edge or agent.searching:
            return False
        
        current_node = agent.node_id
        
        # Check adjacency
        if target_node_id not in self.G.neighbors(current_node):
            return False
        
        # Get edge properties
        edge_meta: EdgeMeta = self.G.edges[current_node, target_node_id]['meta']
        
        # Calculate traversal time
        # Time = distance / speed (in timesteps, where 1 timestep = 5 seconds)
        agent_speed = self.config['agent_speed']  # meters per second
        traversal_time = edge_meta.length / agent_speed  # seconds
        traversal_timesteps = traversal_time / 5.0  # convert to timesteps
        
        # Initiate movement
        agent.on_edge = True
        agent.edge_u = current_node
        agent.edge_v = target_node_id
        agent.edge_eta = max(1.0, traversal_timesteps)  # Minimum 1 timestep
        
        # If assisting someone, they move together
        if agent.assisting_person_id is not None:
            person = self.people.get(agent.assisting_person_id)
            if person and person.is_alive and not person.rescued:
                # Person follows agent
                person.on_edge = True
                person.edge_u = current_node
                person.edge_v = target_node_id
                person.edge_eta = agent.edge_eta  # Same arrival time
                
                # Track edge congestion
                self._edge_inc(current_node, target_node_id)
        
        return True
    
    def assist_person(self, agent_id: int, person_id: int) -> bool:
        """
        Agent starts assisting (escorting) a discovered person.
        
        ASSISTANCE MECHANICS:
        - Person's speed increases by 1.8x
        - Person no longer panicked
        - Person moves with agent automatically
        - Agent can only assist one person at a time
        - Person must be discovered (seen) first
        
        Args:
            agent_id: Firefighter agent
            person_id: Person to assist
        
        Returns:
            True if assistance started, False otherwise
        """
        # Validate IDs
        if agent_id not in self.agents or person_id not in self.people:
            return False
        
        agent = self.agents[agent_id]
        person = self.people[person_id]
        
        # Check preconditions
        if person.node_id != agent.node_id:
            # Person not at agent's location
            return False
        
        if not person.is_alive or person.being_assisted or person.rescued:
            # Person already helped, dead, or rescued
            return False
        
        if not person.seen:
            # Person must be discovered first
            return False
        
        # Start assistance
        agent.assisting_person_id = person_id
        person.being_assisted = True
        person.assisting_agent_id = agent_id
        person.panicked = False  # Agent calms them down
        
        return True
    
    def start_search(self, agent_id: int) -> bool:
        """
        Begin searching the current node for people.
        
        SEARCH MECHANICS:
        - Search time proportional to room area
        - Formula: time = area × search_time_per_sqm
        - 16 sqm room takes ~8 timesteps (40 seconds)
        - While searching, agent cannot move or act
        - Search reveals all people in the room
        - Room marked as "swept" when complete
        
        Args:
            agent_id: Agent who will search
        
        Returns:
            True if search started, False otherwise
        """
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Cannot search if already searching/staging or moving
        if agent.searching or agent.on_edge:
            return False
        
        # Get search time based on room area (realistic)
        node = self.nodes[agent.node_id]
        search_time_per_sqm = self.config.get('search_time_per_sqm', 0.5)
        
        # Calculate search time in timesteps
        # Each timestep = 5 seconds, so 0.5 sec/sqm → 0.1 timesteps/sqm
        search_time = node.area * search_time_per_sqm / 5.0
        search_timesteps = int(max(1, search_time))  # At least 1 timestep
        
        # Initiate search
        agent.searching = True
        agent.search_timer = search_timesteps
        
        return True
    
    # ============================================================================
    # AGENT PROCESSING (INTERNAL)
    # ============================================================================
    
    def process_searches(self) -> None:
        """
        Process ongoing searches and complete them when timers expire.
        
        SEARCH COMPLETION:
        1. Decrement all active search timers
        2. When timer reaches 0:
           a. Mark room as swept
           b. Discover all people in room (mark seen=True)
           c. Compute evacuation distance for each person
           d. Update statistics
        
        NOTE: People are NOT rescued here - only discovered.
        Rescue happens when they physically reach an exit.
        """
        for agent in self.agents.values():
            # Skip agents not searching
            if not agent.searching:
                continue
            
            # Decrement timer
            agent.search_timer -= 1
            self.stats["total_search_time"] += 1
            
            # Check if search complete
            if agent.search_timer <= 0:
                # Search finished
                agent.searching = False
                node = self.nodes[agent.node_id]
                
                # Mark node as swept (if it's a target type)
                if node.ntype in self.config.get("sweep_node_types", {"room"}):
                    if not node.swept:
                        node.swept = True
                        self.stats["nodes_swept"] += 1
                
                # Discover all people in this room
                for pid in node.people:
                    person = self.people[pid]
                    
                    # Only count first-time discoveries
                    if not person.seen:
                        person.seen = True
                        self.stats["people_found"] += 1
                        
                        # Compute shortest distance to exit at discovery
                        # This is used for evaluation metrics
                        person.evac_distance = _shortest_distance_to_any_exit(
                            self, person.node_id
                        )
    
    def process_agent_movement(self) -> None:
        """
        Process agents traversing edges.
        
        MOVEMENT COMPLETION:
        1. Decrement ETAs for all agents on edges
        2. When ETA reaches 0:
           a. Agent arrives at destination
           b. Update agent position flags
           c. Auto-sweep if applicable (hallways/exits)
           d. Person arrives too if being assisted
        """
        for agent in self.agents.values():
            # Skip agents not moving
            if not agent.on_edge:
                continue
            
            # Decrement ETA
            agent.edge_eta -= 1.0
            
            # Check if arrived
            if agent.edge_eta <= 0:
                # Agent arrives at destination
                old_node = agent.node_id
                new_node = agent.edge_v
                
                # Update agent state
                agent.on_edge = False
                agent.node_id = new_node
                agent.edge_u = None
                agent.edge_v = None
                agent.edge_eta = 0.0
                
                # Update agent position flags on all nodes
                self._update_agent_positions()
                
                # Auto-sweep certain node types (hallways, exits)
                self._maybe_auto_sweep_current_node(self.nodes[new_node])
    
    # ============================================================================
    # OBSERVATION UPDATES (PARTIAL OBSERVABILITY)
    # ============================================================================
    
    def _update_agent_positions(self) -> None:
        """
        Update agent_here flags for all nodes.
        
        This is used for:
        1. Node features (agents can see people at their location)
        2. Partial observability (only occupied nodes reveal people)
        """
        # Clear all flags
        for node in self.nodes.values():
            node.agent_here = False
        
        # Set flags for occupied nodes
        # NOTE: Agents in transit (on_edge) don't count as "here"
        for agent in self.agents.values():
            if not agent.on_edge:
                self.nodes[agent.node_id].agent_here = True
    
    def _update_observations(self) -> None:
        """
        Update observable people counts based on agent presence.
        
        PARTIAL OBSERVABILITY LOGIC:
        - By default, people are NOT observable (obs_people_count = 0)
        - Agent presence at a node reveals people there
        - This creates exploration incentive for agents
        - People become "seen" when agent is present
        """
        # Reset all observations to hidden
        for node in self.nodes.values():
            node.obs_people_count = 0
            node.obs_avg_hp = 0.0
        
        # Reveal people at agent-occupied nodes
        for agent in self.agents.values():
            # Can't observe while in transit
            if agent.on_edge:
                continue
            
            node = self.nodes[agent.node_id]
            
            # Count living people
            alive_people = [
                self.people[pid] for pid in node.people 
                if self.people[pid].is_alive
            ]
            count = len(alive_people)
            
            if count > 0:
                # Observable count (capped at 3 for normalization)
                node.obs_people_count = min(count, 3)
                
                # Average HP (normalized to 0-1)
                avg_hp = sum(p.hp for p in alive_people) / count
                node.obs_avg_hp = avg_hp / 100.0
                
                # Mark all people as seen (discovered)
                for pid in node.people:
                    self.people[pid].seen = True
    
    def _compute_fire_distances(self) -> None:
        """
        Compute shortest path distance from each node to nearest fire.
        
        This is a precomputed feature used in node feature vectors.
        Uses NetworkX's multi-source Dijkstra for efficiency.
        
        Distance is normalized:
        - 0.0 = on fire
        - 1.0 = 10+ meters away (capped)
        """
        # Find all burning nodes
        fire_nodes = [n.nid for n in self.nodes.values() if n.on_fire]
        
        if not fire_nodes:
            # No fire yet - all nodes maximally distant
            for node in self.nodes.values():
                node.dist_to_fire_norm = 1.0
            return
        
        # Multi-source shortest path from all fire nodes
        # Weight edges by physical length
        distances = nx.multi_source_dijkstra_path_length(
            self.G, 
            fire_nodes,
            weight=lambda u, v, d: d["meta"].length
        )
        
        # Normalize distances (divide by 10m, cap at 1.0)
        for node in self.nodes.values():
            raw_dist = distances.get(node.nid, 10.0)  # Default 10m if unreachable
            node.dist_to_fire_norm = min(raw_dist / 10.0, 1.0)
    
    # ============================================================================
    # FEATURE EXTRACTION FOR GNN
    # ============================================================================
    
    def get_node_features(self, node: NodeMeta) -> np.ndarray:
        """
        Construct the 11-dimensional feature vector for a node.
        
        Feature breakdown (F = 11):
        [0:4]   One-hot node type (room/hall/exit/floor)
        [4]     Fire indicator (0 or 1)
        [5]     Smoke indicator (0 or 1)
        [6]     Length normalized by 10m
        [7]     Observable people count (0-3, normalized)
        [8]     Observable average HP (0-1)
        [9]     Agent presence indicator (0 or 1)
        [10]    Distance to nearest fire (0-1, normalized)
        
        Args:
            node: Node to featurize
        
        Returns:
            11D numpy array of float32 features
        """
        # Features 0-3: One-hot encoding of node type (4 types: room/hall/exit/floor)
        one_hot = np.zeros(len(NODE_TYPES), dtype=np.float32)
        one_hot[NODE_TYPES[node.ntype]] = 1.0
        
        # Feature 3: Fire indicator
        fire = 1.0 if node.on_fire else 0.0
        
        # Feature 4: Smoke indicator
        smoke = 1.0 if node.smoky else 0.0
        
        # Feature 5: Normalized length (meters / 10)
        length_norm = node.length / 10.0
        
        # Feature 6: Normalized people count (observable only, capped at 3)
        people_count_norm = min(node.obs_people_count, 3) / 3.0
        
        # Feature 7: Normalized average HP (observable only)
        avg_hp_norm = node.obs_avg_hp  # Already 0-1
        
        # Feature 8: Agent presence
        agent_here = 1.0 if node.agent_here else 0.0
        
        # Feature 9: Normalized distance to fire
        dist_fire = node.dist_to_fire_norm
        
        # Assemble feature vector
        features = np.array([
            *one_hot,           # [0:4] one-hot for 4 node types
            fire,               # [4]
            smoke,              # [5]
            length_norm,        # [6]
            people_count_norm,  # [7]
            avg_hp_norm,        # [8]
            agent_here,         # [9]
            dist_fire           # [10]
        ], dtype=np.float32)
        
        assert features.shape[0] == FEATURE_DIM, \
            f"Feature dimension mismatch: {features.shape[0]} != {FEATURE_DIM}"
        
        return features
    
    def to_pytorch_geometric(self) -> Tuple[Data, Dict]:
        """
        Convert current environment state to PyTorch Geometric Data object.
        
        This creates a graph representation suitable for Graph Neural Networks.
        
        GRAPH STRUCTURE:
        - Nodes: Rooms, hallways, exits
        - Edges: Doors, passages (bidirectional)
        - Node features: 10D vectors (see get_node_features)
        - Edge features: 5D vectors (width, length, slope, door, fire_door)
        
        Returns:
            data: PyG Data object with observation (partial observability)
            env_state: Dict with complete ground truth (for evaluation)
        """
        # Update observations and distances before featurizing
        self._update_observations()
        self._compute_fire_distances()
        
        # ========================================
        # CREATE NODE INDEX MAPPING
        # ========================================
        node_ids = list(self.nodes.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # ========================================
        # CONSTRUCT NODE FEATURE MATRIX
        # ========================================
        node_features = []
        for nid in node_ids:
            node = self.nodes[nid]
            features = self.get_node_features(node)
            node_features.append(features)
        
        X = np.stack(node_features, axis=0)  # Shape: [N, 10]
        
        # ========================================
        # CONSTRUCT EDGE INDEX AND FEATURES
        # ========================================
        edge_src, edge_dst, edge_features = [], [], []
        
        for u, v, data in self.G.edges(data=True):
            edge_meta: EdgeMeta = data["meta"]
            
            # Get node indices
            i, j = node_to_idx[u], node_to_idx[v]
            
            # Edge features: [width, length, slope, door, fire_door]
            edge_feat = [
                edge_meta.width,
                edge_meta.length,
                edge_meta.slope,
                float(edge_meta.door),
                float(edge_meta.fire_door)
            ]
            
            # PyG expects directed edges, so add both directions
            # (our building graph is undirected)
            
            # Edge u -> v
            edge_src.append(i)
            edge_dst.append(j)
            edge_features.append(edge_feat)
            
            # Edge v -> u
            edge_src.append(j)
            edge_dst.append(i)
            edge_features.append(edge_feat)
        
        # ========================================
        # CONVERT TO PYTORCH TENSORS
        # ========================================
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        x = torch.tensor(X, dtype=torch.float32)
        
        # ========================================
        # CREATE PYTORCH GEOMETRIC DATA OBJECT
        # ========================================
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data.nid = node_ids  # Store node IDs for debugging
        
        # ========================================
        # GROUND TRUTH STATE (for evaluation)
        # ========================================
        env_state = {
            "nodes": self.nodes,
            "people": self.people,
            "agents": self.agents,
            "time_step": self.time_step,
            "stats": self.stats
        }
        
        return data, env_state
    
    # ============================================================================
    # MAIN SIMULATION STEP
    # ============================================================================
    
    def step(self) -> None:
        """
        Advance environment by one timestep (~5 seconds).
        
        EXECUTION ORDER (critical for consistency):
        1. Process agent searches (discoveries)
        2. Process agent movement (arrivals)
        3. Spread fire and smoke (hazard dynamics)
        4. Move civilians (evacuation behavior)
        5. Degrade health (damage from hazards)
        6. Increment time counter
        
        This order ensures:
        - Agents complete actions before hazards spread
        - Hazards spread before people move (realistic danger response)
        - Health degradation happens after movement (people may escape)
        """
        # ========================================
        # PHASE 1: AGENT SEARCH COMPLETION
        # ========================================
        # Complete ongoing searches, discover people
        self.process_searches()
        
        # ========================================
        # PHASE 2: AGENT MOVEMENT COMPLETION
        # ========================================
        # Agents in transit arrive at destinations
        self.process_agent_movement()
        
        # ========================================
        # PHASE 3: HAZARD DYNAMICS
        # ========================================
        # Fire spreads to adjacent rooms, smoke fills spaces
        self.fire_spread_counter = _spread_hazards(
            self.G,
            self.nodes,
            self.config["fire_spread_prob"],
            self.fire_spread_counter,
            self.config["fire_spread_delay"]
        )
        
        # ========================================
        # PHASE 4: CIVILIAN MOVEMENT
        # ========================================
        # People navigate toward exits (if aware)
        _move_civilians(self)
        
        # ========================================
        # PHASE 5: HEALTH DEGRADATION
        # ========================================
        # People take damage from fire/smoke
        _degrade_health(
            self.nodes,
            self.people,
            self.config["hp_loss_fire"],
            self.config["hp_loss_smoke"],
            self.time_step,
            verbose=False  # Set True for death notifications
        )
        
        # ========================================
        # PHASE 6: INCREMENT TIME
        # ========================================
        self.time_step += 1
    
    # ============================================================================
    # COMPLETION CHECKING
    # ============================================================================
    
    def is_sweep_complete(self) -> bool:
        """
        Check if building sweep is complete.
        
        A sweep is complete when all target rooms have been searched.
        Target rooms are defined by sweep_node_types config (typically "room").
        
        Returns:
            True if all target rooms are swept, False otherwise
        """
        # Get sweep target types from config
        sweep_kinds = self.config.get("sweep_node_types", {"room"})
        
        # Find all target nodes
        targets = [n for n in self.nodes.values() if n.ntype in sweep_kinds]
        
        # Complete if all targets are swept
        # (and there's at least one target - avoid vacuous truth)
        return len(targets) > 0 and all(n.swept for n in targets)
    
    def _maybe_auto_sweep_current_node(self, agent_node: NodeMeta) -> None:
        """
        Auto-sweep certain node types when agent enters.
        
        Hallways and exits don't need thorough search - just passing through
        them counts as swept. This is realistic: firefighters can visually
        clear open spaces quickly.
        
        Args:
            agent_node: Node where agent just arrived
        """
        # Check if auto-sweep is enabled
        if not self.config.get("auto_sweep_on_visit", True):
            return
        
        # Check if this node type is auto-swept
        auto_types = self.config.get("auto_sweep_types", {"hall", "exit"})
        if agent_node.ntype in auto_types:
            # Mark as swept if not already
            if not agent_node.swept:
                agent_node.swept = True
                self.stats["nodes_swept"] += 1
    
    # ============================================================================
    # EDGE CONGESTION TRACKING
    # ============================================================================
    
    def _edge_inc(self, u: str, v: str) -> None:
        """
        Increment edge load (person enters edge).
        
        Used for dynamic congestion-based pathfinding.
        """
        key = (u, v)
        self._edge_load[key] = self._edge_load.get(key, 0) + 1
    
    def _edge_dec(self, u: str, v: str) -> None:
        """
        Decrement edge load (person exits edge).
        
        Used for dynamic congestion-based pathfinding.
        """
        key = (u, v)
        if key in self._edge_load:
            self._edge_load[key] = max(0, self._edge_load[key] - 1)
    
    # ============================================================================
    # STATISTICS AND LOGGING
    # ============================================================================
    
    def get_statistics(self) -> Dict:
        """
        Get current simulation statistics.
        
        Returns:
            Dictionary with statistics:
            - people_rescued: Number evacuated to exits
            - people_found: Number discovered by agents
            - nodes_swept: Number of rooms searched
            - total_search_time: Total timesteps spent searching
            - time_step: Current timestep
            - sweep_complete: Whether all rooms are swept
            - total_people: Total number of people
            - people_alive: Number still alive
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
        """
        Print detailed environment status for debugging.
        
        Shows:
        - Current time (timesteps and approximate seconds)
        - Agent states (location, activity, assistance)
        - People states (location, HP, movement, status)
        - Node states (hazards, swept status, occupancy)
        - Statistics summary
        """
        print(f"\n{'='*70}")
        print(f"Time Step: {self.time_step} (~{self.time_step * 5} seconds = {(self.time_step * 5) / 60:.1f} minutes)")
        print(f"{'='*70}")
        
        # ========================================
        # AGENT STATUS
        # ========================================
        print("\nAgents:")
        for agent in self.agents.values():
            # Determine current activity
            if agent.on_edge:
                status = f"MOVING {agent.edge_u} → {agent.edge_v} (ETA: {agent.edge_eta:.1f})"
            elif agent.searching:
                if self.time_step == 0:
                    status = f"STAGING (preparing, {agent.search_timer} steps left)"
                else:
                    status = f"SEARCHING ({agent.search_timer} steps left)"
            else:
                status = "IDLE"
            
            # Check if assisting someone
            assist_str = ""
            if agent.assisting_person_id is not None:
                assist_str = f" [Assisting P{agent.assisting_person_id}]"
            
            print(f"  Agent {agent.agent_id}: {agent.node_id} ({status}){assist_str}")
        
        # ========================================
        # PEOPLE STATUS
        # ========================================
        print("\nPeople:")
        for person in self.people.values():
            status_parts = []
            
            # Primary status
            if not person.is_alive:
                status_parts.append("DEAD ☠")
            elif person.rescued:
                status_parts.append("RESCUED ✓")
            else:
                # Movement status
                if person.on_edge:
                    status_parts.append(f"MOVING {person.edge_u} → {person.edge_v}")
                
                # Assistance status
                if person.being_assisted:
                    status_parts.append(f"ASSISTED by A{person.assisting_agent_id}")
                
                # Emotional status
                if person.panicked:
                    status_parts.append("PANICKED")
                
                # Discovery status
                if not person.seen:
                    status_parts.append("HIDDEN (undiscovered)")
                
                # Awareness status
                if person.awareness_timer > 0:
                    status_parts.append(f"UNAWARE ({person.awareness_timer} steps)")
            
            status = ", ".join(status_parts) if status_parts else "IDLE"
            
            print(f"  Person {person.pid} ({person.mobility}, age {person.age}): "
                  f"{person.node_id} | HP={person.hp:.1f}/100 | {status}")
        
        # ========================================
        # NODE STATUS
        # ========================================
        print("\nNodes:")
        for node in self.nodes.values():
            # Hazard status
            hazards = []
            if node.on_fire:
                intensity = getattr(node, 'fire_intensity', 1.0)
                hazards.append(f"FIRE({intensity:.1f})")
            if node.smoky:
                hazards.append("SMOKE")
            hazard_str = ", ".join(hazards) if hazards else "clear"
            
            # Sweep status
            swept_str = "SWEPT ✓" if node.swept else "unswept"
            
            # Occupancy
            people_str = f"{len(node.people)} people" if node.people else "empty"
            
            print(f"  {node.nid:12s} [{node.ntype:4s}]: {hazard_str:18s} | "
                  f"{swept_str:10s} | {people_str}")
        
        # ========================================
        # STATISTICS SUMMARY
        # ========================================
        print("\nStatistics:")
        for key, value in self.get_statistics().items():
            print(f"  {key}: {value}")
        print()