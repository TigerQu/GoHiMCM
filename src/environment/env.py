from .occupants import move_civilians as _move_civilians
from .occupants import _shortest_distance_to_any_exit as _shortest_dist_exit

from typing import Dict, List, Tuple, Optional
import math, random, heapq
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


from .config import DEFAULT_CONFIG, FEATURE_DIM, NODE_TYPES
from .entities import Person, NodeMeta, EdgeMeta, Agent
from .hazards import ignite_node as _ignite_node
from .hazards import spread_hazards as _spread_hazards
from .hazards import degrade_health as _degrade_health


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
        self.config.setdefault("sweep_node_types", {"room"})    # only rooms must be searched
        self.config.setdefault("auto_sweep_types", {"hall", "exit"})  # halls/exits auto-sweep
        self.config.setdefault("auto_sweep_on_visit", True)     # stepping onto them marks swept
        self.config.setdefault("required_sweeps", 2)            # each room must be swept twice
        self.config.setdefault("enable_agent_comm", True)       # agents share searched node info
        self.config.setdefault("comm_delay", 0)                 # communication delay in time steps
        
        # Building topology
        self.G = nx.Graph()                    # NetworkX graph for topology
        self.nodes: Dict[str, NodeMeta] = {}   # Node metadata by ID
        
        # Entities
        self.people: Dict[int, Person] = {}    # People by ID
        self.agents: Dict[int, Agent] = {}     # Agents by ID
        
        # Communication system
        self.pending_messages: List[Dict] = []  # Messages with delivery_time
        
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

        # Reward tracking
        self._last_people_found = 0
        self._last_nodes_swept = 0
        self._last_people_alive = 0
        self._last_people_rescued = 0
        
        # For reset
        self._initial_agent_positions: Dict[int, str] = {}
        self._initial_people_state: List[Tuple[str, int, str, float]] = []
        
        # Edge load tracking (for congestion)
        self._edge_load: Dict[Tuple[str, str], int] = {}
        
        # RNG for determinism
        self._rng = random.Random()
        self._np_rng = np.random.RandomState()
        
        
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
        """Spawn person at location."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        
        pid = len(self.people)
        speed_map = {
            'adult': self.config['adult_speed'],
            'child': self.config['child_speed'],
            'limited': self.config['limited_speed']
        }
        v_class = speed_map.get(mobility, self.config['adult_speed'])
        
        # Sample awareness delay
        delay = max(0, int(self._np_rng.normal(
            self.config['awareness_delay_mean'],
            self.config['awareness_delay_std']
        )))
        
        person = Person(
            pid=pid, age=age, mobility=mobility, hp=hp,
            node_id=node_id, v_class=v_class, awareness_timer=delay
        )
        self.people[pid] = person
        self.nodes[node_id].people.append(pid)
        
        # Store for reset
        self._initial_people_state.append((node_id, age, mobility, hp))
        
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
        self._initial_agent_positions[agent_id] = node_id     
        # Update node states
        self._update_agent_positions()

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for deterministic behavior."""
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.RandomState(seed)
            random.seed(seed)
            np.random.seed(seed)

    #all of the functions
    def reset(self, fire_node: Optional[str] = None, seed: Optional[int] = None) -> Data:
        """
        Reset environment for a new episode (PPO requirement).
        
        Args:
            fire_node: Node to ignite (None = random room, "none" = no fire)
        
        Returns:
            Initial observation (PyG Data object)
        """
        if seed is not None:
            self.seed(seed)

        # Reset time
        self.time_step = 0
        self.fire_spread_counter = 0
        self.pending_messages = []
        
        # Reset statistics
        self.stats = {
            "people_rescued": 0,
            "people_found": 0,
            "nodes_swept": 0,
            "total_search_time": 0,
        }
        
        # Reset reward tracking
        self._last_people_found = 0
        self._last_nodes_swept = 0
        self._last_people_alive = len(self.people)
        self._last_people_rescued = 0

        # Reset edge load
        self._edge_load.clear()
        
        # Reset all nodes
        for node in self.nodes.values():
            node.on_fire = False
            node.smoky = False
            node.sweep_count = 0
            node.agent_here = False
            node.obs_people_count = 0
            node.obs_avg_hp = 0.0
            node.people = []  # Clear people lists
        
        # Reset all people to initial state
        for pid, (node_id, age, mobility, hp) in enumerate(self._initial_people_state):
            person = self.people[pid]
            person.hp = hp
            person.seen = False
            person.rescued = False
            person.node_id = node_id

            # Re-sample awareness delay
            delay = max(0, int(self._np_rng.normal(
                self.config['awareness_delay_mean'],
                self.config['awareness_delay_std']
            )))
            
            person.awareness_timer = delay
            person.on_edge = False
            person.edge_u = None
            person.edge_v = None
            person.edge_eta = 0.0
            person.evac_distance = None
            
            # Re-add to node's people list
            self.nodes[node_id].people.append(pid)
        
        # Reset agents to initial positions
        for agent_id, start_pos in self._initial_agent_positions.items():
            if agent_id in self.agents:
                self.agents[agent_id].node_id = start_pos
                self.agents[agent_id].searching = False
                self.agents[agent_id].search_timer = 0
        
        self._update_agent_positions()
        
        # Start fire
        if fire_node is None:
            # Random room
            rooms = [n.nid for n in self.nodes.values() if n.ntype == "room"]
            if rooms:
                _ignite_node(self.nodes, random.choice(rooms))
        elif fire_node != "none":
            _ignite_node(self.nodes, fire_node)
        
        # Get initial observation
        observation = self.get_observation()
        return observation
    
    
    def get_observation(self) -> Data:
        """
        Get current observation (PPO requirement).
        
        Returns:
            PyG Data object with current graph state
        """
        observation, _ = self.to_pytorch_geometric()
        return observation
    
    
    def get_state(self) -> Dict:
        """
        Get full environment state (PPO requirement).
        
        Returns:
            Dictionary with complete environment state
        """
        _, env_state = self.to_pytorch_geometric()
        return env_state
    
    
    def get_valid_actions(self, agent_id: int) -> List[str]:
        """Get valid actions for agent (action masking)."""
        if agent_id not in self.agents:
            return ["wait"]
        
        agent = self.agents[agent_id]
        if agent.searching:
            return ["wait"]
        
        valid_actions = ["search"]
        current = agent.node_id
        for neighbor in self.G.neighbors(current):
            valid_actions.append(f"move_{neighbor}")
        valid_actions.append("wait")
        
        return valid_actions
    
    def get_agent_node_index(self, agent_id: int) -> Optional[int]:
        """Get node index where agent is located."""
        if agent_id not in self.agents:
            return None
        agent = self.agents[agent_id]
        node_ids = list(self.nodes.keys())
        try:
            return node_ids.index(agent.node_id)
        except ValueError:
            return None
    
    def do_action(self, actions: Dict[int, str]):
        """
        Execute actions and return (observation, reward, done, info) (PPO requirement).

        NOTE: reward kept at 0.0 here (no-reward test), matching your smoke test.
        """
        # Parse and execute agent actions
        for agent_id, action_str in actions.items():
            if agent_id not in self.agents:
                continue
            
            if action_str.startswith("move_"):
                target = action_str[5:]  # Remove "move_" prefix
                self.move_agent(agent_id, target)
            elif action_str == "search":
                self.start_search(agent_id)
            # "wait" does nothing
        
        # Advance environment by one step
        self.step()
        
        # reward placeholder
        reward = 0.0
        
        # Check if episode is done
        done = self._is_done()
        
        # Get next observation
        observation = self.get_observation()
        
        # Collect info
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
        1. All rooms swept (success)
        2. Max steps reached (timeout)
        3. All people dead (optional early termination)
        """
        # Success: all rooms swept
        if self.is_sweep_complete():
            return True
        
        # Timeout
        if self.time_step >= self.config["max_steps"]:
            return True
        
        # All people dead (optional)
        if self.people and all(not p.is_alive for p in self.people.values()):
            return True
        
        return False
    

    # fire and smoke occur via hazards.py (ignite_node, spread_hazards, degrade_health)
    
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
            return False
        
        # Move agent
        agent.node_id = target_node_id
        agent.searching = False  # Stop any ongoing search
        agent.search_timer = 0
        
        self._update_agent_positions()
        self._maybe_auto_sweep_current_node(self.nodes[target_node_id])
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
        return True
    
    
    def process_searches(self) -> None:
        """
        Process ongoing searches and complete them when timers expire.
        
        Logic:
        - Decrement search timers
        - When timer reaches 0:
          1. Increment sweep_count for the node
          2. Add to agent's searched_nodes
          3. Broadcast to other agents (if comm enabled)
          4. Reveal all people in node
          5. Update statistics
        """
        required = self.config.get("required_sweeps", 2)
        
        for agent in self.agents.values():
            if not agent.searching:
               continue

            agent.search_timer -= 1
            self.stats["total_search_time"] += 1

            if agent.search_timer <= 0:
               agent.searching = False
               node = self.nodes[agent.node_id]
               
               if node.ntype in self.config.get("sweep_node_types", {"room"}):
                # Check if this specific agent has already searched this node
                if node.nid not in agent.searched_nodes:
                    # This agent hasn't searched here before, so it counts
                    if node.sweep_count < required:
                        node.sweep_count += 1
                        self.stats["nodes_swept"] += 1
                    
                    # Record that this agent searched this node
                    agent.searched_nodes.add(node.nid)
                    
                    # Broadcast to other agents if communication enabled
                    if self.config.get("enable_agent_comm", True):
                        self._broadcast_search_completion(agent.agent_id, node.nid)

               # --- DISCOVERY LOGIC + EVAC DISTANCE ---
               for pid in node.people:
                    person = self.people[pid]

                    first_time_seen = not person.seen
                    if first_time_seen:
                        person.seen = True
                        self.stats["people_found"] += 1

                        # Compute shortest distance to any exit at discovery time
                        person.evac_distance = _shortest_dist_exit(self, person.node_id)

                        # If still alive at discovery, count as evacuated success immediately
                        if person.is_alive and not person.rescued:
                            person.rescued = True
                            self.stats["people_rescued"] += 1

    
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
        [0:4]   One-hot node type (room/hall/exit/floor)
        [4]     Fire indicator (0 or 1)
        [5]     Smoke indicator (0 or 1)
        [6]     Length normalized by 10
        [7]     People count normalized (capped at 3, divided by 3)
        [8]     Average HP normalized (divided by 100)
        [9]     Agent presence indicator (0 or 1)
        [10]    Distance to fire normalized (divided by 10, capped at 1)
        """
        import numpy as np  # local import to avoid circulars in some IDEs
        # Features 0-3: One-hot encoding of node type (4 types)
        one_hot = np.zeros(len(NODE_TYPES), dtype=np.float32)
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
            *one_hot,           # [0:4] one-hot for 4 node types
            fire,               # [4]
            smoke,              # [5]
            length_norm,        # [6]
            people_count_norm,  # [7]
            avg_hp_norm,        # [8]
            agent_here,         # [9]
            dist_fire           # [10]
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
        import numpy as np
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
        self.fire_spread_counter = _spread_hazards(
            self.G, self.nodes,
            self.config["fire_spread_prob"],
            self.fire_spread_counter,
            self.config["fire_spread_delay"]
        )
        # Move civilians
        _move_civilians(self)

        
        # Health degradation
        _degrade_health(
            self.nodes, self.people,
            self.config["hp_loss_fire"], self.config["hp_loss_smoke"],
            self.time_step, verbose=False
        )

        # Advance time
        self.time_step += 1
        
        # Deliver messages that have reached their delivery time
        self._deliver_pending_messages()
    
    
    def is_sweep_complete(self) -> bool:
        sweep_kinds = self.config.get("sweep_node_types", {"room"})
        required = self.config.get("required_sweeps", 2)
        targets = [n for n in self.nodes.values() if n.ntype in sweep_kinds]
        # If there are zero targets, define completion as False (or True if you prefer)
        return len(targets) > 0 and all(n.sweep_count >= required for n in targets)
    

    
    def _maybe_auto_sweep_current_node(self, agent_node: "NodeMeta") -> None:
        if not self.config.get("auto_sweep_on_visit", True):
            return
        required = self.config.get("required_sweeps", 2)
        if agent_node.ntype in self.config.get("auto_sweep_types", {"hall", "exit"}):
            if agent_node.sweep_count < required:
               agent_node.sweep_count += 1
               self.stats["nodes_swept"] += 1

    def _broadcast_search_completion(self, sender_id: int, node_id: str) -> None:
        """
        Broadcast search completion to all other agents with optional delay.
        
        Args:
            sender_id: Agent who completed the search
            node_id: Node that was searched
        """
        delay = self.config.get("comm_delay", 0)
        delivery_time = self.time_step + delay
        
        message = {
            "type": "search_complete",
            "sender": sender_id,
            "node_id": node_id,
            "timestamp": self.time_step,
            "delivery_time": delivery_time
        }
        
        # Add to pending messages queue
        for agent_id in self.agents.keys():
            if agent_id != sender_id:
                self.pending_messages.append({
                    **message,
                    "recipient": agent_id
                })

    def _deliver_pending_messages(self) -> None:
        """
        Deliver messages that have reached their delivery time.
        """
        # Find messages ready for delivery
        ready_messages = [msg for msg in self.pending_messages 
                         if msg["delivery_time"] <= self.time_step]
        
        # Deliver them
        for msg in ready_messages:
            recipient_id = msg["recipient"]
            if recipient_id in self.agents:
                agent = self.agents[recipient_id]
                agent.message_buffer.append(msg)
                agent.known_swept_nodes.add(msg["node_id"])
        
        # Remove delivered messages
        self.pending_messages = [msg for msg in self.pending_messages 
                                if msg["delivery_time"] > self.time_step]

    def get_agent_known_sweeps(self, agent_id: int) -> set:
        """
        Get all nodes known to be swept by this agent (personal + communicated).
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Set of node IDs known to be swept
        """
        if agent_id not in self.agents:
            return set()
        agent = self.agents[agent_id]
        return agent.searched_nodes | agent.known_swept_nodes


    
    
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
            
            swept_str = f"swept\u00d7{node.sweep_count}" if node.sweep_count > 0 else "unswept"
            people_str = f"{len(node.people)} people" if node.people else "empty"
            
            print(f"  {node.nid:6s} [{node.ntype:4s}]: {hazard_str:12s} | {swept_str:8s} | {people_str}")
        
        # Statistics
        print("\nStatistics:")
        for key, value in self.get_statistics().items():
            print(f"  {key}: {value}")
        print()
