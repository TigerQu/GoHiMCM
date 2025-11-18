"""
Improved reward shaping to fix pathological agent loops.

===== KEY IMPROVEMENTS =====
1. First-visit coverage bonus (no repeated reward for revisiting)
2. Edge backtrack penalty (punish A↔B oscillation)
3. Potential-based shaping toward unswept nodes (no reward cycles)
4. WAIT action penalty (make waiting truly undesirable)
5. Distance-based dense rewards to guide exploration

This addresses the catastrophic failure mode where agents loop between 2 nodes.
"""

import torch
import networkx as nx
from typing import Dict, Any, Set, Tuple, List, Optional


class ImprovedRewardShaper:
    """
    Enhanced reward shaper that prevents agent loops and encourages exploration.
    
    Fixes the pathological behavior where agents oscillate between 2 nodes.
    """
    
    def __init__(
        self,
        scenario: str = "office",
        # Coverage rewards
        weight_first_visit: float = 100.0,    # HUGE bonus for first visit to node
        weight_coverage: float = 10.0,         # Standard coverage (already swept nodes)
        
        # Rescue rewards
        weight_rescue: float = 200.0,          # HUGE bonus for rescue
        weight_hp_loss: float = 0.001,         # Minimal HP penalty
        
        # Anti-loop penalties
        weight_backtrack: float = -10.0,       # Penalty for A↔B oscillation
        weight_wait: float = -0.1,             # Penalty for WAIT action
        weight_time: float = -0.01,            # Small time cost per step
        
        # Exploration bonuses
        weight_potential: float = 5.0,         # Potential-based shaping
        weight_redundancy: float = 50.0,       # Redundancy bonus for high-risk rooms
        
        # Hyperparameters
        backtrack_window: int = 5,             # How many recent edges to check
        gamma: float = 0.99,                   # Discount for potential shaping
    ):
        """
        Initialize improved reward shaper.
        
        Args:
            scenario: Building scenario ('office', 'daycare', 'warehouse')
            weight_first_visit: Bonus for visiting unswept node for first time
            weight_coverage: Reward for sweeping already-visited nodes
            weight_rescue: Reward for rescuing people
            weight_hp_loss: Penalty for civilian HP loss
            weight_backtrack: Penalty for traversing same edge within window
            weight_wait: Penalty for WAIT action
            weight_time: Penalty per timestep
            weight_potential: Weight for potential-based shaping
            weight_redundancy: Bonus for high-risk room redundancy
            backtrack_window: Number of recent edges to track for backtrack penalty
            gamma: Discount factor for potential shaping
        """
        self.scenario = scenario
        self.w_first_visit = weight_first_visit
        self.w_coverage = weight_coverage
        self.w_rescue = weight_rescue
        self.w_hp_loss = weight_hp_loss
        self.w_backtrack = weight_backtrack
        self.w_wait = weight_wait
        self.w_time = weight_time
        self.w_potential = weight_potential
        self.w_redundancy = weight_redundancy
        self.backtrack_window = backtrack_window
        self.gamma = gamma
        
        # Mobility weights (vulnerability)
        self.mobility_weights = {
            'child': 3.0,
            'limited': 2.0,
            'adult': 1.0,
            'infant': 3.5,
            'staff': 1.2,
        }
        
        # Previous state tracking
        self.prev_stats = None
        self.prev_people_hp = {}
        self.prev_high_risk_rooms_completed = set()
        
        # First-visit tracking
        self.first_visit_nodes: Set[str] = set()
        
        # Edge backtrack tracking per agent
        self.agent_recent_edges: Dict[int, List[Tuple[str, str]]] = {}
        
        # Previous agent positions for potential shaping
        self.prev_agent_positions: Dict[int, str] = {}
        self.prev_potential: float = 0.0
        
        # Previous action tracking for WAIT penalty
        self.prev_actions: Dict[int, str] = {}
    
    
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_stats = None
        self.prev_people_hp = {}
        self.prev_high_risk_rooms_completed = set()
        self.first_visit_nodes = set()
        self.agent_recent_edges = {}
        self.prev_agent_positions = {}
        self.prev_potential = 0.0
        self.prev_actions = {}
    
    
    def compute_reward(self, env, actions: Dict[int, str] = None) -> float:
        """
        Compute shaped reward from current environment state and actions.
        
        Args:
            env: BuildingFireEnvironment instance
            actions: Dict mapping agent_id -> action_str (optional for backward compatibility)
            
        Returns:
            reward: Shaped reward for this timestep
        """
        stats = env.get_statistics()
        state = env.get_state()
        
        # Default actions if not provided (backward compatibility)
        if actions is None:
            actions = {i: 'wait' for i in range(len(env.agents))}
        
        # Initialize on first call
        if self.prev_stats is None:
            self.prev_stats = stats.copy()
            self._store_people_hp(state['people'])
            self._update_agent_positions(env)
            self.prev_potential = self._compute_potential(env)
            return 0.0
        
        reward = 0.0
        
        # 1. First-visit coverage bonus
        reward += self._first_visit_coverage_reward(env)
        
        # 2. Standard coverage reward
        reward += self._coverage_reward(stats)
        
        # 3. Rescue reward
        reward += self._rescue_reward(stats, state)
        
        # 4. HP loss penalty
        reward += self._hp_loss_penalty(state)
        
        # 5. Edge backtrack penalty (anti-loop)
        reward += self._backtrack_penalty(env, actions)
        
        # 6. WAIT action penalty
        reward += self._wait_penalty(actions)
        
        # 7. Time penalty
        reward += self._time_penalty()
        
        # 8. Potential-based shaping toward unswept nodes
        reward += self._potential_shaping_reward(env)
        
        # 9. Redundancy bonus
        reward += self._redundancy_bonus(env)
        
        # Update previous state
        self.prev_stats = stats.copy()
        self._store_people_hp(state['people'])
        self._update_agent_positions(env)
        self.prev_actions = actions.copy()
        
        return reward
    
    
    def _first_visit_coverage_reward(self, env) -> float:
        """
        HUGE bonus for visiting a node for the first time.
        
        This is the KEY fix for loop behavior - agents get reward only once per node.
        """
        reward = 0.0
        
        # Check which nodes are swept that weren't before
        for node in env.nodes.values():
            if node.ntype in env.config.get("sweep_node_types", {"room"}):
                if node.sweep_count > 0 and node.nid not in self.first_visit_nodes:
                    # First time visiting this node!
                    self.first_visit_nodes.add(node.nid)
                    reward += self.w_first_visit
                    print(f"[FIRST VISIT] Node {node.nid} swept for first time (+{self.w_first_visit:.1f})")
        
        return reward
    
    
    def _coverage_reward(self, stats: Dict) -> float:
        """Standard coverage reward for sweeping (even if already visited)."""
        delta_sweeps = stats['nodes_swept'] - self.prev_stats['nodes_swept']
        if delta_sweeps > 0:
            return self.w_coverage * delta_sweeps
        return 0.0
    
    
    def _rescue_reward(self, stats: Dict, state: Dict) -> float:
        """Reward for finding and rescuing people (weighted by vulnerability)."""
        reward = 0.0
        
        # Reward for newly found people
        delta_found = stats['people_found'] - self.prev_stats['people_found']
        if delta_found > 0:
            reward += self.w_rescue * delta_found * 0.5  # Half reward for finding
        
        # Full reward for newly rescued people
        delta_rescued = stats['people_rescued'] - self.prev_stats['people_rescued']
        if delta_rescued > 0:
            reward += self.w_rescue * delta_rescued
        
        return reward
    
    
    def _hp_loss_penalty(self, state: Dict) -> float:
        """Minimal penalty for civilian HP loss."""
        penalty = 0.0
        people = state['people']
        
        for pid, person in people.items():
            if not person.is_alive:
                continue
            
            prev_hp = self.prev_people_hp.get(pid, 100.0)
            curr_hp = person.hp
            hp_loss = max(0, prev_hp - curr_hp)
            
            if hp_loss > 0:
                mobility = getattr(person, 'mobility', 'adult')
                weight = self.mobility_weights.get(mobility, 1.0)
                penalty -= self.w_hp_loss * weight * hp_loss
        
        return penalty
    
    
    def _backtrack_penalty(self, env, actions: Dict[int, str]) -> float:
        """
        Penalty for traversing the same edge within recent window.
        
        This is the CRITICAL anti-loop mechanism - punishes A↔B oscillation.
        """
        penalty = 0.0
        
        for agent_id, action in actions.items():
            if not action.startswith('move_'):
                continue
            
            # Get edge traversed
            agent = env.agents[agent_id]
            current = agent.node_id
            target = action[5:]  # Remove 'move_' prefix
            
            # Normalize edge (undirected)
            edge = tuple(sorted([current, target]))
            
            # Initialize tracking for this agent
            if agent_id not in self.agent_recent_edges:
                self.agent_recent_edges[agent_id] = []
            
            recent_edges = self.agent_recent_edges[agent_id]
            
            # Check if this edge was recently traversed
            if edge in recent_edges:
                penalty += self.w_backtrack
                print(f"[BACKTRACK] Agent {agent_id} revisited edge {edge} ({self.w_backtrack:.1f})")
            
            # Add to recent edges
            recent_edges.append(edge)
            
            # Keep only last K edges
            if len(recent_edges) > self.backtrack_window:
                recent_edges.pop(0)
        
        return penalty
    
    
    def _wait_penalty(self, actions: Dict[int, str]) -> float:
        """
        Penalty for WAIT action.
        
        Makes waiting truly undesirable unless necessary (searching, carrying).
        """
        penalty = 0.0
        for action in actions.values():
            if action == 'wait':
                penalty += self.w_wait
        return penalty
    
    
    def _time_penalty(self) -> float:
        """Small penalty per timestep to encourage efficiency."""
        return self.w_time
    
    
    def _potential_shaping_reward(self, env) -> float:
        """
        Potential-based shaping: F(s,a,s') = γΦ(s') - Φ(s)
        where Φ(s) = -min_dist(agent, unswept_node)
        
        This provides dense reward signal toward unswept areas WITHOUT
        creating reward cycles (potential-based shaping is reward-invariant).
        """
        current_potential = self._compute_potential(env)
        
        # F(s,a,s') = γΦ(s') - Φ(s)
        shaping_reward = self.gamma * current_potential - self.prev_potential
        
        self.prev_potential = current_potential
        
        return self.w_potential * shaping_reward
    
    
    def _compute_potential(self, env) -> float:
        """
        Compute potential function: Φ(s) = -sum_i min_dist(agent_i, unswept)
        
        Higher potential = closer to unswept nodes = better state.
        """
        # Find unswept nodes
        unswept = []
        for node in env.nodes.values():
            if node.ntype in env.config.get("sweep_node_types", {"room"}):
                if node.sweep_count == 0:
                    unswept.append(node.nid)
        
        if not unswept:
            # All swept - maximum potential
            return 0.0
        
        # Compute minimum distance from each agent to any unswept node
        total_distance = 0.0
        for agent in env.agents.values():
            if not hasattr(agent, 'is_active') or agent.is_active:
                min_dist = self._shortest_path_distance(env, agent.node_id, unswept)
                total_distance += min_dist
        
        # Potential = -distance (closer is better)
        return -total_distance
    
    
    def _shortest_path_distance(self, env, start: str, targets: List[str]) -> float:
        """Compute shortest path distance from start to any target node."""
        try:
            # Use NetworkX shortest path
            min_dist = float('inf')
            for target in targets:
                try:
                    dist = nx.shortest_path_length(env.G, start, target)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue
            
            return min_dist if min_dist != float('inf') else 10.0
        except Exception:
            return 10.0  # Default if graph traversal fails
    
    
    def _redundancy_bonus(self, env) -> float:
        """Bonus when high-risk rooms achieve required redundancy."""
        bonus = 0.0
        
        for node in env.nodes.values():
            if (getattr(node, 'risk_level', 'normal') == 'high' and
                getattr(node, 'required_sweeps', 1) >= 2):
                
                if (node.sweep_count >= node.required_sweeps and
                    node.nid not in self.prev_high_risk_rooms_completed):
                    
                    bonus += self.w_redundancy
                    self.prev_high_risk_rooms_completed.add(node.nid)
        
        return bonus
    
    
    def _update_agent_positions(self, env):
        """Store current agent positions."""
        self.prev_agent_positions = {
            agent_id: agent.node_id
            for agent_id, agent in env.agents.items()
        }
    
    
    def _store_people_hp(self, people: Dict):
        """Store current HP of all people."""
        self.prev_people_hp = {pid: p.hp for pid, p in people.items()}
    
    
    @classmethod
    def for_scenario(cls, scenario: str, curriculum_phase: int = 0) -> 'ImprovedRewardShaper':
        """
        Create reward shaper with scenario-specific weights.
        
        Args:
            scenario: 'office', 'daycare', or 'warehouse'
            curriculum_phase: Training phase (0=full, 1=simple, 2=mild, 3=realistic)
        """
        if curriculum_phase == 1:
            # Phase 1: Simplified - focus on exploration
            return cls(
                scenario=scenario,
                weight_first_visit=200.0,    # HUGE first-visit bonus
                weight_coverage=5.0,
                weight_rescue=100.0,
                weight_hp_loss=0.0,          # No HP penalty
                weight_backtrack=-20.0,      # Strong anti-loop
                weight_wait=-0.2,            # Discourage waiting
                weight_time=0.0,             # No time pressure
                weight_potential=10.0,       # Strong exploration signal
                weight_redundancy=0.0,       # No redundancy yet
            )
        
        elif curriculum_phase == 2:
            # Phase 2: Add rescue focus
            return cls(
                scenario=scenario,
                weight_first_visit=150.0,
                weight_coverage=10.0,
                weight_rescue=200.0,         # Increased rescue reward
                weight_hp_loss=0.01,         # Small HP penalty
                weight_backtrack=-15.0,
                weight_wait=-0.15,
                weight_time=-0.005,          # Small time cost
                weight_potential=8.0,
                weight_redundancy=0.0,
            )
        
        elif curriculum_phase == 3:
            # Phase 3: Add redundancy
            return cls(
                scenario=scenario,
                weight_first_visit=100.0,
                weight_coverage=10.0,
                weight_rescue=200.0,
                weight_hp_loss=0.01,
                weight_backtrack=-10.0,
                weight_wait=-0.1,
                weight_time=-0.01,
                weight_potential=5.0,
                weight_redundancy=50.0,      # Add redundancy bonus
            )
        
        else:
            # Phase 0: Full rewards (default)
            return cls(scenario=scenario)
    
    
    def get_episode_summary(self, env) -> Dict[str, float]:
        """Get summary statistics for completed episode."""
        stats = env.get_statistics()
        
        return {
            'people_rescued': stats.get('people_rescued', 0),
            'people_found': stats.get('people_found', 0),
            'people_alive': stats.get('people_alive', 0),
            'nodes_swept': stats.get('nodes_swept', 0),
            'first_visits': len(self.first_visit_nodes),
            'high_risk_redundancy': stats.get('high_risk_redundancy', 0),
            'sweep_complete': stats.get('sweep_complete', False),
            'time_step': stats.get('time_step', 0),
        }
