"""
Reward shaping module for building evacuation RL.

===== NEW FILE: Computes rewards from environment statistics =====

This module reads env.get_statistics() and env.get_state() to compute
shaped rewards WITHOUT modifying the environment code.

Reward components:
1. Coverage reward: +points for sweeping rooms (weighted by risk)
2. Rescue reward: +points for finding/rescuing people (weighted by vulnerability)
3. HP preservation: -penalty for civilian HP loss
4. Time penalty: -small penalty per timestep
5. Redundancy bonus: +bonus when high-risk rooms get 2+ sweeps
"""

import torch
from typing import Dict, Any


class RewardShaper:
    """
    Computes shaped rewards from environment statistics.
    
    Designed to work with BuildingFireEnvironment without any env modifications.
    """
    
    def __init__(
        self,
        scenario: str = "office",
        weight_coverage: float = 1.0,
        weight_rescue: float = 10.0,
        weight_hp_loss: float = 0.1,
        weight_time: float = 0.01,
        weight_redundancy: float = 5.0,
    ):
        """
        Initialize reward shaper with scenario-specific weights.
        
        Args:
            scenario (str): "office", "daycare", or "warehouse"
            weight_coverage (float): Weight for room sweep rewards
            weight_rescue (float): Weight for rescue rewards
            weight_hp_loss (float): Weight for HP loss penalty
            weight_time (float): Weight for time penalty
            weight_redundancy (float): Weight for redundancy bonus
        """
        self.scenario = scenario
        self.w_coverage = weight_coverage
        self.w_rescue = weight_rescue
        self.w_hp_loss = weight_hp_loss
        self.w_time = weight_time
        self.w_redundancy = weight_redundancy
        
        # Mobility weights (vulnerability)
        # Children and limited mobility are more valuable to rescue
        self.mobility_weights = {
            'child': 3.0,      # Highest priority
            'limited': 2.0,    # Medium-high priority
            'adult': 1.0,      # Baseline
            'infant': 3.5,     # Even higher than child
            'staff': 1.2,      # Slightly above adult
        }
        
        # Risk level weights for rooms
        # High-risk rooms (nurseries, labs) get more reward for sweeping
        self.risk_weights = {
            'high': 2.0,       # High-risk rooms worth 2x
            'normal': 1.0,     # Standard rooms baseline
        }
        
        # Previous state tracking (for delta computation)
        self.prev_stats = None
        self.prev_people_hp = {}
        self.prev_high_risk_rooms_completed = set()
        self.prev_agent_distances = {}  # Track agent distances for shaping rewards
        
    
    def reset(self):
        """Reset internal state for new episode."""
        self.prev_stats = None
        self.prev_people_hp = {}
        self.prev_high_risk_rooms_completed = set()
        self.prev_agent_distances = {}  # Reset distance tracking
    
    
    def compute_reward(self, env) -> float:
        """
        Compute shaped reward from current environment state.
        
        ===== MAIN METHOD: Reads from env without modifying it =====
        
        Args:
            env (BuildingFireEnvironment): The environment instance
            
        Returns:
            reward (float): Shaped reward for this timestep
        """
        # Get current statistics
        stats = env.get_statistics()
        state = env.get_state()
        
        # Initialize on first call
        if self.prev_stats is None:
            self.prev_stats = stats.copy()
            self._store_people_hp(state['people'])
            return 0.0  # No reward on first step
        
        # Compute reward components
        reward = 0.0
        
        # 1. Coverage reward (rooms swept)
        reward += self._coverage_reward(env, stats)
        
        # 2. Rescue reward (people found/rescued)
        reward += self._rescue_reward(stats, state)
        
        # 3. HP loss penalty
        reward += self._hp_loss_penalty(state)
        
        # 4. Time penalty
        reward += self._time_penalty()
        
        # 5. Redundancy bonus (high-risk rooms with 2+ sweeps)
        reward += self._redundancy_bonus(env)
        
        # 6. Distance shaping (encourage agents to move toward people and unseen nodes)
        reward += self._distance_shaping_reward(env)
        
        # Update previous state
        self.prev_stats = stats.copy()
        self._store_people_hp(state['people'])
        
        return reward
    
    
    def _coverage_reward(self, env, stats: Dict) -> float:
        """
        Reward for sweeping rooms (weighted by risk level).
        
        High-risk rooms (daycares, labs) give more reward.
        """
        # Count new sweeps this timestep
        delta_sweeps = stats['nodes_swept'] - self.prev_stats['nodes_swept']
        
        if delta_sweeps <= 0:
            return 0.0
        
        # Weight by room risk levels
        # Check which rooms were just swept
        reward = 0.0
        for node in env.nodes.values():
            if node.ntype in env.config.get("sweep_node_types", {"room"}):
                # Check if this room was just swept
                # (Approximation: distribute delta_sweeps across all rooms)
                risk_level = getattr(node, 'risk_level', 'normal')
                weight = self.risk_weights.get(risk_level, 1.0)
                reward += self.w_coverage * weight * (delta_sweeps / max(1, stats.get('total_people', 1)))
        
        return reward
    
    
    def _rescue_reward(self, stats: Dict, state: Dict) -> float:
        """
        Reward for finding and rescuing people (weighted by vulnerability).
        
        Children and limited mobility people give more reward.
        """
        reward = 0.0
        
        # Reward for newly found people
        delta_found = stats['people_found'] - self.prev_stats['people_found']
        if delta_found > 0:
            # Weight by mobility of found people
            people = state['people']
            for person in people.values():
                if person.seen:  # This person was found
                    mobility = getattr(person, 'mobility', 'adult')
                    weight = self.mobility_weights.get(mobility, 1.0)
                    # Approximate: assume all newly found people contribute equally
                    reward += self.w_rescue * weight * (delta_found / len(people))
        
        # Additional reward for newly rescued people
        delta_rescued = stats['people_rescued'] - self.prev_stats['people_rescued']
        if delta_rescued > 0:
            # Rescued people get double reward (found + evacuated)
            people = state['people']
            for person in people.values():
                if person.rescued:
                    mobility = getattr(person, 'mobility', 'adult')
                    weight = self.mobility_weights.get(mobility, 1.0)
                    reward += self.w_rescue * weight * (delta_rescued / len(people))
        
        return reward
    
    
    def _hp_loss_penalty(self, state: Dict) -> float:
        """
        Penalty for civilian HP loss (weighted by vulnerability).
        
        Losing HP on children/limited mobility people is more costly.
        """
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
    
    
    def _time_penalty(self) -> float:
        """
        Small penalty per timestep to encourage faster sweeps.
        """
        return -self.w_time
    
    
    def _redundancy_bonus(self, env) -> float:
        """
        Bonus when high-risk rooms achieve required redundancy (2+ sweeps by different agents).
        
        Only awarded once per room when it first reaches required_sweeps.
        """
        bonus = 0.0
        
        for node in env.nodes.values():
            # Only check high-risk rooms that require redundancy
            if (getattr(node, 'risk_level', 'normal') == 'high' and
                getattr(node, 'required_sweeps', 1) >= 2):
                
                # Check if this room just completed redundancy requirement
                if (node.sweep_count >= node.required_sweeps and
                    node.nid not in self.prev_high_risk_rooms_completed):
                    
                    bonus += self.w_redundancy
                    self.prev_high_risk_rooms_completed.add(node.nid)
        
        return bonus
    
    
    def _distance_shaping_reward(self, env) -> float:
        """
        Reward for moving closer to people or unseen nodes.
        
        This provides a denser reward signal to guide exploration.
        - Reward for getting closer to unseen people
        - Reward for moving toward uncovered rooms
        """
        reward = 0.0
        
        if self.w_rescue == 0 and self.w_coverage == 0:
            return 0.0  # Skip if both main rewards are zero
        
        try:
            # Get current agent positions
            current_distances = {}
            for agent_id in range(len(env.agents)):
                agent_pos = env.get_agent_node_index(agent_id)
                current_distances[agent_id] = agent_pos
            
            # Compare with previous positions to reward movement
            # Reward small movements that get closer to goals
            if self.prev_agent_distances:
                for agent_id, prev_pos in self.prev_agent_distances.items():
                    curr_pos = current_distances.get(agent_id, prev_pos)
                    
                    # Reward for moving to a different node (encourage exploration)
                    if curr_pos != prev_pos:
                        reward += self.w_coverage * 0.1  # Small bonus for moving
                        
                        # Check if new node has unseen people
                        try:
                            curr_node = env.nodes[curr_pos]
                            if hasattr(curr_node, 'people') and curr_node.people:
                                # Bonus for reaching a node with people
                                reward += self.w_rescue * 0.2
                        except:
                            pass
            
            self.prev_agent_distances = current_distances
        
        except Exception as e:
            # Silently skip distance shaping if there's an error
            pass
        
        return reward
    
    def _store_people_hp(self, people: Dict):
        """Store current HP of all people for next delta computation."""
        self.prev_people_hp = {pid: p.hp for pid, p in people.items()}
    
    
    @classmethod
    def for_scenario(cls, scenario: str, curriculum_phase: int = 0) -> 'RewardShaper':
        """
        Create reward shaper with scenario-specific weights.
        
        Args:
            scenario (str): "office", "daycare", or "warehouse"
            curriculum_phase (int): 0=full, 1=simple (no penalties), 2=mild HP, 3=realistic
            
        Returns:
            RewardShaper: Configured reward shaper
        """
        # Phase 1: Simplified curriculum - ONLY coverage + rescue, NO penalties
        if curriculum_phase == 1:
            return cls(
                scenario=scenario,
                weight_coverage=30.0,     # HIGH: Make sweeping the PRIMARY goal
                weight_rescue=30.0,       # Equal weight to coverage - must sweep AND rescue
                weight_hp_loss=0.0,       # NO HP penalty - keep it simple!
                weight_time=0.0,          # NO time penalty - allow thinking time
                weight_redundancy=0.0,    # NO redundancy yet
            )
        
        # Phase 2: Mild realism - add small HP penalty
        if curriculum_phase == 2:
            return cls(
                scenario=scenario,
                weight_coverage=5.0,      # Increased from 0.5 - sweep is priority
                weight_rescue=5.0,        # Equal weight - must balance both
                weight_hp_loss=0.02,      # Small penalty for HP loss
                weight_time=0.0,          # Still no time penalty
                weight_redundancy=0.0,    # Still no redundancy
            )
        
        # Phase 3: Realistic - add redundancy
        if curriculum_phase == 3:
            return cls(
                scenario=scenario,
                weight_coverage=5.0,      # Increased - sweep is still important
                weight_rescue=5.0,        # Equal weight
                weight_hp_loss=0.02,
                weight_time=0.0,
                weight_redundancy=5.0,    # Increased redundancy bonus (was 1.0)
            )
        
        # Phase 0 (default): Full reward structure
        if scenario == "office":
            # Standard office: balanced rewards
            # CRITICAL FIX: HP loss was killing training!
            return cls(
                scenario="office",
                weight_coverage=10.0,      # High for exploration
                weight_rescue=100.0,       # Very high - make rescue extremely rewarding
                weight_hp_loss=0.01,       # DRASTICALLY reduced (was 1.0, then 0.5, now 0.01!)
                weight_time=0.0,           # Removed completely for now
                weight_redundancy=20.0,    # High redundancy bonus
            )
        
        elif scenario == "daycare":
            # Daycare: heavily weight children and redundancy
            return cls(
                scenario="daycare",
                weight_coverage=1.5,      # Higher coverage reward
                weight_rescue=15.0,       # Much higher rescue reward
                weight_hp_loss=0.3,       # Stronger HP loss penalty
                weight_time=0.005,        # Lower time penalty (take time to be thorough)
                weight_redundancy=10.0,   # Very high redundancy bonus
            )
        
        elif scenario == "warehouse":
            # Warehouse: weight labs and hazardous materials
            return cls(
                scenario="warehouse",
                weight_coverage=2.0,      # High coverage for labs
                weight_rescue=12.0,       # High rescue for workers
                weight_hp_loss=0.15,      # Moderate HP loss penalty
                weight_time=0.015,        # Higher time penalty (efficiency matters)
                weight_redundancy=8.0,    # High redundancy for labs
            )
        
        else:
            # Default to office
            return cls(scenario="office")
    
    
    def get_episode_summary(self, env) -> Dict[str, float]:
        """
        Get summary statistics for completed episode.
        
        Useful for logging and analysis.
        
        Returns:
            summary (Dict): Summary statistics
        """
        stats = env.get_statistics()
        
        return {
            'people_rescued': stats.get('people_rescued', 0),
            'people_found': stats.get('people_found', 0),
            'people_alive': stats.get('people_alive', 0),
            'nodes_swept': stats.get('nodes_swept', 0),
            'high_risk_redundancy': stats.get('high_risk_redundancy', 0),
            'sweep_complete': stats.get('sweep_complete', False),
            'active_agents': stats.get('active_agents', 0),
            'time_step': stats.get('time_step', 0),
        }