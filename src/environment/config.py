# Node type encoding for one-hot vectors 
NODE_TYPES = {
    "room": 0,    # Regular rooms where people may be located
    "hall": 1,    # Hallways connecting rooms
    "exit": 2,     # Exit points of the building
    "floor": 3,    # Floor nodes for multi-story buildings
}

# Feature dimension: F = 11 (4-dim one-hot for node types: room/hall/exit/floor)
# Structure: [4 one-hot] + [fire, smoke, length, people_count, avg_hp, agent_here, dist_to_fire] = 11D
FEATURE_DIM = 11

# Default configuration values
DEFAULT_CONFIG = {
 
    # HAZARD SPREADING PARAMETERS
 
    "fire_spread_prob": 0.15,        # Base probability of fire spreading to adjacent node (15%)
    "fire_spread_delay": 5,          # Timesteps between fire spread attempts (realistic spread rate)
    
    # ===== CHANGE 7: Fire intensity configuration (NEW) =====
    # These parameters control continuous fire/smoke intensity modeling (0-1 scale)
    # instead of binary on/off states, enabling more realistic hazard representation
    "fire_intensity_init": 0.4,      # Initial fire intensity when node ignites (0.4 = moderate start)
    "fire_intensity_growth": 0.1,    # Intensity increase per spread cycle (grows to 1.0 over time)
    "smoke_density_base": 0.3,       # Initial smoke density when smoke appears (0.3 = light smoke)
    "smoke_density_growth": 0.1,     # Smoke density increase per cycle (accumulates over time)
  
    # CIVILIAN HEALTH DEGRADATION PARAMETERS (per timestep, 1 timestep ≈ 5 seconds)
 
    "hp_loss_fire": 1.0,             # Base HP loss in fire (1.0 HP/timestep → ~8 min survival in fire)
    "hp_loss_smoke": 0.3,            # Base HP loss in smoke (0.3 HP/timestep → ~27 min in smoke)
    
    # ===== CHANGE 8: Mobility-specific vulnerability multipliers (NEW) =====
    # These multipliers account for physiological differences in hazard tolerance:
    # - Children: smaller body mass, faster metabolism → more vulnerable (1.5x damage)
    # - Limited mobility: pre-existing health conditions → more vulnerable (1.8x damage)
    # - Adults: baseline healthy adult tolerance (1.0x damage)
    # 
    # Final damage = base_hp_loss * intensity * mobility_multiplier
    # Example: Child in full fire → 1.0 * 1.0 * 1.5 = 1.5 HP/timestep (5.3 min survival)
    "hp_mult_child": 1.5,            # Children 50% more vulnerable
    "hp_mult_adult": 1.0,            # Adults baseline
    "hp_mult_limited": 1.8,          # Limited mobility 80% more vulnerable
   
    # RESPONDER SAFETY PARAMETERS (NEW)
  
    # ===== CHANGE 9: Responder exposure tracking (NEW) =====
    # Models cumulative exposure ξ_j(t) and enforces safety withdrawal protocols.
    # Responders must withdraw when:
    #   1. Cumulative exposure exceeds Xi_max (too much smoke/heat exposure)
    #   2. HP depletes to 0 (responder injured)
    #
    # This creates a resource constraint forcing triage decisions in policy learning.
    "responder_hp_loss_fire": 0.5,   # HP loss per timestep in fire (slower than civilians due to gear)
    "responder_hp_loss_smoke": 0.2,  # HP loss per timestep in smoke (protective equipment helps)
    "responder_exposure_fire": 1.0,  # Exposure increment in fire (1.0 per timestep)
    "responder_exposure_smoke": 0.3, # Exposure increment in smoke (0.3 per timestep)
    "responder_Xi_max": 100.0,       # Maximum cumulative exposure before forced withdrawal
                                      # At default rates: ~100 timesteps in fire, ~333 in smoke
    # GENERAL SIMULATION PARAMETERS
 
    "agent_speed": 1.5,              # Movement speed in m/s (walking with gear)
    "search_time_per_sqm": 0.5,      # Search time per square meter (thorough room search)
    "max_steps": 500,                # Maximum episode length (~42 minutes real-time)
    
 
    # CIVILIAN MOVEMENT PARAMETERS
  
    "child_speed": 0.5,              # Children move slowly, especially when scared (m/s)
    "adult_speed": 0.8,              # Adults move cautiously in fire/smoke (m/s)
    "limited_speed": 0.4,            # Limited mobility (wheelchair, elderly, etc.) (m/s)
    "assisted_speed_multiplier": 1.8, # Speed boost when assisted by responder (confidence)

    
    # CIVILIAN BEHAVIOR PARAMETERS
   
    "awareness_delay_mean": 5,       # Mean timesteps before civilian reacts to alarm
    "awareness_delay_std": 2,        # Standard deviation (some react faster than others)
    "panic_speed_reduction": 0.7,    # Speed reduction when panicked (before being assisted)
    
    # PATHFINDING AND MOVEMENT PENALTIES

    "theta_density": 0.8,            # Congestion sensitivity (how much crowding slows movement)
    "hazard_penalty_fire": 100.0,    # Path cost penalty for fire (effectively blocks path)
    "hazard_penalty_smoke": 10.0,    # Path cost penalty for smoke (discourage but passable)
    
    
    # SWEEP CONFIGURATION
    "sweep_node_types": {"room"},    # Node types that require thorough search
    "auto_sweep_types": {"hall", "exit"},  # Node types that auto-sweep on visit
    "auto_sweep_on_visit": True,     # Whether to mark auto-sweep nodes as swept immediately
}