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
    # Hazard spreading parameters
    "fire_spread_prob": 0.15,        # Reduced from 0.3 - more realistic spread
    "fire_spread_delay": 5,          # Increased from 2 - fire spreads slower
    
    # Health degradation parameters (per timestep, 1 timestep = ~5 seconds)
    "hp_loss_fire": 1.0,             # Reduced from 5.0 - people survive ~8 minutes in fire
    "hp_loss_smoke": 0.3,            # Reduced from 2.0 - smoke is dangerous but slower
    
    # Agent parameters (1 timestep = ~5 seconds)
    "agent_speed": 1.5,              # m/s (walking speed with gear)
    "search_time_per_sqm": 0.5,      # Seconds per square meter (more realistic)
    "max_steps": 500,                # ~40 minutes max episode
    
    # People parameters
    "child_speed": 0.5,              # Reduced - children move slower, especially in fear
    "adult_speed": 0.8,              # Reduced - people move cautiously in fire
    "limited_speed": 0.4,            # Reduced - mobility impaired
    "assisted_speed_multiplier": 1.8, # Speed boost when assisted by agent

    # Civilian behavior
    "awareness_delay_mean": 5,       # Increased - people take time to react
    "awareness_delay_std": 2,        # More variation
    "panic_speed_reduction": 0.7,    # People slow down when panicked (not yet assisted)
    
    # Dynamic pathfinding weights
    "theta_density": 0.8,            # Congestion sensitivity
    "hazard_penalty_fire": 100.0,    # Avoid fire
    "hazard_penalty_smoke": 10.0,    # Discourage smoke
    
    # Sweep configuration
    "sweep_node_types": {"room"},    # Only rooms need thorough search
    "auto_sweep_types": {"hall", "exit"},  # Auto-sweep on visit
    "auto_sweep_on_visit": True,
}