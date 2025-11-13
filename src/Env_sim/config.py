# Node type encoding for one-hot vectors 
NODE_TYPES = {
    "room": 0,    # Regular rooms where people may be located
    "hall": 1,    # Hallways connecting rooms
    "exit": 2,     # Exit points of the building
    "floor": 3,    # Floor nodes for multi-story buildings
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
    "max_steps": 500,                 # Max steps per episode
    
    # People parameters (for future evacuation modeling)
    "child_speed": 0.9,              # m/s
    "adult_speed": 1.2,              # m/s
    "limited_speed": 0.7,            # m/s (elderly/disabled)

    # Civilian behavior
    "awareness_delay_mean": 3,  # Time steps before moving
    "awareness_delay_std": 1,
    
    # Dynamic pathfinding weights
    "theta_density": 0.8,  # Congestion sensitivity
    "hazard_penalty_fire": 100.0,  # Avoid fire
    "hazard_penalty_smoke": 10.0,  # Discourage smoke
}
