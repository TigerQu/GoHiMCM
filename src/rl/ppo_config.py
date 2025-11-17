"""
Configuration dataclasses for PPO training.

===== NEW FILE: Structured hyperparameter management =====

This makes hyperparameter tuning experiments reproducible and organized.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class PPOConfig:
    """
    Complete PPO configuration for reproducible experiments.
    
    Structured config instead of scattered kwargs
    Makes hyperparameter experiments systematic and reproducible.
    """
    
    # Experiment metadata
    scenario: str = "office"              # "office", "daycare", or "warehouse"
    experiment_name: str = "baseline"      # Name for logging
    seed: int = 42                         # Random seed for reproducibility
    
    # Agent configuration
    num_agents: int = 2                    # Number of firefighter agents
    max_actions: int = 15                  # Max action space size
    
    # PPO hyperparameters (optimized for RTX 5090)
    lr_policy: float = 5e-4                # Policy learning rate (increased for faster learning)
    lr_value: float = 1e-3                 # Value learning rate
    gamma: float = 0.99                    # Discount factor
    gae_lambda: float = 0.95               # GAE lambda parameter
    clip_epsilon: float = 0.2              # PPO clipping parameter
    entropy_coef: float = 0.02             # Entropy bonus coefficient (increased for exploration)
    value_loss_coef: float = 0.5           # Value loss weight
    max_grad_norm: float = 0.5             # Gradient clipping threshold
    
    # Training configuration (RTX 5090 optimized - larger batches)
    num_iterations: int = 8000             # Total training iterations (increased for better learning)
    steps_per_rollout: int = 100           # Max steps per episode (balanced speed/quality)
    num_ppo_epochs: int = 4                # PPO update epochs per iteration (reduced for speed)
    num_parallel_envs: int = 1             # Number of parallel environments (1=no parallel)
    batch_size: int = 64                   # Minibatch size for updates (RTX 5090 can handle large batches)
    
    # Evaluation configuration
    eval_interval: int = 200               # Evaluate every N iterations (much less frequent for speed)
    num_eval_episodes: int = 10            # Episodes per evaluation (reduced for speed)
    num_train_layouts: int = 50            # Training layout seeds (reduced)
    num_eval_layouts: int = 10             # Evaluation layout seeds (reduced)
    
    # Logging configuration
    log_interval: int = 10                 # Log every N iterations
    checkpoint_interval: int = 50          # Save checkpoint every N iterations
    use_tensorboard: bool = False          # Enable TensorBoard logging
    
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    
    @classmethod
    def load(cls, path: str) -> 'PPOConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    
    @classmethod
    def get_default(cls, scenario: str) -> 'PPOConfig':
        """
        Get default config for a scenario.
        
        ===== SCENARIO-SPECIFIC DEFAULTS =====
        Different scenarios may need different training settings.
        """
        if scenario == "office":
            return cls(
                scenario="office",
                experiment_name="office_baseline_rtx5090",
                num_iterations=8000,
                steps_per_rollout=100,
                num_ppo_epochs=4,
                batch_size=64,
                lr_policy=5e-4,
                entropy_coef=0.02,
            )
        
        elif scenario == "daycare":
            return cls(
                scenario="daycare",
                experiment_name="daycare_baseline_rtx5090",
                num_iterations=6000,         # More iterations for complex scenario
                steps_per_rollout=120,       # Longer episodes (GPU optimized)
                num_ppo_epochs=4,
                batch_size=64,
                entropy_coef=0.02,           # Higher exploration for multi-floor
            )
        
        elif scenario == "warehouse":
            return cls(
                scenario="warehouse",
                experiment_name="warehouse_baseline_rtx5090",
                num_iterations=6000,
                steps_per_rollout=120,
                num_ppo_epochs=4,
                batch_size=64,
                lr_policy=1e-4,              # Lower LR for sparse rewards
            )
        
        else:
            return cls(scenario=scenario)
    
    
    def __repr__(self) -> str:
        """Pretty print config."""
        lines = ["PPOConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}={value},")
        lines.append(")")
        return "\n".join(lines)


def create_hparam_grid():
    """
    Create hyperparameter search grid.
    
    ===== CHANGE 4: Systematic hyperparameter tuning =====
    Returns list of configs for grid search.
    
    For HiMCM, we use a small, manageable grid:
    - lr_policy: {1e-4, 3e-4}
    - entropy_coef: {0.01, 0.02}
    - clip_epsilon: {0.1, 0.2}
    
    Total: 2 × 2 × 2 = 8 configurations
    """
    base_config = PPOConfig.get_default("office")
    
    configs = []
    
    for lr in [1e-4, 3e-4]:
        for entropy in [0.01, 0.02]:
            for clip_eps in [0.1, 0.2]:
                config = PPOConfig.get_default("office")
                config.lr_policy = lr
                config.entropy_coef = entropy
                config.clip_epsilon = clip_eps
                config.experiment_name = f"office_lr{lr}_ent{entropy}_clip{clip_eps}"
                configs.append(config)
    
    return configs


def create_scenario_configs():
    """
    Create configs for all three scenarios with best hyperparameters.
    
    Use after finding best config on office, apply to other scenarios.
    """
    # Assuming best config from grid search:
    best_lr = 3e-4
    best_entropy = 0.01
    best_clip = 0.2
    
    configs = []
    
    for scenario in ["office", "daycare", "warehouse"]:
        config = PPOConfig.get_default(scenario)
        config.lr_policy = best_lr
        config.entropy_coef = best_entropy
        config.clip_epsilon = best_clip
        config.num_iterations = 10000  # Longer training for final runs
        configs.append(config)
    
    return configs