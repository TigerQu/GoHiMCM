"""
Quick training script with optimized settings for fast iteration.

Use this for rapid prototyping and testing.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


def create_fast_config(scenario: str = "office") -> PPOConfig:
    """
    Create a fast training config for quick experiments.
    
    Optimized for speed over sample efficiency:
    - Fewer iterations (2000 instead of 5000)
    - Shorter rollouts (50 instead of 100)
    - Less frequent evaluation (every 500 iterations)
    - Fewer PPO epochs (2 instead of 4)
    
    Should complete in ~30-45 minutes on RTX 5090.
    """
    config = PPOConfig.get_default(scenario)
    
    # Fast training overrides
    config.experiment_name = f"{scenario}_fast_rtx5090"
    config.num_iterations = 2000           # Reduced from 5000
    config.steps_per_rollout = 50          # Reduced from 100
    config.num_ppo_epochs = 2              # Reduced from 4
    
    # Evaluation settings
    config.eval_interval = 500             # Very infrequent
    config.num_eval_episodes = 5           # Quick eval
    
    # Logging
    config.log_interval = 20               # Log less frequently
    config.checkpoint_interval = 500       # Save less frequently
    
    print("=" * 60)
    print("FAST TRAINING MODE")
    print("=" * 60)
    print(f"Total iterations: {config.num_iterations}")
    print(f"Steps per rollout: {config.steps_per_rollout}")
    print(f"PPO epochs: {config.num_ppo_epochs}")
    print(f"Estimated time: ~30-45 minutes")
    print("=" * 60 + "\n")
    
    return config


if __name__ == "__main__":
    # Parse scenario from command line
    scenario = sys.argv[1] if len(sys.argv) > 1 else "office"
    
    # Create fast config
    config = create_fast_config(scenario)
    
    # Train
    trainer = EnhancedPPOTrainer(config)
    trainer.train()
