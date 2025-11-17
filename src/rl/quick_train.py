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
    
    Optimized for LEARNING while still being reasonably fast:
    - 3000 iterations (increased from 2000 for better learning)
    - 80 steps per rollout (increased from 50)
    - 3 PPO epochs (increased from 2)
    
    Should complete in ~1.5-2 hours on RTX 5090 with actual learning.
    """
    config = PPOConfig.get_default(scenario)
    
    # Balanced training settings
    config.experiment_name = f"{scenario}_fast_rtx5090"
    config.num_iterations = 3000           # Increased from 2000
    config.steps_per_rollout = 80          # Increased from 50
    config.num_ppo_epochs = 3              # Increased from 2
    
    # Better learning rates for faster convergence
    config.lr_policy = 5e-4                # Slightly higher
    config.lr_value = 1e-3
    config.entropy_coef = 0.02             # More exploration
    
    # Evaluation settings
    config.eval_interval = 300             # Less frequent
    config.num_eval_episodes = 10          # Reasonable
    
    # Logging
    config.log_interval = 20               # More frequent for monitoring
    config.checkpoint_interval = 300       # Match eval
    
    print("=" * 60)
    print("FAST TRAINING MODE (Optimized for Learning)")
    print("=" * 60)
    print(f"Total iterations: {config.num_iterations}")
    print(f"Steps per rollout: {config.steps_per_rollout}")
    print(f"PPO epochs: {config.num_ppo_epochs}")
    print(f"Learning rate: {config.lr_policy}")
    print(f"Entropy coef: {config.entropy_coef}")
    print(f"Estimated time: ~1.5-2 hours")
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
