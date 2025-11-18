#!/usr/bin/env python3
"""
Curriculum Phase 1: Run full PPO training on simplified environment

Configuration:
  - Scenario: office (11 nodes, 2 agents, 6 people)
  - Rewards: coverage (0.5) + rescue (5.0)
  - No penalties: hp_loss=0, time=0, redundancy=0
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppo_config import PPOConfig
from reward_shaper import RewardShaper
from enhanced_training import EnhancedPPOTrainer


def train_phase1():
    """Run PPO training on Phase 1 environment"""
    
    print("\n" + "="*80)
    print("CURRICULUM LEARNING: PHASE 1 PPO TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  Scenario: office (11 nodes, 2 agents, 6 people)")
    print("  Rewards: coverage (0.5) + rescue (5.0)")
    print("  No penalties: hp_loss=0, time=0, redundancy=0")
    print("="*80 + "\n")
    
    # Create config
    config = PPOConfig(
        scenario="office",
        experiment_name="phase1_simplified_rewards",
        seed=42,
        num_agents=2,
        lr_policy=3e-4,            # Baseline learning rate
        lr_value=1e-4,             # CRITICAL: Much slower critic learning (1/3x from 3e-4)
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,         # CRITICAL: Reduced from 0.02 - entropy was too strong
        value_loss_coef=0.1,       # CRITICAL: Reduced from 0.25 - critic too dominant
        num_iterations=100,        # Run 100 iterations for convergence check
        steps_per_rollout=100,
        num_ppo_epochs=1,          # CRITICAL: Only 1 epoch - fewer trust region violations
        batch_size=64,
        batch_rollout_size=1,
        eval_interval=50,          # Eval at iterations 50 and 100
        num_eval_episodes=5,
        log_interval=5,            # Print every 5 iterations for detailed monitoring
        checkpoint_interval=50,
        clip_epsilon=0.2,          # Keep PPO clipping stable
        max_grad_norm=0.5,         # Gradient clipping
    )
    
    # Create trainer
    print("[1] Initializing trainer...")
    trainer = EnhancedPPOTrainer(config)
    
    # Override with Phase 1 reward shaper
    print("[2] Setting up Phase 1 rewards...")
    reward_shaper = RewardShaper.for_scenario(
        scenario="office",
        curriculum_phase=1
    )
    trainer.reward_shaper = reward_shaper
    
    print(f"    - coverage: {reward_shaper.w_coverage}")
    print(f"    - rescue: {reward_shaper.w_rescue}")
    print(f"    - hp_loss: {reward_shaper.w_hp_loss}")
    print(f"    - time: {reward_shaper.w_time}")
    print(f"    - redundancy: {reward_shaper.w_redundancy}")
    
    # Train
    print("\n[3] Starting PPO training...\n")
    trainer.train()
    
    print("\n" + "="*80)
    print("PHASE 1 TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_phase1()
