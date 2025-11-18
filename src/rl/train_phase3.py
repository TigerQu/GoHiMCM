#!/usr/bin/env python3
"""
Curriculum Phase 3: Add redundancy bonus to test full reward structure

Configuration:
  - Scenario: office (11 nodes, 2 agents, 6 people)
  - Rewards: coverage (0.5) + rescue (5.0) + redundancy (1.0)
  - Penalties: hp_loss (0.002) - REDUCED for stability
  - No time penalty yet
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppo_config import PPOConfig
from reward_shaper import RewardShaper
from enhanced_training import EnhancedPPOTrainer


def train_phase3():
    """Run PPO training on Phase 3 environment"""
    
    print("\n" + "="*80)
    print("CURRICULUM LEARNING: PHASE 3 PPO TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  Scenario: office (11 nodes, 2 agents, 6 people)")
    print("  Rewards: coverage (0.5) + rescue (5.0) + redundancy (1.0)")
    print("  Penalties: hp_loss (0.002)")
    print("="*80 + "\n")
    
    # Create config
    config = PPOConfig(
        scenario="office",
        experiment_name="phase3_with_redundancy",
        seed=42,
        num_agents=2,
        lr_policy=3e-4,
        lr_value=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.1,
        num_iterations=100,
        steps_per_rollout=100,
        num_ppo_epochs=1,
        batch_size=64,
        batch_rollout_size=1,
        eval_interval=50,
        num_eval_episodes=5,
        log_interval=5,
        checkpoint_interval=50,
        clip_epsilon=0.2,
        max_grad_norm=0.5,
    )
    
    # Create trainer
    print("[1] Initializing trainer...")
    trainer = EnhancedPPOTrainer(config)
    
    # Load Phase 2 best model as initialization
    print("[1b] Loading Phase 2 best model...")
    import glob
    phase2_logs = sorted(glob.glob("logs/phase2_mild_hp_penalty_*/checkpoints/best_model.pt"))
    if phase2_logs:
        best_model_path = phase2_logs[-1]
        print(f"    Loaded from: {best_model_path}")
        trainer.load_checkpoint(best_model_path)
    else:
        print("    WARNING: No Phase 2 checkpoint found, trying Phase 1...")
        phase1_logs = sorted(glob.glob("logs/phase1_simplified_rewards_*/checkpoints/best_model.pt"))
        if phase1_logs:
            best_model_path = phase1_logs[-1]
            print(f"    Loaded from: {best_model_path}")
            trainer.load_checkpoint(best_model_path)
    
    # Override with Phase 3 reward shaper
    print("[2] Setting up Phase 3 rewards...")
    reward_shaper = RewardShaper.for_scenario(
        scenario="office",
        curriculum_phase=3
    )
    # Keep HP penalty reduced
    reward_shaper.w_hp_loss = 0.002
    trainer.reward_shaper = reward_shaper
    
    print(f"    - coverage: {reward_shaper.w_coverage}")
    print(f"    - rescue: {reward_shaper.w_rescue}")
    print(f"    - hp_loss: {reward_shaper.w_hp_loss} (REDUCED)")
    print(f"    - time: {reward_shaper.w_time}")
    print(f"    - redundancy: {reward_shaper.w_redundancy}")
    
    # Train
    print("\n[3] Starting PPO training...\n")
    trainer.train()
    
    print("\n" + "="*80)
    print("PHASE 3 TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_phase3()
