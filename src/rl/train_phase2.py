#!/usr/bin/env python3
"""
Curriculum Phase 2: Add mild HP penalty to test multi-objective learning

Configuration:
  - Scenario: office (11 nodes, 2 agents, 6 people)
  - Rewards: coverage (0.5) + rescue (5.0)
  - Small penalty: hp_loss (0.02)
  - No time/redundancy penalties yet
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppo_config import PPOConfig
from reward_shaper import RewardShaper
from enhanced_training import EnhancedPPOTrainer


def train_phase2():
    """Run PPO training on Phase 2 environment"""
    
    print("\n" + "="*80)
    print("CURRICULUM LEARNING: PHASE 2 PPO TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  Scenario: office (11 nodes, 2 agents, 6 people)")
    print("  Rewards: coverage (0.5) + rescue (5.0)")
    print("  Small penalty: hp_loss (0.02)")
    print("  No time/redundancy penalties")
    print("="*80 + "\n")
    
    # Create config
    config = PPOConfig(
        scenario="office",
        experiment_name="phase2_mild_hp_penalty",
        seed=42,
        num_agents=2,
        lr_policy=3e-4,            # Keep policy learning rate stable
        lr_value=1e-4,             # Keep critic learning slow
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,         # Keep entropy moderate
        value_loss_coef=0.1,       # Keep critic weight low
        num_iterations=100,        # Run 100 iterations for phase 2 convergence
        steps_per_rollout=100,
        num_ppo_epochs=1,          # Keep 1 epoch to stay in trust region
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
    
    # Load Phase 1 best model as initialization (transfer learning)
    print("[1b] Loading Phase 1 best model...")
    import os
    import glob
    phase1_logs = sorted(glob.glob("logs/phase1_simplified_rewards_*/checkpoints/best_model.pt"))
    if phase1_logs:
        best_model_path = phase1_logs[-1]  # Most recent
        print(f"    Loaded from: {best_model_path}")
        trainer.load_checkpoint(best_model_path)
    else:
        print("    WARNING: No Phase 1 checkpoint found, starting from random")
    
    # Override with Phase 2 reward shaper
    print("[2] Setting up Phase 2 rewards...")
    reward_shaper = RewardShaper.for_scenario(
        scenario="office",
        curriculum_phase=2
    )
    # CRITICAL: Reduce HP penalty from 0.02 to 0.002 to avoid overwhelming rescue signal
    # 0.02 was causing net negative returns (return ~-5), 0.002 gives ~0.2 penalty per timestep
    # vs rescue reward of 5.0, which allows learning to continue
    reward_shaper.w_hp_loss = 0.002
    trainer.reward_shaper = reward_shaper
    
    print(f"    - coverage: {reward_shaper.w_coverage}")
    print(f"    - rescue: {reward_shaper.w_rescue}")
    print(f"    - hp_loss: {reward_shaper.w_hp_loss} (REDUCED for stability)")
    print(f"    - time: {reward_shaper.w_time}")
    print(f"    - redundancy: {reward_shaper.w_redundancy}")
    
    # Train
    print("\n[3] Starting PPO training...\n")
    trainer.train()
    
    print("\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_phase2()
