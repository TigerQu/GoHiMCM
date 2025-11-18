#!/usr/bin/env python3
"""
FINAL ENVIRONMENT TRAINING: Use Phase 3-trained model on warehouse scenario.

Configuration:
  - Scenario: warehouse (single-floor grid: 4x6 halls + 12 rooms)
  - Base model: Phase 3 checkpoint (curriculum-trained)
  - Rewards: coverage (5.0) + rescue (5.0) + redundancy (5.0) + time (-0.01)
  - Penalties: hp_loss (0.002) - reduced from curriculum
  - Visualization: ENABLED - saves agent trajectory maps during eval

Metrics tracked:
  - Return per episode
  - People rescued per episode
  - Room coverage (nodes swept)
  - Redundancy (high-risk rooms swept 2+ times)
  - Approx KL, clip_fraction, entropy (training diagnostics)

Output:
  - Best model checkpoint: logs/final_environment_full_rewards_*/checkpoints/best_model.pt
  - Episode metrics: logs/final_environment_full_rewards_*/metrics.csv
  - Trajectory visualizations: logs/final_environment_full_rewards_*/trajectories/*.png
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppo_config import PPOConfig
from reward_shaper import RewardShaper
from enhanced_training import EnhancedPPOTrainer
from final_env_visualizer import FinalEnvironmentVisualizer


def train_final_environment():
    """Run PPO training on final warehouse environment with full rewards"""
    
    print("\n" + "="*80)
    print("FINAL ENVIRONMENT TRAINING: WAREHOUSE WITH FULL REWARDS")
    print("="*80)
    print("\nWarehouse Scenario (Single-Floor Grid):")
    print("  - Grid: 4×6 halls (H_r_c format)")
    print("  - Rooms: 3×5 cargo/lab rooms (R_r_c format)")
    print("  - Labs (high-risk): 5 designated rooms require 2 sweeps each")
    print("  - Exits: LEFT (EXIT_WH_LEFT) and RIGHT (EXIT_WH_RIGHT)")
    print("  - Agents: 2 (starting at opposite exits)")
    print("\nReward Configuration:")
    print("  - Coverage: 5.0 (sweep prioritization)")
    print("  - Rescue: 5.0 (find and evacuate people)")
    print("  - Redundancy: 5.0 (cover high-risk rooms 2+ times)")
    print("  - Time: -0.01 (encourage efficiency)")
    print("  - HP loss penalty: 0.002 (REDUCED - stabilized)")
    print("\nVisualization Output:")
    print("  - Trajectory maps: Agent paths overlaid on warehouse grid")
    print("  - Node visit heatmaps: Shows coverage frequency")
    print("  - Metrics plots: Returns, rescues, coverage over training")
    print("="*80 + "\n")
    
    # Create config for warehouse environment
    config = PPOConfig(
        scenario="warehouse",  # Full warehouse environment (single floor)
        experiment_name="final_environment_full_rewards",
        seed=42,
        num_agents=2,
        # Policy/Value networks
        lr_policy=3e-4,
        lr_value=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.1,
        # Training parameters
        num_iterations=200,  # Full training run
        steps_per_rollout=100,
        num_ppo_epochs=1,
        batch_size=64,
        batch_rollout_size=1,
        # Evaluation & checkpointing
        eval_interval=50,     # Eval every 50 iterations (saves visualizations)
        num_eval_episodes=5,
        log_interval=10,
        checkpoint_interval=50,
        # PPO clipping & stability
        clip_epsilon=0.2,
        max_grad_norm=0.5,
    )
    
    # Create trainer
    print("[1] Initializing trainer for warehouse environment...")
    trainer = EnhancedPPOTrainer(config)
    
    # Load Phase 3 best model as initialization
    print("[1b] Loading Phase 3 best checkpoint...")
    import glob
    phase3_logs = sorted(glob.glob("logs/phase3_with_redundancy_*/checkpoints/best_model.pt"))
    if phase3_logs:
        best_model_path = phase3_logs[-1]
        print(f"    ✓ Loaded from: {best_model_path}")
        trainer.load_checkpoint(best_model_path)
    else:
        print("    ⚠ No Phase 3 checkpoint found, trying Phase 2...")
        phase2_logs = sorted(glob.glob("logs/phase2_mild_hp_penalty_*/checkpoints/best_model.pt"))
        if phase2_logs:
            best_model_path = phase2_logs[-1]
            print(f"    ✓ Loaded from: {best_model_path}")
            trainer.load_checkpoint(best_model_path)
        else:
            print("    ⚠ No Phase 2 checkpoint found, trying Phase 1...")
            phase1_logs = sorted(glob.glob("logs/phase1_simplified_rewards_*/checkpoints/best_model.pt"))
            if phase1_logs:
                best_model_path = phase1_logs[-1]
                print(f"    ✓ Loaded from: {best_model_path}")
                trainer.load_checkpoint(best_model_path)
    
    # Configure final environment rewards
    print("\n[2] Setting up FINAL environment rewards...")
    reward_shaper = RewardShaper(
        scenario="warehouse",
        weight_coverage=5.0,       # Prioritize sweeping entire warehouse
        weight_rescue=5.0,         # Rescue people equally important
        weight_hp_loss=0.002,      # Reduced - stabilized penalty
        weight_time=-0.01,         # Small efficiency penalty
        weight_redundancy=5.0,     # Bonus for covering lab rooms twice
    )
    trainer.reward_shaper = reward_shaper
    
    print(f"    Coverage reward: {reward_shaper.w_coverage}")
    print(f"    Rescue reward: {reward_shaper.w_rescue}")
    print(f"    Redundancy reward: {reward_shaper.w_redundancy}")
    print(f"    HP loss penalty: {reward_shaper.w_hp_loss}")
    print(f"    Time penalty: {reward_shaper.w_time}")
    
    # Override evaluate to add visualization
    original_evaluate = trainer.evaluate
    eval_count = [0]  # Counter for visualization
    
    def evaluate_with_visualization(num_episodes=None):
        """Evaluate and save trajectory visualizations."""
        results = original_evaluate(num_episodes)
        
        # Generate trajectory visualization
        try:
            traj_dir = os.path.join(trainer.logger.exp_dir, "trajectories")
            traj_data = FinalEnvironmentVisualizer.plot_agent_trajectories(
                trainer,
                episode_idx=eval_count[0],
                max_steps=config.steps_per_rollout,
                deterministic=True,
                save_dir=traj_dir
            )
            print(f"\n  Trajectory visualization saved:")
            print(f"    - Coverage: {traj_data['coverage']} nodes")
            print(f"    - Rescued: {traj_data['rescued']} people")
            print(f"    - Steps: {traj_data['steps']}")
            eval_count[0] += 1
        except Exception as e:
            print(f"\n  ⚠ Warning: Trajectory visualization failed: {e}")
        
        return results
    
    trainer.evaluate = evaluate_with_visualization
    
    # Train
    print("\n[3] Starting FINAL environment training...")
    print("    - Diagnostics printed every 10 iterations")
    print("    - Evaluation & trajectory visualization every 50 iterations")
    print("    - Checkpoints saved every 50 iterations")
    print("    - Best model will be saved automatically\n")
    trainer.train()
    
    print("\n" + "="*80)
    print("FINAL ENVIRONMENT TRAINING COMPLETE")
    print("="*80)
    print("\nOutput Files:")
    print(f"  📁 Main directory: {trainer.logger.exp_dir}")
    print(f"  📊 Best model: {trainer.checkpoint_dir}/best_model.pt")
    print(f"  📈 Metrics CSV: {trainer.logger.exp_dir}/metrics.csv")
    print(f"  🗺️  Trajectories: {trainer.logger.exp_dir}/trajectories/")
    print(f"  📝 Config: {trainer.logger.exp_dir}/config.json")
    print("\nKey Metrics to Check:")
    print("  - Final return (should be > 50)")
    print("  - Coverage: nodes swept (target: 15+ out of 27 total)")
    print("  - Rescues: people evacuated (target: 4+ out of 6)")
    print("  - Redundancy: high-risk labs covered 2+ times")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_final_environment()
