"""
Quick test to verify comprehensive iteration diagnostics work.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


def test_iteration_diagnostics():
    """Test comprehensive iteration diagnostics printing."""
    print("\n" + "="*80)
    print("TESTING COMPREHENSIVE ITERATION DIAGNOSTICS")
    print("="*80)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 2
    config.steps_per_rollout = 15
    config.log_interval = 1  # Log every iteration for testing
    config.batch_rollout_size = 1  # Single rollout mode
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect a rollout
    print("\nCollecting rollout...")
    rollout = trainer.collect_rollout(num_steps=15)
    
    # Compute advantages
    print("Computing advantages...")
    advantages, returns = trainer.compute_advantages(
        rollout['rewards'],
        rollout['dones'],
        rollout['values'],
        rollout['final_value']
    )
    
    # Update policy
    print("Updating policy...")
    losses = trainer.update_policy(
        rollout['observations'],
        rollout['agent_indices'],
        rollout['actions'],
        rollout['log_probs'],
        advantages,
        returns
    )
    
    # Print comprehensive diagnostics
    print("\n" + "="*80)
    print("PRINTING COMPREHENSIVE DIAGNOSTICS")
    print("="*80)
    
    trainer.print_iteration_metrics(
        iteration=0,
        rollout=rollout,
        batch=None,
        losses=losses,
        advantages=advantages
    )
    
    print("\n✓ Diagnostics printed successfully!")
    
    return True


def test_batch_diagnostics():
    """Test diagnostics with batch rollouts."""
    print("\n" + "="*80)
    print("TESTING BATCH DIAGNOSTICS")
    print("="*80)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 2
    config.steps_per_rollout = 15
    config.batch_rollout_size = 2
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect batch
    print("\nCollecting batch of 2 rollouts...")
    batch = trainer.collect_batch_rollouts(
        num_rollouts=2,
        num_steps=15
    )
    
    # Update policy
    print("Updating policy with batch data...")
    losses = trainer.update_policy(
        batch['observations'],
        batch['agent_indices'],
        batch['actions'],
        batch['log_probs'],
        batch['advantages'],
        batch['returns']
    )
    
    # Extract first episode for diagnostics
    steps_per_ep = batch['num_transitions'] // batch['num_episodes']
    first_rollout = {
        'actions': batch['actions'][:steps_per_ep],
        'rewards': batch['rewards'][:steps_per_ep],
        'values': batch['values'][:steps_per_ep],
        'episode_return': batch['episode_returns'][0],
        'episode_stats': batch['episode_stats'][0],
    }
    
    # Print diagnostics
    print("\n" + "="*80)
    print("BATCH MODE DIAGNOSTICS (showing first episode)")
    print("="*80)
    
    trainer.print_iteration_metrics(
        iteration=0,
        rollout=first_rollout,
        batch=batch,
        losses=losses,
        advantages=batch['advantages'][:steps_per_ep]
    )
    
    print("\n✓ Batch diagnostics printed successfully!")
    
    return True


if __name__ == "__main__":
    try:
        test_iteration_diagnostics()
        test_batch_diagnostics()
        
        print("\n" + "="*80)
        print("ALL DIAGNOSTIC TESTS PASSED ✓")
        print("="*80)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
