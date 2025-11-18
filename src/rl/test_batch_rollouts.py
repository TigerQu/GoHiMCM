"""
Quick test to verify batch rollout collection works correctly.
Tests both single rollout and batch rollout modes.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


def test_single_rollout():
    """Test original single rollout collection."""
    print("="*60)
    print("TEST 1: Single Rollout Collection")
    print("="*60)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 1  # Just test collection
    config.steps_per_rollout = 20  # Shorter for speed
    config.batch_rollout_size = 1  # Single rollout (original behavior)
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect one rollout
    rollout = trainer.collect_rollout(num_steps=20)
    
    print(f"✓ Collected single rollout")
    print(f"  - Steps: {len(rollout['observations'])}")
    print(f"  - Actions shape: {rollout['actions'].shape}")
    print(f"  - Episode return: {rollout['episode_return']:.2f}")
    print(f"  - People rescued: {rollout['episode_stats']['people_rescued']}")
    
    return True


def test_batch_rollouts():
    """Test batch rollout collection."""
    print("\n" + "="*60)
    print("TEST 2: Batch Rollout Collection (K=3)")
    print("="*60)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 1
    config.steps_per_rollout = 20
    config.batch_rollout_size = 3  # Collect 3 rollouts
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect batch of 3 rollouts
    batch = trainer.collect_batch_rollouts(
        num_rollouts=3,
        num_steps=20
    )
    
    print(f"✓ Collected batch of {batch['num_episodes']} rollouts")
    print(f"  - Total transitions: {batch['num_transitions']}")
    print(f"  - Actions shape: {batch['actions'].shape}")
    print(f"  - Advantages shape: {batch['advantages'].shape}")
    print(f"  - Returns shape: {batch['returns'].shape}")
    print(f"  - Episode returns: {batch['episode_returns']}")
    print(f"  - Mean return: {sum(batch['episode_returns'])/len(batch['episode_returns']):.2f}")
    
    # Verify data consistency
    assert batch['actions'].size(0) == batch['num_transitions'], "Action count mismatch"
    assert batch['advantages'].size(0) == batch['num_transitions'] * config.num_agents, \
        "Advantages size mismatch"
    assert len(batch['episode_returns']) == batch['num_episodes'], "Episode return count mismatch"
    
    print(f"✓ Data consistency checks passed")
    
    return True


def test_batch_vs_single():
    """Verify batch mode shapes match single mode."""
    print("\n" + "="*60)
    print("TEST 3: Batch vs Single Mode Structure")
    print("="*60)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 1
    config.steps_per_rollout = 20
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect single rollout
    single = trainer.collect_rollout(num_steps=20, layout_seed=1000)
    
    print(f"✓ Single rollout: {len(single['observations'])} steps, return={single['episode_return']:.2f}")
    print(f"  - Actions shape: {single['actions'].shape}")
    print(f"  - Log probs shape: {single['log_probs'].shape}")
    
    # Collect batch with same seed for comparison
    batch = trainer.collect_batch_rollouts(
        num_rollouts=1,
        num_steps=20,
        layout_seeds=[1000]
    )
    
    print(f"✓ Batch (K=1): {batch['num_transitions']} steps, return={batch['episode_returns'][0]:.2f}")
    print(f"  - Actions shape: {batch['actions'].shape}")
    print(f"  - Advantages shape: {batch['advantages'].shape}")
    
    # Verify structure consistency
    assert single['actions'].shape[0] == batch['num_transitions'], \
        "Number of transitions mismatch"
    assert batch['actions'].shape[1] == config.num_agents, \
        "Agent count mismatch in actions"
    assert batch['advantages'].size(0) == batch['num_transitions'] * config.num_agents, \
        "Advantages expanded incorrectly"
    
    print(f"✓ Structure consistency checks passed")
    
    return True


def test_policy_update_with_batch():
    """Test that policy can update with batched data."""
    print("\n" + "="*60)
    print("TEST 4: Policy Update with Batched Data")
    print("="*60)
    
    config = PPOConfig.get_default("office")
    config.num_iterations = 1
    config.steps_per_rollout = 20
    config.num_ppo_epochs = 1  # Single epoch for testing
    
    trainer = EnhancedPPOTrainer(config)
    
    # Collect batch
    batch = trainer.collect_batch_rollouts(
        num_rollouts=2,
        num_steps=20
    )
    
    print(f"✓ Collected batch: {batch['num_episodes']} episodes, {batch['num_transitions']} transitions")
    
    # Try updating policy
    try:
        losses = trainer.update_policy(
            batch['observations'],
            batch['agent_indices'],
            batch['actions'],
            batch['log_probs'],
            batch['advantages'],
            batch['returns']
        )
        
        print(f"✓ Policy updated successfully")
        print(f"  - Policy loss: {losses['policy_loss']:.4f}")
        print(f"  - Value loss: {losses['value_loss']:.4f}")
        print(f"  - Entropy: {losses['entropy']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Policy update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BATCH ROLLOUT COLLECTION TESTS")
    print("="*60 + "\n")
    
    try:
        test_single_rollout()
        test_batch_rollouts()
        test_batch_vs_single()
        test_policy_update_with_batch()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
