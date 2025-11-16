"""
Performance monitoring and benchmarking script.

Helps identify bottlenecks in training.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


def benchmark_components(scenario: str = "office"):
    """Benchmark individual components to identify bottlenecks."""
    
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60 + "\n")
    
    config = PPOConfig.get_default(scenario)
    config.num_iterations = 10  # Only for testing
    
    print(f"Initializing trainer...")
    trainer = EnhancedPPOTrainer(config)
    
    # Test 1: Rollout collection
    print("\n1. Testing rollout collection (100 steps)...")
    start = time.time()
    rollout = trainer.collect_rollout(num_steps=100, layout_seed=42)
    rollout_time = time.time() - start
    print(f"   ✓ Rollout time: {rollout_time:.2f}s")
    print(f"   ✓ Steps collected: {len(rollout['rewards'])}")
    print(f"   ✓ Time per step: {rollout_time / len(rollout['rewards']) * 1000:.1f}ms")
    
    # Test 2: Advantage computation
    print("\n2. Testing advantage computation...")
    start = time.time()
    advantages, returns = trainer.compute_advantages(
        rollout['rewards'],
        rollout['dones'],
        rollout['values'],
        rollout['final_value']
    )
    adv_time = time.time() - start
    print(f"   ✓ Advantage computation: {adv_time * 1000:.1f}ms")
    
    # Test 3: Policy update (single epoch)
    print("\n3. Testing policy update (1 epoch)...")
    start = time.time()
    losses = trainer.update_policy(
        rollout['observations'],
        rollout['agent_indices'],
        rollout['actions'],
        rollout['log_probs'],
        advantages,
        returns
    )
    update_time = time.time() - start
    print(f"   ✓ Policy update time: {update_time:.2f}s")
    print(f"   ✓ Policy loss: {losses['policy_loss']:.4f}")
    print(f"   ✓ Value loss: {losses['value_loss']:.4f}")
    
    # Test 4: Full iteration
    print("\n4. Estimated times:")
    single_epoch_time = update_time / config.num_ppo_epochs
    full_iter_time = rollout_time + (single_epoch_time * config.num_ppo_epochs)
    print(f"   ✓ Single PPO epoch: {single_epoch_time:.2f}s")
    print(f"   ✓ All {config.num_ppo_epochs} epochs: {single_epoch_time * config.num_ppo_epochs:.2f}s")
    print(f"   ✓ Full iteration: {full_iter_time:.2f}s")
    
    # Test 5: Projected training time
    print("\n5. Projected training time:")
    total_time = full_iter_time * config.num_iterations
    print(f"   ✓ Total iterations: {config.num_iterations}")
    print(f"   ✓ Time per iteration: {full_iter_time:.2f}s")
    print(f"   ✓ Total training time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")
    
    # Test 6: GPU info
    print("\n6. GPU Information:")
    if torch.cuda.is_available():
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   ✓ Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   ✓ Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("   ✗ CUDA not available (running on CPU)")
    
    # Bottleneck analysis
    print("\n7. Bottleneck Analysis:")
    rollout_pct = (rollout_time / full_iter_time) * 100
    update_pct = ((single_epoch_time * config.num_ppo_epochs) / full_iter_time) * 100
    
    print(f"   • Environment rollout: {rollout_pct:.1f}% of iteration time")
    print(f"   • Policy updates: {update_pct:.1f}% of iteration time")
    
    if rollout_pct > 70:
        print("\n   ⚠️  BOTTLENECK: Environment interaction is the slowest part")
        print("   → This is normal and expected")
        print("   → Consider reducing steps_per_rollout for faster iterations")
    elif update_pct > 70:
        print("\n   ⚠️  BOTTLENECK: Policy updates are slow")
        print("   → Check GPU utilization with nvidia-smi")
        print("   → Consider reducing num_ppo_epochs")
    else:
        print("\n   ✓ Balanced workload between environment and training")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if full_iter_time > 5:
        print("\n⏰ Training is slow (>5s per iteration)")
        print("\nQuick fixes:")
        print("  1. Reduce steps_per_rollout: 100 → 50")
        print("  2. Reduce num_ppo_epochs: 4 → 2")
        print("  3. Use quick_train.py for fast prototyping")
        print("\nRun: python src/rl/quick_train.py office")
    else:
        print("\n✅ Training speed is good (<5s per iteration)")
        print("   Continue with current settings")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "office"
    benchmark_components(scenario)
