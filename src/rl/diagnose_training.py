"""
Training diagnosis script - understand why model isn't learning.

Checks:
1. Reward distribution
2. Action diversity
3. Environment dynamics
4. Value function accuracy
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from collections import Counter

from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


def diagnose_training(scenario: str = "office", num_rollouts: int = 10):
    """Run diagnosis on training setup."""
    
    print("=" * 70)
    print("TRAINING DIAGNOSIS")
    print("=" * 70 + "\n")
    
    # Setup trainer
    config = PPOConfig.get_default(scenario)
    config.num_iterations = 1  # Just for setup
    trainer = EnhancedPPOTrainer(config)
    
    print(f"Scenario: {scenario}")
    print(f"Agents: {config.num_agents}")
    print(f"Max actions: {config.max_actions}\n")
    
    # Collect multiple rollouts
    print(f"Collecting {num_rollouts} rollouts for analysis...\n")
    
    all_returns = []
    all_rewards = []
    all_actions = []
    all_episode_stats = []
    
    for i in range(num_rollouts):
        rollout = trainer.collect_rollout(
            num_steps=100,
            layout_seed=1000 + i,
            deterministic=False
        )
        
        episode_return = rollout['episode_return']
        rewards = rollout['rewards'].cpu().numpy()
        actions = rollout['actions'].cpu().numpy()
        
        all_returns.append(episode_return)
        all_rewards.extend(rewards)
        all_actions.extend(actions.flatten())
        all_episode_stats.append(rollout['episode_stats'])
        
        print(f"  Rollout {i+1}: return={episode_return:.2f}, "
              f"rescued={rollout['episode_stats']['people_rescued']}, "
              f"found={rollout['episode_stats']['people_found']}, "
              f"swept={rollout['episode_stats']['nodes_swept']}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULTS")
    print("=" * 70 + "\n")
    
    # 1. Return analysis
    print("1. RETURN ANALYSIS")
    print(f"   Mean return: {np.mean(all_returns):.2f}")
    print(f"   Std return: {np.std(all_returns):.2f}")
    print(f"   Min return: {np.min(all_returns):.2f}")
    print(f"   Max return: {np.max(all_returns):.2f}")
    
    if np.mean(all_returns) < 0:
        print("   ⚠️  WARNING: Negative average return!")
        print("   → Model is being penalized more than rewarded")
        print("   → Check reward weights (especially time penalty)")
    
    # 2. Reward distribution
    print("\n2. REWARD DISTRIBUTION")
    print(f"   Mean reward per step: {np.mean(all_rewards):.3f}")
    print(f"   Positive steps: {np.sum(np.array(all_rewards) > 0)} / {len(all_rewards)} "
          f"({np.sum(np.array(all_rewards) > 0) / len(all_rewards) * 100:.1f}%)")
    print(f"   Negative steps: {np.sum(np.array(all_rewards) < 0)} / {len(all_rewards)} "
          f"({np.sum(np.array(all_rewards) < 0) / len(all_rewards) * 100:.1f}%)")
    print(f"   Zero steps: {np.sum(np.array(all_rewards) == 0)} / {len(all_rewards)}")
    
    if np.mean(all_rewards) < 0:
        print("   ⚠️  WARNING: Negative average reward per step!")
        print("   → Time penalty may be too high")
        print("   → Or coverage/rescue rewards too low")
    
    # 3. Action diversity
    print("\n3. ACTION DIVERSITY")
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    print(f"   Total actions taken: {total_actions}")
    print(f"   Unique actions: {len(action_counts)}")
    print(f"   Action distribution:")
    
    action_names = ["wait", "search"] + [f"move_{i}" for i in range(config.max_actions - 2)]
    for action_idx in sorted(action_counts.keys()):
        count = action_counts[action_idx]
        pct = count / total_actions * 100
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"action_{action_idx}"
        print(f"      {action_name}: {count} ({pct:.1f}%)")
    
    # Check for action collapse
    most_common_action, most_common_count = action_counts.most_common(1)[0]
    if most_common_count / total_actions > 0.7:
        print(f"   ⚠️  WARNING: Action collapse detected!")
        print(f"   → {most_common_count/total_actions*100:.1f}% of actions are '{action_names[most_common_action]}'")
        print(f"   → Model not exploring enough")
        print(f"   → Increase entropy_coef (currently {config.entropy_coef})")
    
    # 4. Performance metrics
    print("\n4. PERFORMANCE METRICS")
    avg_rescued = np.mean([s['people_rescued'] for s in all_episode_stats])
    avg_found = np.mean([s['people_found'] for s in all_episode_stats])
    avg_swept = np.mean([s['nodes_swept'] for s in all_episode_stats])
    total_people = all_episode_stats[0].get('total_people', 'unknown')
    
    print(f"   People rescued: {avg_rescued:.1f} / {total_people}")
    print(f"   People found: {avg_found:.1f} / {total_people}")
    print(f"   Rooms swept: {avg_swept:.1f}")
    
    if avg_rescued == 0:
        print("   ❌ CRITICAL: No people rescued in ANY episode!")
        print("   → Model hasn't learned rescue behavior")
        print("   → Possible causes:")
        print("      1. Reward for rescue too small")
        print("      2. Time penalty too high")
        print("      3. Not enough training iterations")
        print("      4. Learning rate too low")
    elif avg_rescued < total_people * 0.3:
        print(f"   ⚠️  WARNING: Low rescue rate (<30%)")
        print("   → Model needs more training")
    else:
        print(f"   ✅ Rescue rate: {avg_rescued/total_people*100:.1f}%")
    
    # 5. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70 + "\n")
    
    if avg_rescued == 0:
        print("🔧 URGENT FIXES NEEDED:")
        print("   1. Increase rescue reward: weight_rescue = 50.0 (currently 20.0)")
        print("   2. Reduce time penalty: weight_time = 0.01 (currently 0.02)")
        print("   3. Increase training: num_iterations = 10000+ (currently", config.num_iterations, ")")
        print("   4. Increase learning rate: lr_policy = 1e-3 (currently", config.lr_policy, ")")
        print("   5. More exploration: entropy_coef = 0.05 (currently", config.entropy_coef, ")")
    elif np.mean(all_returns) < 0:
        print("⚠️  ISSUES DETECTED:")
        print("   1. Reduce time penalty (negative returns)")
        print("   2. Increase coverage/rescue rewards")
    else:
        print("✅ Training setup looks reasonable")
        print("   Continue training for more iterations")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "office"
    num_rollouts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    diagnose_training(scenario, num_rollouts)
