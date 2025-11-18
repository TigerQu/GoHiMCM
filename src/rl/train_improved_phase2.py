#!/usr/bin/env python3
"""
IMPROVED PHASE 2 TRAINING: Daycare and Warehouse with Rescue Focus

Building on Phase 1 (exploration), Phase 2 adds:
- HP penalties (small, to introduce consequence)
- Time penalties (small, to encourage urgency)
- HIGHER rescue rewards (200+) to outweigh penalties
- On-the-way rewards for approaching people
- Enhanced sweeping rewards

Scenarios:
- Daycare: Smaller layout, more people with varied mobility
- Warehouse: Larger layout, hazards, HP decay

Key Changes from Phase 1:
- Rescue rewards >> penalties (positive bias)
- Potential shaping guides toward people
- First-visit bonus remains high (150)
- Anti-loop mechanisms still active
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.ppo_config import PPOConfig
from rl.improved_reward_shaper import ImprovedRewardShaper
from rl.enhanced_training import EnhancedPPOTrainer


class Phase2Trainer(EnhancedPPOTrainer):
    """
    Phase 2 trainer with rescue focus.
    
    Maintains exploration while adding rescue objectives.
    """
    
    def __init__(self, config: PPOConfig, epsilon_explore: float = 0.15, temperature: float = 1.3):
        super().__init__(config)
        self.epsilon_explore = epsilon_explore
        self.temperature = temperature
        
        # Use Phase 2 reward shaper (tuned hyperparameters)
        self.reward_shaper = ImprovedRewardShaper.for_scenario(
            config.scenario,
            curriculum_phase=2  # Phase 2: rescue focus with small penalties
        )
        
        print(f"\n🚀 PHASE 2 TRAINER INITIALIZED:")
        print(f"  Scenario: {config.scenario}")
        print(f"  Epsilon-greedy: {epsilon_explore*100:.0f}% random actions")
        print(f"  Temperature: {temperature}")
        print(f"  Entropy coef: {config.entropy_coef}")
        print(f"  Reward weights:")
        print(f"    - First visit: +{self.reward_shaper.w_first_visit}")
        print(f"    - Rescue: +{self.reward_shaper.w_rescue}")
        print(f"    - HP loss: {self.reward_shaper.w_hp_loss}")
        print(f"    - Time: {self.reward_shaper.w_time}/step")
        print(f"    - Backtrack: {self.reward_shaper.w_backtrack}")


def train_phase2(scenario: str = "daycare", num_agents: int = 2, num_iterations: int = 3000):
    """
    Train Phase 2 with rescue focus.
    
    Args:
        scenario: 'daycare' or 'warehouse'
        num_agents: Number of agents (2-4 recommended)
        num_iterations: Training iterations (3000+ recommended for Phase 2)
    """
    
    print("="*80)
    print(f"IMPROVED PHASE 2 TRAINING: {scenario.upper()} - {num_agents} AGENTS")
    print("="*80)
    print("\n🎯 Phase 2 Objectives:")
    print("  1. Maintain exploration (from Phase 1)")
    print("  2. Learn to rescue people efficiently")
    print("  3. Balance coverage vs. rescue priority")
    print("\n💰 Reward Structure (POSITIVE BIAS):")
    print("  ✅ Rescue: +200 per person")
    print("  ✅ First visit: +150 per node")
    print("  ✅ Coverage: +10 per sweep")
    print("  ✅ Potential shaping: guides toward people")
    print("  ⚠️  HP loss: -0.01 (small penalty)")
    print("  ⚠️  Time: -0.005/step (small penalty)")
    print("  ❌ Backtrack: -15 (anti-loop)")
    print("\n📊 Environment:")
    print(f"  - Scenario: {scenario}")
    print(f"  - Agents: {num_agents}")
    print(f"  - Iterations: {num_iterations}")
    print("="*80 + "\n")
    
    # Configuration optimized for Phase 2
    config = PPOConfig(
        scenario=scenario,
        experiment_name=f"improved_phase2_{scenario}_{num_agents}agents",
        seed=42,
        num_agents=num_agents,
        
        # === BALANCED EXPLORATION (reduced from Phase 1) ===
        entropy_coef=0.08,             # Slightly lower than Phase 1 (was 0.1)
        lr_policy=3e-4,
        lr_value=1e-4,
        
        # PPO stability
        gamma=0.99,                    # Standard discount
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        
        # Training parameters (longer for rescue learning)
        num_iterations=num_iterations,
        steps_per_rollout=150,         # Longer episodes for rescue
        num_ppo_epochs=4,
        batch_size=64,
        batch_rollout_size=4,
        
        # Evaluation
        eval_interval=50,
        num_eval_episodes=5,
        log_interval=100,              # Console output every 100 iters
        checkpoint_interval=50,
    )
    
    # Create Phase 2 trainer
    print("[1] Initializing Phase 2 trainer...")
    trainer = Phase2Trainer(
        config,
        epsilon_explore=0.15,  # 15% random (less than Phase 1's 20%)
        temperature=1.3,       # Softer than greedy, firmer than Phase 1
    )
    
    print("\n[2] Starting training...\n")
    print("="*80)
    print("TRAINING PROGRESS (detailed metrics saved to CSV)")
    print("="*80)
    print("Iter  | Return  | Coverage | Rescued | P_loss  V_loss")
    print("-"*80)
    
    # Train
    try:
        trainer.train()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        
        # Get final stats
        log_dir = trainer.logger.log_dir
        print(f"\n📁 Results saved to: {log_dir}")
        print(f"   - Best checkpoint: {log_dir}/checkpoints/best_model.pt")
        print(f"   - Metrics CSV: {log_dir}/metrics.csv")
        print(f"   - Plots: {log_dir}/plots/")
        print(f"   - Trajectory videos: {log_dir}/eval_agent_trajectories_iter*.png")
        
        # Test the trained model
        print("\n" + "="*80)
        print("🧪 TESTING TRAINED MODEL (10 episodes)")
        print("="*80)
        
        test_episodes = 10
        test_results = {
            'returns': [],
            'coverage': [],
            'rescued': [],
            'alive': [],
            'found': [],
            'steps': [],
        }
        
        for ep in range(test_episodes):
            env = trainer.env
            obs = env.reset()
            trainer.reward_shaper.reset()
            
            done = False
            ep_return = 0.0
            step = 0
            
            while not done and step < 300:
                # Get valid actions
                valid_actions_list = [env.get_valid_actions(i) for i in range(config.num_agents)]
                agent_indices = torch.tensor([env.get_agent_node_index(i) for i in range(config.num_agents)],
                                            device=trainer.device)
                
                # Select actions (deterministic for testing)
                with torch.no_grad():
                    actions, _, _ = trainer.policy.select_actions(
                        obs, agent_indices, valid_actions_list, deterministic=True
                    )
                
                # Map to action strings
                action_strs = {}
                for i, (valid_actions, action_idx) in enumerate(zip(valid_actions_list, actions)):
                    sorted_actions = sorted(valid_actions)
                    if action_idx.item() < len(sorted_actions):
                        action_strs[i] = sorted_actions[action_idx.item()]
                    else:
                        action_strs[i] = 'wait'
                
                # Execute
                obs, _, done, info = env.do_action(action_strs)
                reward = trainer.reward_shaper.compute_reward(env, action_strs)
                ep_return += reward
                step += 1
            
            # Record results
            stats = env.get_statistics()
            
            test_results['returns'].append(ep_return)
            test_results['coverage'].append(stats.get('nodes_swept', 0))
            test_results['rescued'].append(stats.get('people_rescued', 0))
            test_results['alive'].append(stats.get('people_alive', 0))
            test_results['found'].append(stats.get('people_found', 0))
            test_results['steps'].append(step)
            
            total_people = stats.get('people_rescued', 0) + stats.get('people_alive', 0) + stats.get('people_dead', 0)
            print(f"Episode {ep+1:2d}: Return={ep_return:7.1f}, Coverage={stats['nodes_swept']:2d}, "
                  f"Rescued={stats['people_rescued']:2d}/{total_people}, "
                  f"Alive={stats['people_alive']:2d}, Steps={step:3d}")
        
        # Summary
        print("\n" + "="*80)
        print("📈 TEST SUMMARY (10 episodes)")
        print("="*80)
        print(f"Average return:   {np.mean(test_results['returns']):7.1f} ± {np.std(test_results['returns']):6.1f}")
        print(f"Average coverage: {np.mean(test_results['coverage']):7.1f} ± {np.std(test_results['coverage']):6.1f}")
        print(f"Average rescued:  {np.mean(test_results['rescued']):7.1f} ± {np.std(test_results['rescued']):6.1f}")
        print(f"Average alive:    {np.mean(test_results['alive']):7.1f} ± {np.std(test_results['alive']):6.1f}")
        print(f"Average found:    {np.mean(test_results['found']):7.1f} ± {np.std(test_results['found']):6.1f}")
        print(f"Average steps:    {np.mean(test_results['steps']):7.1f} ± {np.std(test_results['steps']):6.1f}")
        
        # Verdict
        print("\n" + "="*80)
        print("🎯 PHASE 2 VERDICT:")
        print("="*80)
        
        avg_rescued = np.mean(test_results['rescued'])
        avg_coverage = np.mean(test_results['coverage'])
        
        # Get total people from last episode
        total_people = (test_results['rescued'][-1] + test_results['alive'][-1] + 
                       (stats.get('people_dead', 0) if 'stats' in locals() else 0))
        
        rescue_rate = (avg_rescued / total_people * 100) if total_people > 0 else 0
        
        if rescue_rate >= 70 and avg_coverage >= 15:
            print("✅ EXCELLENT! High rescue rate and good coverage.")
            print(f"   - Rescue rate: {rescue_rate:.1f}%")
            print(f"   - Coverage: {avg_coverage:.1f} nodes")
            print("   → Ready for Phase 3 (add redundancy) or deployment!")
        elif rescue_rate >= 50 or avg_coverage >= 12:
            print("⚠️  GOOD PROGRESS. Agents learning rescue behavior.")
            print(f"   - Rescue rate: {rescue_rate:.1f}%")
            print(f"   - Coverage: {avg_coverage:.1f} nodes")
            print("   → Consider extending training or tuning")
        else:
            print("❌ NEEDS MORE TRAINING. Rescue/coverage still low.")
            print(f"   - Rescue rate: {rescue_rate:.1f}%")
            print(f"   - Coverage: {avg_coverage:.1f} nodes")
            print("   → Extend iterations or adjust reward weights")
        
        print("\n" + "="*80)
        
        return {
            'scenario': scenario,
            'num_agents': num_agents,
            'avg_return': np.mean(test_results['returns']),
            'avg_rescued': avg_rescued,
            'avg_coverage': avg_coverage,
            'rescue_rate': rescue_rate,
        }
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(f"{trainer.logger.log_dir}/checkpoints/interrupted.pt")
        print("Checkpoint saved!")
        return None
    
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


def compare_scenarios(scenarios=["daycare", "warehouse"], num_agents=2, iterations=2000):
    """
    Compare performance across daycare and warehouse scenarios.
    
    Args:
        scenarios: List of scenarios to test
        num_agents: Number of agents
        iterations: Iterations per scenario
    """
    print("\n" + "="*80)
    print("🔬 PHASE 2 SCENARIO COMPARISON")
    print("="*80)
    print(f"Testing scenarios: {scenarios}")
    print(f"Agents: {num_agents}")
    print(f"Iterations per scenario: {iterations}")
    print("="*80 + "\n")
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Testing {scenario.upper()} scenario")
        print(f"{'='*80}\n")
        
        result = train_phase2(scenario=scenario, num_agents=num_agents, num_iterations=iterations)
        
        if result:
            results.append(result)
        
        print(f"\n✓ Completed {scenario}\n")
    
    # Summary comparison
    if results:
        print("\n" + "="*80)
        print("📊 SCENARIO COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Scenario':<12} | {'Return':<10} | {'Rescued':<10} | {'Coverage':<10} | {'Rescue %'}")
        print("-"*80)
        
        for r in results:
            print(f"{r['scenario']:<12} | {r['avg_return']:<10.1f} | "
                  f"{r['avg_rescued']:<10.1f} | {r['avg_coverage']:<10.1f} | "
                  f"{r['rescue_rate']:.1f}%")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved Phase 2 policy with rescue focus')
    parser.add_argument('--scenario', type=str, default='daycare', 
                       choices=['daycare', 'warehouse'],
                       help='Scenario to train (default: daycare)')
    parser.add_argument('--agents', type=int, default=2, 
                       help='Number of agents (default: 2)')
    parser.add_argument('--iterations', type=int, default=3000, 
                       help='Training iterations (default: 3000)')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare daycare vs warehouse')
    parser.add_argument('--compare-scenarios', type=str, nargs='+', 
                       default=['daycare', 'warehouse'],
                       help='Scenarios to compare (default: daycare warehouse)')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run scenario comparison
        compare_scenarios(
            scenarios=args.compare_scenarios,
            num_agents=args.agents,
            iterations=args.iterations
        )
    else:
        # Single training run
        train_phase2(
            scenario=args.scenario,
            num_agents=args.agents,
            num_iterations=args.iterations
        )
