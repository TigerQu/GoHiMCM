#!/usr/bin/env python3
"""
PHASE 2 TRAINING: Daycare with Rescue Objectives

Building on Phase 1's exploration foundation, Phase 2 focuses on daycare:
1. Multi-floor building with children
2. HP degradation and time penalties
3. INCREASED rescue rewards (to outweigh penalties)
4. On-the-way bonuses for approaching people
5. Sweeping rewards with progress tracking

Key Features:
- Children with varying mobility (child, adult caregivers)
- Multi-floor layout (3 floors)
- Moderate complexity
- Focus on finding and rescuing vulnerable populations
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict
import networkx as nx


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.ppo_config import PPOConfig
from rl.improved_reward_shaper import ImprovedRewardShaper
from rl.enhanced_training import EnhancedPPOTrainer


class Phase2DaycareRewardShaper(ImprovedRewardShaper):
    """
    Phase 2 reward shaper for DAYCARE scenario with BOOSTED positive rewards.
    """
    
    def __init__(
        self,
        # BOOSTED positive rewards for daycare
        weight_first_visit: float = 200.0,     # Strong exploration
        weight_coverage: float = 15.0,         # Sweeping reward
        weight_rescue: float = 500.0,          # HUGE rescue bonus
        weight_person_found: float = 100.0,    # Discovery bonus
        weight_approaching: float = 5.0,       # Dense guidance
        
        # Time and HP penalties (kept low)
        weight_hp_loss: float = 0.5,           # HP degradation penalty
        weight_time: float = 0.1,              # Time penalty per step
        
        # Anti-loop penalties
        weight_backtrack: float = -20.0,
        weight_wait: float = -0.2,
        
        # Exploration bonuses
        weight_potential: float = 10.0,
        weight_redundancy: float = 75.0,
        
        backtrack_window: int = 5,
        gamma: float = 0.99,
    ):
        super().__init__(
            scenario="daycare",
            weight_first_visit=weight_first_visit,
            weight_coverage=weight_coverage,
            weight_rescue=weight_rescue,
            weight_hp_loss=weight_hp_loss,
            weight_backtrack=weight_backtrack,
            weight_wait=weight_wait,
            weight_time=weight_time,
            weight_potential=weight_potential,
            weight_redundancy=weight_redundancy,
            backtrack_window=backtrack_window,
            gamma=gamma,
        )
        
        self.w_person_found = weight_person_found
        self.w_approaching = weight_approaching
        self.prev_min_distance_to_people = {}
    
    
    def reset(self):
        """Reset episode-level tracking."""
        super().reset()
        self.prev_min_distance_to_people = {}
    
    
    def compute_reward(self, env, actions: Dict[int, str]) -> float:
        """Compute reward with BOOSTED positive rewards for daycare."""
        # Use parent's compute_reward for Bug 1 & 2 fixes, then add extras
        reward = super().compute_reward(env, actions)
        
        stats = env.get_statistics()
        
        # Add daycare-specific rewards
        # 1. Person found bonus (not in parent)
        person_found_reward = 0.0
        prev_found = getattr(self, 'prev_people_found', 0)
        curr_found = stats.get('people_found', 0)
        if curr_found > prev_found:
            new_found = curr_found - prev_found
            person_found_reward = new_found * self.w_person_found
        self.prev_people_found = curr_found
        reward += person_found_reward
        
        # 2. Approaching reward (not in parent)
        approaching_reward = self._approaching_reward(env)
        reward += approaching_reward
        
        return reward
    
    
    def _approaching_reward(self, env) -> float:
        """Dense reward for getting closer to people."""
        reward = 0.0
        
        for agent_id in env.agents.keys():
            agent_node = env.agents[agent_id].node_id
            min_dist = float('inf')
            
            for person_id, person in env.people.items():
                if person.rescued:
                    continue
                
                person_node = person.node_id
                
                try:
                    dist = nx.shortest_path_length(env.G, agent_node, person_node)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    continue
            
            if min_dist < float('inf'):
                prev_dist = self.prev_min_distance_to_people.get(agent_id, min_dist)
                
                if min_dist < prev_dist:
                    improvement = prev_dist - min_dist
                    reward += improvement * self.w_approaching
                
                self.prev_min_distance_to_people[agent_id] = min_dist
        
        return reward


class Phase2DaycareTrainer(EnhancedPPOTrainer):
    """Enhanced PPO trainer for Phase 2 Daycare."""
    
    def __init__(self, config: PPOConfig, reward_shaper: Phase2DaycareRewardShaper):
        super().__init__(config)
        self.reward_shaper = reward_shaper
        
        print(f"\n🚀 PHASE 2 DAYCARE TRAINER INITIALIZED:")
        print(f"  Scenario: daycare")
        print(f"  Agents: {config.num_agents}")
        print(f"  Reward weights:")
        print(f"    • Rescue: {reward_shaper.w_rescue}")
        print(f"    • First visit: {reward_shaper.w_first_visit}")
        print(f"    • Person found: {reward_shaper.w_person_found}")
        print(f"    • Approaching: {reward_shaper.w_approaching}")
        print(f"    • HP loss: -{reward_shaper.w_hp_loss}")
        print(f"    • Time: -{reward_shaper.w_time}")


def train_phase2_daycare(num_agents: int = 3, num_iterations: int = 3000):
    """
    Train Phase 2 on daycare with rescue objectives.
    
    Args:
        num_agents: Number of agents (default: 3, recommended: 2-4)
        num_iterations: Training iterations (default: 3000)
    """
    
    # Memory management for daycare environment
    import torch
    if torch.cuda.is_available():
        # Clear CUDA cache before training
        torch.cuda.empty_cache()
        # Enable memory efficient settings
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU
        # Set memory allocator to reduce fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("🔧 GPU Memory Management:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print()
    
    print("="*80)
    print(f"PHASE 2 TRAINING: DAYCARE SCENARIO ({num_agents} AGENTS)")
    print("="*80)
    print("\n🏫 Daycare Features:")
    print("  • Multi-floor building (3 floors)")
    print("  • Children with varying mobility")
    print("  • Adult caregivers")
    print("  • Moderate complexity")
    print("\n🎯 Objectives:")
    print("  1. Explore and sweep building (boosted rewards)")
    print("  2. Find and rescue children (MASSIVE bonuses)")
    print("  3. Minimize HP loss and time (small penalties)")
    print("\n💪 Boosted Rewards:")
    print("  ✓ Rescue: +500 per person")
    print("  ✓ Person found: +100")
    print("  ✓ Approaching people: +5 per step closer")
    print("  ✓ First visit: +200")
    print("  ✓ Coverage: +15")
    print("\n⏰ Penalties (kept low):")
    print("  • HP loss: -0.5 per HP")
    print("  • Time: -0.1 per step")
    print("  • Backtrack: -20")
    print(f"\n📊 Training: {num_agents} agents | {num_iterations} iterations")
    print("="*80 + "\n")
    
    # Configuration
    config = PPOConfig(
        scenario="daycare",
        experiment_name=f"phase2_daycare_{num_agents}agents",
        seed=42,
        num_agents=num_agents,
        
        # Exploration settings
        entropy_coef=0.08,
        lr_policy=3e-4,
        lr_value=1e-4,
        
        # PPO stability
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        
        # Training parameters (heavily memory-optimized for shared GPU)
        num_iterations=num_iterations,
        steps_per_rollout=100,       # Reduced from 150 (memory)
        num_ppo_epochs=2,            # Reduced from 3 (memory)
        batch_size=24,               # Reduced from 48 (shared GPU)
        batch_rollout_size=2,        # Reduced from 3 (memory)
        
        # Evaluation and checkpointing (save frequently!)
        eval_interval=50,            # More frequent eval
        num_eval_episodes=3,         # Fewer episodes (faster)
        log_interval=25,             # More frequent logging
        checkpoint_interval=25,      # Save every 25 iters (not 100!)
    )
    
    # Create Phase 2 Daycare reward shaper
    print("[1] Initializing Phase 2 Daycare reward shaper...")
    reward_shaper = Phase2DaycareRewardShaper()
    
    # Create trainer
    print("[2] Initializing trainer...")
    trainer = Phase2DaycareTrainer(config, reward_shaper)
    
    # Save initial checkpoint (untrained model)
    import torch
    from pathlib import Path
    checkpoint_dir = Path(trainer.logger.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = checkpoint_dir / "initial_model.pt"
    
    checkpoint = {
        'iteration': 0,
        'policy_state': trainer.policy.state_dict(),
        'value_state': trainer.value.state_dict(),
        'policy_optimizer_state': trainer.policy_optimizer.state_dict(),
        'value_optimizer_state': trainer.value_optimizer.state_dict(),
        'config': trainer.config.to_dict(),
        'best_eval_return': trainer.best_eval_return,
        'scaler_state': trainer.scaler.state_dict() if trainer.scaler else None,
        'extra': {},
    }
    torch.save(checkpoint, str(initial_checkpoint))
    print(f"✅ Saved initial checkpoint: {initial_checkpoint}")
    
    print("\n[3] Starting training...\n")
    print("="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    print("Iter  | Return  | Swept | Rescued | Found | P_loss  V_loss")
    print("-"*80)
    
    # Train
    try:
        trainer.train()
        
        # Clean up GPU memory after training
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n🔧 GPU Memory after training:")
            print(f"   Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"   Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        print("\n" + "="*80)
        print("✅ PHASE 2 DAYCARE TRAINING COMPLETE!")
        print("="*80)
        
        log_dir = trainer.logger.log_dir
        print(f"\n📁 Results saved to: {log_dir}")
        print(f"   - Best checkpoint: {log_dir}/checkpoints/best_model.pt")
        print(f"   - Metrics: {log_dir}/metrics.csv")
        
        # Test
        print("\n" + "="*80)
        print("🧪 TESTING TRAINED MODEL (5 episodes)")
        print("="*80)
        
        test_results = []
        
        for ep in range(5):
            env = trainer.env
            obs = env.reset()
            trainer.reward_shaper.reset()
            
            done = False
            ep_return = 0.0
            step = 0
            initial_hp = sum(p.hp for p in env.people.values())
            
            while not done and step < 200:
                valid_actions_list = [env.get_valid_actions(i) for i in range(config.num_agents)]
                agent_indices_list = [env.get_agent_node_index(i) for i in range(config.num_agents)]
                if None in agent_indices_list:
                    print(f"Warning: Agent position error at step {step}, skipping episode")
                    break
                agent_indices = torch.tensor(agent_indices_list)
                
                with torch.no_grad():
                    actions, _, _ = trainer.policy.select_actions(
                        obs, agent_indices, valid_actions_list, deterministic=True
                    )
                
                action_strs = {}
                for i, (valid_actions, action_idx) in enumerate(zip(valid_actions_list, actions)):
                    sorted_actions = sorted(valid_actions)
                    if action_idx.item() < len(sorted_actions):
                        action_strs[i] = sorted_actions[action_idx.item()]
                    else:
                        action_strs[i] = 'wait'
                
                obs, _, done, info = env.do_action(action_strs)
                reward = trainer.reward_shaper.compute_reward(env, action_strs)
                ep_return += reward
                step += 1
            
            stats = env.get_statistics()
            final_hp = sum(p.hp for p in env.people.values())
            hp_loss = initial_hp - final_hp
            
            test_results.append({
                'return': ep_return,
                'swept': stats['nodes_swept'],
                'rescued': stats['people_rescued'],
                'found': stats['people_found'],
                'alive': stats['people_alive'],
                'hp_loss': hp_loss
            })
            
            print(f"Ep {ep+1}: Return={ep_return:.0f}, Swept={stats['nodes_swept']}, "
                  f"Rescued={stats['people_rescued']}, Found={stats['people_found']}, "
                  f"Alive={stats['people_alive']}, HP_loss={hp_loss:.0f}")
        
        # Summary
        print("\n" + "="*80)
        print("📈 TEST SUMMARY")
        print("="*80)
        avg_rescued = np.mean([r['rescued'] for r in test_results])
        avg_alive = np.mean([r['alive'] for r in test_results])
        total_people = len(env.people)
        
        print(f"Average return:  {np.mean([r['return'] for r in test_results]):7.1f}")
        print(f"Average swept:   {np.mean([r['swept'] for r in test_results]):7.1f}")
        print(f"Average rescued: {avg_rescued:7.1f} / {total_people} ({100*avg_rescued/total_people:.1f}%)")
        print(f"Average found:   {np.mean([r['found'] for r in test_results]):7.1f}")
        print(f"Average alive:   {avg_alive:7.1f} / {total_people} ({100*avg_alive/total_people:.1f}%)")
        
        print("\n" + "="*80)
        
        return test_results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
        trainer.save_checkpoint(f"{trainer.logger.log_dir}/checkpoints/interrupted.pt")
        return None
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def compare_agent_counts(agent_counts=[2, 3, 4, 5], iterations_per_config=2000):
    """Compare performance across different agent counts for daycare."""
    
    print("\n" + "="*80)
    print("🔬 PHASE 2 DAYCARE: AGENT COUNT COMPARISON")
    print("="*80)
    print(f"Testing: {agent_counts}")
    print(f"Iterations: {iterations_per_config} per config")
    print("="*80 + "\n")
    
    results = []
    
    for num_agents in agent_counts:
        print(f"\n{'='*80}")
        print(f"Testing {num_agents} agents")
        print(f"{'='*80}\n")
        
        test_results = train_phase2_daycare(num_agents=num_agents, num_iterations=iterations_per_config)
        
        if test_results:
            avg_rescued = np.mean([r['rescued'] for r in test_results])
            avg_alive = np.mean([r['alive'] for r in test_results])
            results.append({
                'agents': num_agents,
                'rescued': avg_rescued,
                'alive': avg_alive,
            })
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("📊 COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Agents':<8} | {'Rescued':<10} | {'Alive':<10}")
        print("-"*80)
        
        for r in results:
            print(f"{r['agents']:<8} | {r['rescued']:<10.1f} | {r['alive']:<10.1f}")
        
        best = max(results, key=lambda x: x['rescued'])
        print(f"\n🏆 BEST: {best['agents']} agents (Rescued: {best['rescued']:.1f})")
        print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phase 2: Daycare with rescue objectives')
    parser.add_argument('--agents', type=int, default=3,
                       help='Number of agents (default: 3, recommended: 2-5)')
    parser.add_argument('--iterations', type=int, default=3000,
                       help='Training iterations (default: 3000)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different agent counts')
    parser.add_argument('--compare-agents', type=int, nargs='+', default=[2, 3, 4, 5],
                       help='Agent counts to compare (default: 2 3 4 5)')
    parser.add_argument('--compare-iterations', type=int, default=2000,
                       help='Iterations per config when comparing (default: 2000)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_agent_counts(
            agent_counts=args.compare_agents,
            iterations_per_config=args.compare_iterations
        )
    else:
        train_phase2_daycare(num_agents=args.agents, num_iterations=args.iterations)
