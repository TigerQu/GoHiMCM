#!/usr/bin/env python3
"""
PHASE 2 TRAINING: Daycare and Warehouse with Rescue Objectives

Building on Phase 1's exploration foundation, Phase 2 adds:
1. HP degradation and time penalties
2. INCREASED rescue rewards (to outweigh penalties)
3. On-the-way bonuses for approaching people
4. Sweeping rewards with progress tracking

Scenarios:
- Daycare: Multi-floor, moderate complexity, children with varying mobility
- Warehouse: Large single-floor, high-risk zones, industrial hazards

Key Changes from Phase 1:
- Add rescue objectives (people with HP)
- Time penalty and HP degradation
- HIGHER positive rewards (rescue, sweeping, approaching)
- Distance-based dense rewards for guidance
- Multi-scenario support
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


class Phase2RewardShaper(ImprovedRewardShaper):
    """
    Phase 2 reward shaper with BOOSTED positive rewards.
    
    Increases rescue and sweeping rewards to heavily outweigh time/HP penalties.
    """
    
    def __init__(
        self,
        scenario: str = "daycare",
        # BOOSTED positive rewards
        weight_first_visit: float = 200.0,     # INCREASED from 100 (exploration)
        weight_coverage: float = 15.0,         # INCREASED from 10 (sweeping)
        weight_rescue: float = 500.0,          # MASSIVELY INCREASED from 200 (rescue!)
        weight_person_found: float = 100.0,    # NEW: bonus for finding people
        weight_approaching: float = 5.0,       # NEW: dense reward when getting closer
        
        # Time and HP penalties (kept low)
        weight_hp_loss: float = 0.5,           # HP degradation penalty
        weight_time: float = 0.1,              # Time penalty per step
        
        # Anti-loop penalties (from Phase 1)
        weight_backtrack: float = -20.0,       # INCREASED penalty (was -10)
        weight_wait: float = -0.2,             # Penalty for WAIT action
        
        # Exploration bonuses
        weight_potential: float = 10.0,        # INCREASED from 5
        weight_redundancy: float = 75.0,       # INCREASED from 50 (high-risk rooms)
        
        # Hyperparameters
        backtrack_window: int = 5,
        gamma: float = 0.99,
    ):
        super().__init__(
            scenario=scenario,
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
        
        # New weights for Phase 2
        self.w_person_found = weight_person_found
        self.w_approaching = weight_approaching
        
        # Track previous distances to people for approaching reward
        self.prev_min_distance_to_people = {}
    
    
    def reset(self):
        """Reset episode-level tracking."""
        super().reset()
        self.prev_min_distance_to_people = {}
    
    
    def compute_reward(self, env, actions: Dict[int, str]) -> float:
        """
        Compute reward with BOOSTED positive rewards.
        
        Phase 2 focuses on:
        1. Heavy rescue bonuses (500 per person!)
        2. Finding people bonuses (100)
        3. Approaching people bonuses (dense)
        4. Strong sweeping rewards (200 first visit, 15 repeat)
        5. Low penalties (time, HP) to encourage action
        """
        reward = 0.0
        stats = env.get_statistics()
        
        # === 1. FIRST-VISIT BONUS (exploration) ===
        first_visit_reward = self._first_visit_coverage_reward(env, actions)
        reward += first_visit_reward
        
        # === 2. COVERAGE REWARD (sweeping already-visited nodes) ===
        coverage_reward = 0.0
        for agent_id in actions:
            node_id = env.agents[agent_id].node_id
            if node_id in env.swept_nodes:
                coverage_reward += self.w_coverage
        reward += coverage_reward
        
        # === 3. RESCUE REWARD (MASSIVE BONUS) ===
        rescue_reward = stats.get('people_rescued', 0) * self.w_rescue
        reward += rescue_reward
        
        # === 4. PERSON FOUND BONUS (discovering people) ===
        person_found_reward = 0.0
        prev_found = getattr(self, 'prev_people_found', 0)
        curr_found = stats.get('people_found', 0)
        if curr_found > prev_found:
            new_found = curr_found - prev_found
            person_found_reward = new_found * self.w_person_found
        self.prev_people_found = curr_found
        reward += person_found_reward
        
        # === 5. APPROACHING REWARD (dense guidance toward people) ===
        approaching_reward = self._approaching_reward(env, actions)
        reward += approaching_reward
        
        # === 6. HP LOSS PENALTY (small) ===
        hp_loss_reward = 0.0
        prev_hp = getattr(self, 'prev_total_hp', sum(p.hp for p in env.people.values()))
        curr_hp = sum(p.hp for p in env.people.values())
        hp_loss = max(0, prev_hp - curr_hp)
        hp_loss_reward = -hp_loss * self.w_hp_loss
        self.prev_total_hp = curr_hp
        reward += hp_loss_reward
        
        # === 7. TIME PENALTY (small) ===
        time_penalty = -self.w_time
        reward += time_penalty
        
        # === 8. BACKTRACK PENALTY (anti-loop) ===
        backtrack_penalty = self._backtrack_penalty(env, actions)
        reward += backtrack_penalty
        
        # === 9. WAIT PENALTY ===
        wait_penalty = self._wait_penalty(actions)
        reward += wait_penalty
        
        # === 10. POTENTIAL SHAPING (guide toward objectives) ===
        potential_reward = self._potential_shaping_reward(env, actions)
        reward += potential_reward
        
        # === 11. REDUNDANCY BONUS (high-risk rooms) ===
        redundancy_reward = stats.get('high_risk_redundancy', 0) * self.w_redundancy
        reward += redundancy_reward
        
        return reward
    
    
    def _approaching_reward(self, env, actions: Dict[int, str]) -> float:
        """
        Dense reward for getting closer to people.
        
        Provides gradient toward rescue objectives.
        """
        reward = 0.0
        
        # Compute shortest path distances to all unrescued people
        for agent_id in actions:
            agent_node = env.agents[agent_id].node_id
            
            # Find closest unrescued person
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
            
            # Check if we got closer
            if min_dist < float('inf'):
                prev_dist = self.prev_min_distance_to_people.get(agent_id, min_dist)
                
                if min_dist < prev_dist:
                    # Got closer! Reward proportional to improvement
                    improvement = prev_dist - min_dist
                    reward += improvement * self.w_approaching
                
                self.prev_min_distance_to_people[agent_id] = min_dist
        
        return reward
    
    
    @classmethod
    def for_phase2(cls, scenario: str) -> 'Phase2RewardShaper':
        """
        Factory method for Phase 2 training.
        
        Args:
            scenario: 'daycare' or 'warehouse'
        """
        if scenario == "daycare":
            # Daycare: Multi-floor, children with varying mobility
            return cls(
                scenario="daycare",
                weight_first_visit=200.0,    # Strong exploration
                weight_coverage=15.0,        # Sweeping reward
                weight_rescue=500.0,         # HUGE rescue bonus
                weight_person_found=100.0,   # Discovery bonus
                weight_approaching=5.0,      # Dense guidance
                weight_hp_loss=0.5,          # Small HP penalty
                weight_time=0.1,             # Small time penalty
                weight_backtrack=-20.0,      # Anti-loop
                weight_wait=-0.2,
                weight_potential=10.0,
                weight_redundancy=75.0,
            )
        
        elif scenario == "warehouse":
            # Warehouse: Large single-floor, high-risk zones
            return cls(
                scenario="warehouse",
                weight_first_visit=250.0,    # Very strong exploration (large space)
                weight_coverage=20.0,        # Higher sweeping (more nodes)
                weight_rescue=600.0,         # MASSIVE rescue bonus (harder to find)
                weight_person_found=150.0,   # Higher discovery (sparse people)
                weight_approaching=8.0,      # Stronger guidance (large distances)
                weight_hp_loss=1.0,          # Higher HP penalty (high-risk zones)
                weight_time=0.15,            # Higher time penalty (urgency)
                weight_backtrack=-25.0,      # Stronger anti-loop (large space)
                weight_wait=-0.3,
                weight_potential=15.0,       # Stronger shaping
                weight_redundancy=100.0,     # Higher redundancy (critical zones)
            )
        
        else:
            return cls(scenario=scenario)


class Phase2Trainer(EnhancedPPOTrainer):
    """Enhanced PPO trainer for Phase 2."""
    
    def __init__(self, config: PPOConfig, reward_shaper: Phase2RewardShaper):
        super().__init__(config)
        self.reward_shaper = reward_shaper
        
        print(f"\n🚀 PHASE 2 TRAINER INITIALIZED:")
        print(f"  Scenario: {config.scenario}")
        print(f"  Agents: {config.num_agents}")
        print(f"  Reward weights:")
        print(f"    • Rescue: {reward_shaper.w_rescue}")
        print(f"    • First visit: {reward_shaper.w_first_visit}")
        print(f"    • Person found: {reward_shaper.w_person_found}")
        print(f"    • Approaching: {reward_shaper.w_approaching}")
        print(f"    • HP loss: -{reward_shaper.w_hp_loss}")
        print(f"    • Time: -{reward_shaper.w_time}")


def train_phase2(scenario: str = "daycare", num_agents: int = 3, num_iterations: int = 3000):
    """
    Train Phase 2 with rescue objectives.
    
    Args:
        scenario: 'daycare' or 'warehouse'
        num_agents: Number of agents
        num_iterations: Training iterations
    """
    
    print("="*80)
    print(f"PHASE 2 TRAINING: {scenario.upper()} SCENARIO")
    print("="*80)
    print("\n🎯 Objectives:")
    print("  1. Explore and sweep building (boosted rewards)")
    print("  2. Find and rescue people (MASSIVE bonuses)")
    print("  3. Minimize HP loss and time (small penalties)")
    print("\n💪 Boosted Rewards:")
    print("  ✓ Rescue: +500 (daycare) / +600 (warehouse)")
    print("  ✓ Person found: +100 / +150")
    print("  ✓ Approaching people: +5 / +8 per step closer")
    print("  ✓ First visit: +200 / +250")
    print("  ✓ Coverage: +15 / +20")
    print("\n⏰ Penalties (kept low):")
    print("  • HP loss: -0.5 / -1.0 per HP")
    print("  • Time: -0.1 / -0.15 per step")
    print("  • Backtrack: -20 / -25")
    print(f"\n📊 Environment: {scenario} | {num_agents} agents | {num_iterations} iterations")
    print("="*80 + "\n")
    
    # Configuration
    config = PPOConfig(
        scenario=scenario,
        experiment_name=f"phase2_{scenario}_{num_agents}agents",
        seed=42,
        num_agents=num_agents,
        
        # Exploration settings
        entropy_coef=0.08,           # High exploration (complex scenarios)
        lr_policy=3e-4,
        lr_value=1e-4,
        
        # PPO stability
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        
        # Training parameters
        num_iterations=num_iterations,
        steps_per_rollout=150,       # Longer episodes for rescue
        num_ppo_epochs=4,
        batch_size=64,
        batch_rollout_size=4,
        
        # Evaluation
        eval_interval=100,
        num_eval_episodes=5,
        log_interval=50,             # Less frequent console output
        checkpoint_interval=100,
    )
    
    # Create Phase 2 reward shaper
    print("[1] Initializing Phase 2 reward shaper...")
    reward_shaper = Phase2RewardShaper.for_phase2(scenario)
    
    # Create trainer
    print("[2] Initializing trainer...")
    trainer = Phase2Trainer(config, reward_shaper)
    
    print("\n[3] Starting training...\n")
    print("="*80)
    print("TRAINING PROGRESS (detailed metrics saved to CSV)")
    print("="*80)
    print("Iter  | Return  | Swept | Rescued | Found | HP Loss | Loops")
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
        print(f"   - Metrics: {log_dir}/metrics.csv")
        print(f"   - Plots: {log_dir}/plots/")
        print(f"   - Trajectories: {log_dir}/eval_agent_trajectories_*.png")
        
        # Test the trained model
        print("\n" + "="*80)
        print("🧪 TESTING TRAINED MODEL (5 episodes)")
        print("="*80)
        
        test_episodes = 5
        test_results = {
            'returns': [],
            'coverage': [],
            'rescued': [],
            'found': [],
            'alive': [],
            'hp_final': [],
        }
        
        for ep in range(test_episodes):
            env = trainer.env
            obs = env.reset()
            trainer.reward_shaper.reset()
            
            done = False
            ep_return = 0.0
            step = 0
            
            initial_hp = sum(p.hp for p in env.people.values())
            
            while not done and step < 200:
                # Get valid actions
                valid_actions_list = [env.get_valid_actions(i) for i in range(config.num_agents)]
                agent_indices = torch.tensor([env.get_agent_node_index(i) for i in range(config.num_agents)])
                
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
            final_hp = sum(p.hp for p in env.people.values())
            
            test_results['returns'].append(ep_return)
            test_results['coverage'].append(stats['nodes_swept'])
            test_results['rescued'].append(stats['people_rescued'])
            test_results['found'].append(stats['people_found'])
            test_results['alive'].append(stats['people_alive'])
            test_results['hp_final'].append(final_hp)
            
            hp_loss = initial_hp - final_hp
            print(f"Ep {ep+1}: Return={ep_return:.0f}, Swept={stats['nodes_swept']}, "
                  f"Rescued={stats['people_rescued']}, Found={stats['people_found']}, "
                  f"Alive={stats['people_alive']}, HP_loss={hp_loss:.0f}")
        
        # Summary
        print("\n" + "="*80)
        print("📈 TEST SUMMARY (5 episodes)")
        print("="*80)
        print(f"Average return:  {np.mean(test_results['returns']):7.1f} ± {np.std(test_results['returns']):.1f}")
        print(f"Average swept:   {np.mean(test_results['coverage']):7.1f}")
        print(f"Average rescued: {np.mean(test_results['rescued']):7.1f}")
        print(f"Average found:   {np.mean(test_results['found']):7.1f}")
        print(f"Average alive:   {np.mean(test_results['alive']):7.1f}")
        
        # Verdict
        print("\n" + "="*80)
        print("🎯 VERDICT:")
        print("="*80)
        
        avg_rescued = np.mean(test_results['rescued'])
        avg_alive = np.mean(test_results['alive'])
        total_people = len(env.people)
        
        rescue_rate = (avg_rescued / total_people * 100) if total_people > 0 else 0
        survival_rate = (avg_alive / total_people * 100) if total_people > 0 else 0
        
        if rescue_rate >= 80 and survival_rate >= 90:
            print("✅ EXCELLENT! High rescue and survival rates.")
            print(f"   - Rescue: {rescue_rate:.1f}%")
            print(f"   - Survival: {survival_rate:.1f}%")
        elif rescue_rate >= 60 or survival_rate >= 75:
            print("⚠️  GOOD. Decent performance but room for improvement.")
            print(f"   - Rescue: {rescue_rate:.1f}%")
            print(f"   - Survival: {survival_rate:.1f}%")
        else:
            print("❌ NEEDS IMPROVEMENT.")
            print(f"   - Rescue: {rescue_rate:.1f}%")
            print(f"   - Survival: {survival_rate:.1f}%")
            print("   → Consider more training or tuning rewards")
        
        print("\n" + "="*80)
        
        return {
            'scenario': scenario,
            'num_agents': num_agents,
            'avg_return': np.mean(test_results['returns']),
            'avg_rescued': avg_rescued,
            'avg_alive': avg_alive,
            'rescue_rate': rescue_rate,
            'survival_rate': survival_rate,
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


def compare_scenarios_and_agents():
    """
    Compare performance across scenarios and agent counts.
    """
    print("\n" + "="*80)
    print("🔬 PHASE 2 COMPARISON STUDY")
    print("="*80)
    print("Testing: daycare (2,3,4 agents) + warehouse (3,4,5 agents)")
    print("Iterations: 2000 per config (faster comparison)")
    print("="*80 + "\n")
    
    results = []
    
    # Daycare with 2, 3, 4 agents
    for num_agents in [2, 3, 4]:
        print(f"\n{'='*80}")
        print(f"DAYCARE with {num_agents} agents")
        print(f"{'='*80}\n")
        
        result = train_phase2(scenario="daycare", num_agents=num_agents, num_iterations=2000)
        if result:
            results.append(result)
    
    # Warehouse with 3, 4, 5 agents
    for num_agents in [3, 4, 5]:
        print(f"\n{'='*80}")
        print(f"WAREHOUSE with {num_agents} agents")
        print(f"{'='*80}\n")
        
        result = train_phase2(scenario="warehouse", num_agents=num_agents, num_iterations=2000)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Scenario':<12} | {'Agents':<7} | {'Return':<10} | {'Rescued':<9} | {'Alive':<9} | {'Rescue%':<9}")
    print("-"*80)
    
    for r in results:
        print(f"{r['scenario']:<12} | {r['num_agents']:<7} | {r['avg_return']:<10.0f} | "
              f"{r['avg_rescued']:<9.1f} | {r['avg_alive']:<9.1f} | {r['rescue_rate']:<9.1f}%")
    
    # Best configurations
    if results:
        best_daycare = max([r for r in results if r['scenario'] == 'daycare'], 
                          key=lambda x: x['rescue_rate'], default=None)
        best_warehouse = max([r for r in results if r['scenario'] == 'warehouse'], 
                            key=lambda x: x['rescue_rate'], default=None)
        
        print("\n" + "="*80)
        if best_daycare:
            print(f"🏆 BEST DAYCARE: {best_daycare['num_agents']} agents "
                  f"(Rescue: {best_daycare['rescue_rate']:.1f}%)")
        if best_warehouse:
            print(f"🏆 BEST WAREHOUSE: {best_warehouse['num_agents']} agents "
                  f"(Rescue: {best_warehouse['rescue_rate']:.1f}%)")
        print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phase 2: Rescue objectives')
    parser.add_argument('--scenario', type=str, default='daycare', 
                       choices=['daycare', 'warehouse'],
                       help='Scenario to train (default: daycare)')
    parser.add_argument('--agents', type=int, default=3,
                       help='Number of agents (default: 3)')
    parser.add_argument('--iterations', type=int, default=3000,
                       help='Training iterations (default: 3000)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare scenarios and agent counts')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison study
        compare_scenarios_and_agents()
    else:
        # Single training run
        train_phase2(
            scenario=args.scenario,
            num_agents=args.agents,
            num_iterations=args.iterations
        )
