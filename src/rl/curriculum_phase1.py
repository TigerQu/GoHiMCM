"""
Curriculum Learning Phase 1: Simplified Environment Diagnostic

Tests whether PPO can learn on a SIMPLE task before tackling the full problem.

Phase 1 uses ONLY:
- Coverage reward (0.5 weight)
- Rescue reward (5.0 weight)
- NO HP loss, NO time penalty, NO redundancy bonus

Strategy: Start with small/empty building (no hazards) to isolate:
1. Can PPO learn at all? (If no → pipeline issue)
2. Can it learn basic coverage/rescue? (If yes → ready for Phase 2)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.layouts import build_standard_office_layout
from reward_shaper import RewardShaper
from ppo_config import PPOConfig
from enhanced_training import EnhancedPPOTrainer


def run_curriculum_phase1_diagnostic():
    """
    Run Phase 1 curriculum diagnostic: simple environment with only coverage+rescue rewards.
    
    Returns:
        results (Dict): Results including improvement metric
    """
    print("\n" + "="*80)
    print("CURRICULUM PHASE 1: SIMPLIFIED ENVIRONMENT DIAGNOSTIC")
    print("="*80)
    print("\nPhase 1 Configuration:")
    print("  - Rewards: coverage (0.5) + rescue (5.0)")
    print("  - NO HP loss, NO time penalty, NO redundancy")
    print("="*80 + "\n")
    
    # ===== STEP 1: Create environment =====
    print("\n[STEP 1] Creating environment for Phase 1...")
    print("-" * 60)
    
    env = build_standard_office_layout()
    state = env.reset()
    stats = env.get_statistics()
    n_nodes = len(env.nodes)
    n_agents = len([a for a in env.agents.values()])
    
    print(f"  ✓ Environment created")
    print(f"    - Nodes: {n_nodes}")
    print(f"    - Agents: {n_agents}")
    print(f"    - People: {stats.get('people_alive', 0)}")
    
    # Create reward shaper (Phase 1 = simplified)
    reward_shaper = RewardShaper.for_scenario(
        scenario="office",
        curriculum_phase=1  # PHASE 1: Only coverage + rescue
    )
    print(f"  ✓ Reward shaper configured (Phase 1):")
    print(f"    - coverage: {reward_shaper.w_coverage}")
    print(f"    - rescue: {reward_shaper.w_rescue}")
    print(f"    - hp_loss: {reward_shaper.w_hp_loss} (OFF)")
    print(f"    - time: {reward_shaper.w_time} (OFF)")
    print(f"    - redundancy: {reward_shaper.w_redundancy} (OFF)")
    
    # ===== STEP 2: Random policy baseline =====
    print("\n[STEP 2] Random policy baseline (3 episodes)...")
    print("-" * 60)
    
    random_rewards = []
    
    for ep in range(3):
        reward_shaper.reset()
        state = env.reset()
        ep_reward = 0.0
        
        for step in range(100):
            # Get available actions per agent
            actions = {}
            for agent_id in range(n_agents):
                valid = env.get_valid_actions(agent_id)
                actions[agent_id] = np.random.choice(valid) if valid else 0
            
            # Step environment
            state, _, done, _ = env.do_action(actions)
            reward = reward_shaper.compute_reward(env)
            ep_reward += reward
            
            if done:
                break
        
        random_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: reward={ep_reward:.3f}")
    
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    print(f"\n  Random baseline: {random_mean:.3f} ± {random_std:.3f}")
    
    # ===== STEP 3: PPO training =====
    print("\n[STEP 3] PPO training (100 iterations)...")
    print("-" * 60)
    
    # Configure PPO
    ppo_config = PPOConfig(
        scenario="office",
        num_agents=n_agents,
        seed=42,
        lr_policy=1e-3,
        lr_value=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        batch_rollout_size=1,
        num_ppo_epochs=1,
        log_interval=100,
    )
    
    # Create trainer
    trainer = EnhancedPPOTrainer(ppo_config)
    trainer.reward_shaper = reward_shaper  # Override with Phase 1 shaper
    
    # Train
    print(f"  Training for 100 iterations...")
    train_rewards = []
    
    for iteration in range(100):
        try:
            rollout_data = trainer.collect_rollout(num_steps=100)
            
            if rollout_data is None:
                continue
            
            # Train on rollout
            _ = trainer.train_on_rollout(rollout_data)
            
            # Track episode reward
            ep_reward = rollout_data.get('episode_return', 0.0)
            train_rewards.append(ep_reward)
            
            if (iteration + 1) % 25 == 0:
                avg_reward = np.mean(train_rewards[-5:]) if len(train_rewards) >= 5 else 0
                print(f"    Iteration {iteration+1}/100: avg_reward={avg_reward:.3f}")
        except Exception as e:
            continue
    
    if len(train_rewards) > 0:
        train_mean = np.mean(train_rewards)
        train_std = np.std(train_rewards)
        print(f"\n  PPO trained: {train_mean:.3f} ± {train_std:.3f}")
    else:
        train_mean = 0.0
        train_std = 0.0
        print(f"\n  PPO training: failed")
    
    # ===== STEP 4: Analysis =====
    print("\n[STEP 4] Diagnostic Analysis")
    print("-" * 60)
    
    improvement = train_mean - random_mean
    improvement_pct = (improvement / abs(random_mean)) * 100 if random_mean != 0 else 0
    
    print(f"\n  Baseline (random):    {random_mean:>8.3f}")
    print(f"  Trained (PPO):        {train_mean:>8.3f}")
    print(f"  Improvement:          {improvement:>8.3f} ({improvement_pct:+.1f}%)")
    
    # Interpretation
    print("\n  Interpretation:")
    if improvement > 5.0:
        print(f"    ✓ EXCELLENT: PPO learned! ({improvement:.1f})")
        status = "PASS"
    elif improvement > 0:
        print(f"    ✓ GOOD: PPO improved. ({improvement:.1f})")
        status = "PASS"
    else:
        print(f"    ✗ FAILED: PPO did not improve. ({improvement:.1f})")
        status = "FAIL"
    
    print("\n" + "="*80)
    print(f"PHASE 1 RESULT: {status}")
    print("="*80 + "\n")
    
    return {
        'status': status,
        'random_mean': random_mean,
        'train_mean': train_mean,
        'improvement': improvement,
        'n_training_episodes': len(train_rewards),
    }


if __name__ == "__main__":
    results = run_curriculum_phase1_diagnostic()
    print("\nFinal Summary:")
    print(f"  Random:  {results['random_mean']:.3f}")
    print(f"  Trained: {results['train_mean']:.3f}")
    print(f"  Improvement: {results['improvement']:.3f}")
    print(f"  Status: {results['status']}")
