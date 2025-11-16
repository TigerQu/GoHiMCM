"""
Hyperparameter search script.

===== NEW FILE: Systematic hyperparameter tuning =====

Runs grid search over hyperparameters and logs results.
For HiMCM: small, explicit grid is sufficient and more interpretable.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from typing import List
import numpy as np

from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig, create_hparam_grid


def run_single_config(config: PPOConfig, results_file: str):
    """
    Train with single config and log results.
    
    ===== CHANGE 13: Single hyperparameter experiment =====
    
    Args:
        config: PPOConfig to train with
        results_file: CSV file to append results
    """
    print(f"\n{'='*80}")
    print(f"🔬 Testing config: {config.experiment_name}")
    print(f"   lr_policy={config.lr_policy}, entropy_coef={config.entropy_coef}, clip_epsilon={config.clip_epsilon}")
    print(f"{'='*80}\n")
    
    # Shorter training for hyperparameter search
    config.num_iterations = 50  # Quick evaluation
    config.eval_interval = 25
    
    try:
        # Train
        trainer = EnhancedPPOTrainer(config)
        trainer.train()
        
        # Get final evaluation
        final_eval = trainer.evaluate(num_episodes=20)
        
        # Log results
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                config.scenario,
                config.lr_policy,
                config.lr_value,
                config.entropy_coef,
                config.clip_epsilon,
                config.seed,
                final_eval['return_mean'],
                final_eval['return_std'],
                final_eval['people_rescued_mean'],
                final_eval['high_risk_redundancy_mean'],
                final_eval['sweep_complete_rate'],
            ])
        
        print(f"\n✅ Config complete: return={final_eval['return_mean']:.2f}")
        return final_eval
    
    except Exception as e:
        print(f"\n❌ Config failed: {e}")
        return None


def run_hparam_search(scenario: str = "office"):
    """
    Run full hyperparameter grid search.
    
    ===== CHANGE 14: Grid search over hyperparameters =====
    
    For HiMCM paper, this provides evidence that hyperparameters
    were systematically chosen, not arbitrarily guessed.
    
    Args:
        scenario: Which scenario to tune on ("office", "daycare", "warehouse")
    """
    # Create results directory
    os.makedirs("results", exist_ok=True)
    results_file = f"results/hparam_search_{scenario}.csv"
    
    # Initialize CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'scenario',
            'lr_policy',
            'lr_value',
            'entropy_coef',
            'clip_epsilon',
            'seed',
            'eval_return_mean',
            'eval_return_std',
            'eval_people_rescued_mean',
            'eval_high_risk_redundancy_mean',
            'eval_sweep_complete_rate',
        ])
    
    # Create grid
    configs = create_hparam_grid()
    
    # Update scenario for all configs
    for config in configs:
        config.scenario = scenario
    
    print(f"\n🔍 Starting hyperparameter search for {scenario}")
    print(f"   Total configurations: {len(configs)}")
    print(f"   Results will be saved to: {results_file}\n")
    
    # Run all configs
    results = []
    for i, config in enumerate(configs):
        print(f"\n[Config {i+1}/{len(configs)}]")
        result = run_single_config(config, results_file)
        results.append(result)
    
    # Analyze results
    valid_results = [r for r in results if r is not None]
    if valid_results:
        returns = [r['return_mean'] for r in valid_results]
        best_idx = np.argmax(returns)
        best_config = configs[best_idx]
        best_result = valid_results[best_idx]
        
        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION FOUND:")
        print(f"{'='*80}")
        print(f"   lr_policy: {best_config.lr_policy}")
        print(f"   entropy_coef: {best_config.entropy_coef}")
        print(f"   clip_epsilon: {best_config.clip_epsilon}")
        print(f"\n   Performance:")
        print(f"   Return: {best_result['return_mean']:.2f} ± {best_result['return_std']:.2f}")
        print(f"   People Rescued: {best_result['people_rescued_mean']:.1f}")
        print(f"   High-Risk Redundancy: {best_result['high_risk_redundancy_mean']:.2%}")
        print(f"   Sweep Complete: {best_result['sweep_complete_rate']:.2%}")
        print(f"{'='*80}\n")
        
        # Save best config
        best_config.num_iterations = 200  # Reset for full training
        best_config.save(f"results/best_config_{scenario}.json")
        print(f"💾 Best config saved to: results/best_config_{scenario}.json")


def run_multi_seed_experiment(config: PPOConfig, num_seeds: int = 5):
    """
    Run same config with multiple seeds to assess stability.
    
    ===== CHANGE 15: Multi-seed evaluation =====
    Tests whether performance is consistent across seeds.
    
    Args:
        config: Configuration to test
        num_seeds: Number of random seeds to try
    """
    results_file = f"results/multi_seed_{config.experiment_name}.csv"
    
    # Initialize CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed',
            'eval_return_mean',
            'eval_people_rescued_mean',
            'eval_high_risk_redundancy_mean',
        ])
    
    print(f"\n🎲 Running multi-seed experiment: {config.experiment_name}")
    print(f"   Testing {num_seeds} random seeds\n")
    
    all_returns = []
    
    for seed in range(num_seeds):
        config.seed = 1000 + seed
        config.experiment_name = f"{config.scenario}_seed{seed}"
        
        print(f"\n[Seed {seed+1}/{num_seeds}]")
        
        trainer = EnhancedPPOTrainer(config)
        trainer.train()
        
        final_eval = trainer.evaluate(num_episodes=20)
        all_returns.append(final_eval['return_mean'])
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                seed,
                final_eval['return_mean'],
                final_eval['people_rescued_mean'],
                final_eval['high_risk_redundancy_mean'],
            ])
    
    print(f"\n{'='*60}")
    print(f"📊 Multi-Seed Results:")
    print(f"   Mean Return: {np.mean(all_returns):.2f}")
    print(f"   Std Return: {np.std(all_returns):.2f}")
    print(f"   Min Return: {np.min(all_returns):.2f}")
    print(f"   Max Return: {np.max(all_returns):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search for PPO")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['grid', 'multi_seed'],
        default='grid',
        help="Search mode: 'grid' for hyperparameter search, 'multi_seed' for seed stability"
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['office', 'daycare', 'warehouse'],
        default='office',
        help="Scenario to run experiments on"
    )
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=5,
        help="Number of seeds for multi_seed mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'grid':
        run_hparam_search(args.scenario)
    
    elif args.mode == 'multi_seed':
        config = PPOConfig.get_default(args.scenario)
        # Use best hyperparameters (after grid search)
        config.lr_policy = 3e-4
        config.entropy_coef = 0.01
        config.clip_epsilon = 0.2
        run_multi_seed_experiment(config, args.num_seeds)