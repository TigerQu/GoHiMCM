import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import
from rl.enhanced_training import PPOTrainer

# Quick test with the existing training module
print("Testing PPO training with reward shaper...")
trainer = PPOTrainer(scenario="office", num_agents=2)

print("Running 5 training iterations...\n")
for iteration in range(5):
    rollout = trainer.collect_rollout(num_steps=50)
    
    advantages, returns = trainer.compute_advantages(
        rollout['rewards'],
        rollout['dones'],
        rollout['values'],
        rollout['final_value']
    )
    
    losses = trainer.update_policy(
        rollout['observations'],
        rollout['agent_indices'],
        rollout['actions'],
        rollout['log_probs'],
        advantages,
        returns
    )
    
    mean_reward = rollout['rewards'].mean().item()
    print(f"Iteration {iteration}: Reward={mean_reward:.4f}, Policy Loss={losses['policy_loss']:.4f}, Value Loss={losses['value_loss']:.4f}, Entropy={losses['entropy']:.4f}")

print("\n Training test successful!") 