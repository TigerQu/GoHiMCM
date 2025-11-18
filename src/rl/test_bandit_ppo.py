#!/usr/bin/env python3
"""
Minimal bandit test for PPO pipeline validation.

This is a 2-action, 1-step environment where:
- Action 0 (CLEAR): +1 reward (optimal)
- Action 1 (WAIT):  0 reward

If PPO learns anything, policy should converge to action 0.
This tests: action masking, log probs, clipping, advantage computation.

Expected behavior:
- Episode return should reach ~1.0
- Policy should select action 0 with >90% probability
- If this fails, core PPO pipeline has a bug
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple
import numpy as np


# Simple MLP Policy
class BanditPolicy(nn.Module):
    def __init__(self, obs_dim=1, num_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, num_actions)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        logits = self.fc2(x)
        return logits


# Simple MLP Value
class BanditValue(nn.Module):
    def __init__(self, obs_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        value = self.fc2(x)
        return value


def test_ppo_pipeline():
    """Run minimal PPO training test."""
    
    print("\n" + "="*80)
    print("BANDIT TEST: Minimal PPO Pipeline Validation")
    print("="*80)
    print("\nEnvironment: 2 actions (CLEAR=+1, WAIT=0), 1-step episodes")
    print("Expected: Policy should learn to select CLEAR >90% of the time\n")
    
    device = 'cpu'
    
    # Initialize policy and value
    policy = BanditPolicy().to(device)
    value = BanditValue().to(device)
    
    policy_opt = Adam(policy.parameters(), lr=3e-4)
    value_opt = Adam(value.parameters(), lr=3e-4)
    
    num_iterations = 100
    clip_epsilon = 0.2
    gamma = 0.99
    entropy_coef = 0.02
    value_loss_coef = 0.25
    
    print(f"Hyperparameters:")
    print(f"  lr_policy={3e-4}, lr_value={3e-4}")
    print(f"  clip_epsilon={clip_epsilon}, entropy_coef={entropy_coef}")
    print(f"  value_loss_coef={value_loss_coef}\n")
    print(f"{'Iter':>4} | {'Reward':>7} | {'Action0_prob':>11} | {'Approx_KL':>10} | "
          f"{'Clip_frac':>9} | {'Explained_var':>12} | {'Policy_Loss':>11} | {'Value_Loss':>10}")
    print("-" * 95)
    
    # Training loop
    for iteration in range(num_iterations):
        # === COLLECT ROLLOUT ===
        batch_size = 64
        obs = torch.ones(batch_size, 1).to(device)  # Dummy observation (constant)
        
        # Forward policy - DETACH to get old policy gradient-free
        logits_old = policy(obs).detach()  # Remove from graph but keep values
        action_probs_old = F.softmax(logits_old, dim=-1)
        log_probs_old = F.log_softmax(logits_old, dim=-1)
        
        # Sample actions (should prefer action 0)
        dist = torch.distributions.Categorical(action_probs_old)
        actions = dist.sample()  # [batch_size]
        sampled_log_probs_old = log_probs_old[torch.arange(batch_size), actions].detach()  # FREEZE
        
        # Value estimates
        with torch.no_grad():
            values_old = value(obs).squeeze(-1)  # [batch_size]
        
        # Compute rewards: action 0 gets +1, action 1 gets 0
        rewards = actions.float()  # [batch_size]: 1 if action 0, 0 if action 1
        dones = torch.ones(batch_size).to(device)  # Always done (1-step)
        
        # Compute returns (TD target)
        with torch.no_grad():
            next_obs = obs  # Same state
            next_values = value(next_obs).squeeze(-1)  # [batch_size]
            returns = rewards + gamma * (1 - dones) * next_values
        
        # === UPDATE POLICY & VALUE ===
        policy_opt.zero_grad()
        value_opt.zero_grad()
        
        # Forward new policy AGAIN (fresh forward pass with gradients)
        logits_new = policy(obs)
        action_probs_new = F.softmax(logits_new, dim=-1)
        log_probs_new = F.log_softmax(logits_new, dim=-1)
        sampled_log_probs_new = log_probs_new[torch.arange(batch_size), actions]
        
        # === COMPUTE DIAGNOSTICS ===
        # 1. Approximate KL
        approx_kl = (sampled_log_probs_old - sampled_log_probs_new).mean().item()
        
        # 2. Clipping fraction
        ratio = torch.exp(sampled_log_probs_new - sampled_log_probs_old)
        clipped = torch.abs(ratio - 1.0) > clip_epsilon
        clip_frac = clipped.float().mean().item()
        
        # DEBUG: Check if policy is actually changing
        if iteration == 0 or iteration % 20 == 0:
            print(f"\n[DEBUG iter {iteration}] Policy state:")
            print(f"  Old logps (first 5): {sampled_log_probs_old[:5].detach().cpu().numpy()}")
            print(f"  New logps (first 5): {sampled_log_probs_new[:5].detach().cpu().numpy()}")
            print(f"  Diff (first 5):     {(sampled_log_probs_old[:5] - sampled_log_probs_new[:5]).detach().cpu().numpy()}")
            print(f"  Ratio (first 5):    {ratio[:5].detach().cpu().numpy()}")
            print(f"  Action0 old logp mean: {log_probs_old[:, 0].mean().item():.6f}")
            print(f"  Action1 old logp mean: {log_probs_old[:, 1].mean().item():.6f}")
            print(f"  Action0 new logp mean: {log_probs_new[:, 0].mean().item():.6f}")
            print(f"  Action1 new logp mean: {log_probs_new[:, 1].mean().item():.6f}")
        
        # 3. Explained variance
        new_values = value(obs).squeeze(-1)
        var_returns = torch.var(returns)
        var_residual = torch.var(returns - new_values)
        explained_var = 1.0 - (var_residual / (var_returns + 1e-8)).item()
        
        # Advantage (TD residual)
        advantages = returns - values_old  # [batch_size]
        
        # Normalize for stability BUT keep the scale for policy loss
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # DEBUG: Print advantage statistics for first iteration
        if iteration == 0 or iteration % 20 == 0:
            adv_debug = advantages.detach().cpu().numpy()
            adv_norm_debug = advantages_norm.detach().cpu().numpy()
            print(f"\n[DEBUG iter {iteration}] Advantage stats:")
            print(f"  Raw: mean={adv_debug.mean():.4f}, std={adv_debug.std():.4f}, "
                  f"min={adv_debug.min():.4f}, max={adv_debug.max():.4f}")
            print(f"  Norm: mean={adv_norm_debug.mean():.4f}, std={adv_norm_debug.std():.4f}, "
                  f"min={adv_norm_debug.min():.4f}, max={adv_norm_debug.max():.4f}")
            print(f"  Rewards mean: {rewards.mean().item():.4f}")
            print(f"  Values mean:  {values_old.mean().item():.4f}")
            print(f"  Ratio mean:   {ratio.mean().item():.4f}")
            print(f"  Action0: {(actions == 0).float().mean().item():.4f}, Action1: {(actions == 1).float().mean().item():.4f}")
        
        # === COMPUTE LOSSES ===
        # Policy loss (PPO) - use UNNORMALIZED advantages for stronger signal
        surr1 = ratio * advantages  # Use raw advantages (positive for action 0, zero for action 1)
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy bonus
        entropy = -(action_probs_new * log_probs_new).sum(dim=1).mean()
        
        # Total loss
        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        
        # DEBUG: Check loss computation
        if iteration == 0 or iteration % 20 == 0:
            print(f"\n[DEBUG iter {iteration}] Loss components:")
            print(f"  policy_loss: {policy_loss.item():.8f} (requires_grad={policy_loss.requires_grad})")
            print(f"  value_loss:  {value_loss.item():.8f}")
            print(f"  entropy:     {entropy.item():.8f}")
            print(f"  total_loss:  {total_loss.item():.8f} (requires_grad={total_loss.requires_grad})")
        
        # Backprop
        total_loss.backward()
        
        # DEBUG: Check gradients
        if iteration == 0 or iteration % 20 == 0:
            print(f"\n[DEBUG iter {iteration}] Gradients and Parameters:")
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.8f}, param_norm={param_norm:.6f}")
                    if name == "fc2.weight":
                        print(f"      fc2.weight[0,0]={param[0,0].item():.8f}")
                else:
                    print(f"  {name}: NO GRADIENT!")
        
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value.parameters(), 0.5)
        
        # DEBUG: Print weights BEFORE step
        if iteration % 10 == 0:  # Every 10 iterations
            fc2_w_before = policy.fc2.weight[0, 0].item()
            print(f"  BEFORE step: fc2.weight[0,0] = {fc2_w_before:.10f}")
        
        policy_opt.step()
        value_opt.step()
        
        # DEBUG: Print weights AFTER step
        if iteration % 10 == 0:
            fc2_w_after = policy.fc2.weight[0, 0].item()
            print(f"  AFTER step:  fc2.weight[0,0] = {fc2_w_after:.10f}")
            print(f"  DELTA:       {fc2_w_after - fc2_w_before:.10f}")
        
        # === GET ACTION 0 PROBABILITY ===
        with torch.no_grad():
            logits_current = policy(obs)
            action_probs_current = F.softmax(logits_current, dim=-1)
            action0_prob = action_probs_current[:, 0].mean().item()
        
        # === PRINT DIAGNOSTICS ===
        mean_reward = rewards.mean().item()
        
        if iteration % 10 == 0:
            print(f"{iteration:4d} | {mean_reward:7.4f} | {action0_prob:11.4f} | "
                  f"{approx_kl:10.6f} | {clip_frac:9.4f} | {explained_var:12.4f} | "
                  f"{policy_loss.item():11.6f} | {value_loss.item():10.6f}")
    
    # === FINAL TEST ===
    print("\n" + "="*95)
    print("FINAL EVALUATION:")
    print("="*95)
    
    with torch.no_grad():
        obs_test = torch.ones(1000, 1).to(device)
        logits_test = policy(obs_test)
        action_probs_test = F.softmax(logits_test, dim=-1)
        
        action0_final_prob = action_probs_test[:, 0].mean().item()
        action1_final_prob = action_probs_test[:, 1].mean().item()
        
        print(f"\nPolicy probabilities:")
        print(f"  Action 0 (CLEAR):  {action0_final_prob:.4f}")
        print(f"  Action 1 (WAIT):   {action1_final_prob:.4f}")
        
        # Sample actions
        actions_test = torch.argmax(action_probs_test, dim=1)
        action0_count = (actions_test == 0).sum().item()
        
        if action0_final_prob > 0.9:
            print(f"\n✓ PASS: Policy correctly learned to select action 0 ({action0_final_prob:.1%})")
        else:
            print(f"\n✗ FAIL: Policy did not converge to action 0 ({action0_final_prob:.1%})")
            print("  This indicates a bug in: action masking, log probs, clipping, or advantage computation")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_ppo_pipeline()
