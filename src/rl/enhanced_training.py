"""
Enhanced PPO training with logging, checkpointing, and evaluation.

===== MAJOR IMPROVEMENTS FROM ORIGINAL =====
1. Structured configuration with PPOConfig
2. Proper logging to CSV and TensorBoard
3. Model checkpointing (save/load)
4. Separate evaluation episodes
5. Train/eval layout splits
6. Best model tracking
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
import random
from torch.cuda.amp import autocast, GradScaler

from environment.layouts import (
    build_standard_office_layout,
    build_babycare_layout,
    build_two_floor_warehouse
)
from rl.new_ppo import Policy, Value
from rl.reward_shaper import RewardShaper
from rl.logging_utils import ExperimentLogger
from rl.ppo_config import PPOConfig


class EnhancedPPOTrainer:
    """
    Complete PPO trainer with all production features.
    
    ===== IMPROVEMENTS OVER ORIGINAL =====
    1. Uses PPOConfig for all hyperparameters
    2. Tracks best model with checkpointing
    3. Separate train/eval layout splits
    4. Proper logging infrastructure
    5. Evaluation mode with deterministic actions
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize trainer with structured config.
        
        ===== CHANGE 5: Config-driven initialization =====
        All hyperparameters now come from PPOConfig.
        """
        self.config = config
        
        # ===== GPU Setup for RTX 5090 =====
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # RTX 5090 optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmul
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("WARNING: CUDA not available, running on CPU")
        
        # Mixed precision training scaler for RTX 5090
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Set random seeds for reproducibility
        self._set_seeds(config.seed)
        
        # ===== Initialize networks =====
        self.policy = Policy(
            num_agents=config.num_agents,
            max_actions=config.max_actions
        ).to(self.device)
        self.value = Value(num_agents=config.num_agents).to(self.device)
        
        # ===== Optimizers =====
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr_policy
        )
        self.value_optimizer = optim.Adam(
            self.value.parameters(),
            lr=config.lr_value
        )
        
        # ===== Reward shaper =====
        self.reward_shaper = RewardShaper.for_scenario(config.scenario)
        
        # ===== Environment =====
        self.env = self._create_env(config.scenario)
        
        # ===== CHANGE 6: Train/eval layout splits =====
        # Generate layout seeds for train/eval separation
        self.train_layout_seeds = list(range(1000, 1000 + config.num_train_layouts))
        self.eval_layout_seeds = list(range(2000, 2000 + config.num_eval_layouts))
        random.shuffle(self.train_layout_seeds)
        
        # ===== CHANGE 7: Logging infrastructure =====
        self.logger = ExperimentLogger(
            log_dir="logs",
            experiment_name=config.experiment_name,
            use_tensorboard=config.use_tensorboard
        )
        
        # ===== CHANGE 8: Best model tracking =====
        self.best_eval_return = float('-inf')
        self.checkpoint_dir = os.path.join(self.logger.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        config.save(os.path.join(self.logger.exp_dir, "config.json"))
        
        print(f"Initialized {config.scenario} trainer")
        print(f"   Train layouts: {config.num_train_layouts}")
        print(f"   Eval layouts: {config.num_eval_layouts}")
    
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    
    def _create_env(self, scenario: str):
        """Create environment for scenario."""
        if scenario == "office":
            return build_standard_office_layout()
        elif scenario == "daycare":
            return build_babycare_layout()
        elif scenario == "warehouse":
            return build_two_floor_warehouse()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    
    def collect_rollout(
        self,
        num_steps: int,
        deterministic: bool = False,
        layout_seed: int = None,
    ) -> Dict:
        """
        Collect one rollout with optional layout randomization.
        
        ===== CHANGE 9: Added layout seed parameter =====
        Enables train/eval split and domain randomization.
        
        Args:
            num_steps: Maximum steps per episode
            deterministic: If True, use greedy actions
            layout_seed: Seed for environment reset (for reproducibility)
        """
        # Reset with specific seed
        obs = self.env.reset(seed=layout_seed)
        self.reward_shaper.reset()
        
        # Storage
        observations = []
        agent_indices_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        valid_actions_list_per_step = []  # Store valid actions for each step
        
        done = False
        step = 0
        
        while not done and step < num_steps:
            # Get agent positions (create directly on GPU)
            agent_indices = torch.tensor([
                self.env.get_agent_node_index(i)
                for i in range(self.config.num_agents)
            ], device=self.device)
            
            # Get valid actions
            valid_actions_list = [
                self.env.get_valid_actions(i)
                for i in range(self.config.num_agents)
            ]
            
            # Move observation to GPU once
            obs_gpu = obs.to(self.device) if hasattr(obs, 'to') else obs
            
            # Select actions and get value in single inference (reduce overhead)
            with torch.no_grad():
                actions, log_probs, action_probs = self.policy.select_actions(
                    obs_gpu, agent_indices, valid_actions_list, deterministic
                )
                values = self.value(obs_gpu, agent_indices)
            
            # Convert to env format
            action_dict = {}
            for i in range(self.config.num_agents):
                action_idx = actions[i].item()
                action_str = self._idx_to_action_str(
                    action_idx,
                    valid_actions_list[i]
                )
                action_dict[i] = action_str
            
            # Execute
            obs_next, _, done, info = self.env.do_action(action_dict)
            reward = self.reward_shaper.compute_reward(self.env)
            
            # Store
            observations.append(obs)
            agent_indices_list.append(agent_indices)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            rewards_list.append(torch.tensor(reward, device=self.device))
            dones_list.append(torch.tensor(float(done), device=self.device))
            values_list.append(values)
            valid_actions_list_per_step.append(valid_actions_list)  # Store for later
            
            obs = obs_next
            step += 1
        
        # Bootstrap value
        agent_indices = torch.tensor([
            self.env.get_agent_node_index(i)
            for i in range(self.config.num_agents)
        ], device=self.device)
        with torch.no_grad():
            obs_gpu = obs.to(self.device) if hasattr(obs, 'to') else obs
            final_value = self.value(obs_gpu, agent_indices)
        
        # Get episode summary
        episode_stats = self.reward_shaper.get_episode_summary(self.env)
        episode_return = torch.stack(rewards_list).sum().item()
        
        return {
            'observations': observations,
            'agent_indices': agent_indices_list,
            'actions': torch.stack(actions_list),
            'log_probs': torch.stack(log_probs_list),
            'rewards': torch.stack(rewards_list),
            'dones': torch.stack(dones_list),
            'values': torch.cat(values_list),
            'final_value': final_value,
            'episode_stats': episode_stats,
            'episode_return': episode_return,
            'valid_actions_per_step': valid_actions_list_per_step,  # NEW
        }
    
    
    def collect_batch_rollouts(
        self,
        num_rollouts: int,
        num_steps: int,
        deterministic: bool = False,
        layout_seeds: List[int] = None,
    ) -> Dict:
        """
        Collect multiple rollouts and aggregate them into batched data.
        
        ===== NEW METHOD: Batch rollout collection =====
        Collects K complete episodes sequentially and stacks all transitions
        for efficient batched policy updates.
        
        Args:
            num_rollouts: Number of complete episodes to collect
            num_steps: Max steps per episode
            deterministic: If True, use greedy actions
            layout_seeds: List of seeds for environment randomization
                         If None, samples randomly
            
        Returns:
            batch: Aggregated data from all rollouts
                Contains:
                - observations: List of observations across all episodes
                - agent_indices: List of agent index tensors
                - actions: All actions stacked [total_steps, num_agents]
                - log_probs: Log probabilities [total_steps, num_agents]
                - rewards: Rewards [total_steps]
                - dones: Done flags [total_steps]
                - values: Value estimates [total_steps, num_agents]
                - advantages: GAE advantages [total_steps]
                - returns: Returns [total_steps]
                - episode_returns: List of return per episode
                - episode_stats: List of stats per episode
        """
        # Generate or use provided seeds
        if layout_seeds is None:
            layout_seeds = [random.choice(self.train_layout_seeds) for _ in range(num_rollouts)]
        
        # Collect all rollouts
        all_observations = []
        all_agent_indices = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_final_values = []
        all_episode_returns = []
        all_episode_stats = []
        episode_step_boundaries = [0]  # For tracking episode boundaries
        
        for ep in range(num_rollouts):
            rollout = self.collect_rollout(
                num_steps=num_steps,
                deterministic=deterministic,
                layout_seed=layout_seeds[ep]
            )
            
            # Stack episode data
            all_observations.extend(rollout['observations'])
            all_agent_indices.extend(rollout['agent_indices'])
            all_actions.append(rollout['actions'])
            all_log_probs.append(rollout['log_probs'])
            all_rewards.append(rollout['rewards'])
            all_dones.append(rollout['dones'])
            all_values.append(rollout['values'])
            all_final_values.append(rollout['final_value'])
            all_episode_returns.append(rollout['episode_return'])
            all_episode_stats.append(rollout['episode_stats'])
            
            # Track episode boundaries for advantage computation
            episode_step_boundaries.append(episode_step_boundaries[-1] + len(rollout['observations']))
        
        # Stack all tensors
        actions_all = torch.cat(all_actions, dim=0)  # [total_steps, num_agents]
        log_probs_all = torch.cat(all_log_probs, dim=0)  # [total_steps, num_agents]
        rewards_all = torch.cat(all_rewards, dim=0)  # [total_steps]
        dones_all = torch.cat(all_dones, dim=0)  # [total_steps]
        values_all = torch.cat(all_values, dim=0)  # [total_steps, num_agents] or [total_steps*num_agents]
        
        # Compute advantages across all episodes
        # Need to handle each episode separately for GAE
        all_advantages = []
        all_returns = []
        
        for ep_idx in range(num_rollouts):
            start_idx = episode_step_boundaries[ep_idx]
            end_idx = episode_step_boundaries[ep_idx + 1]
            
            ep_rewards = rewards_all[start_idx:end_idx]
            ep_dones = dones_all[start_idx:end_idx]
            
            # Get values for this episode
            T_ep = end_idx - start_idx
            if values_all.dim() == 2:  # [total_steps, num_agents]
                ep_values = values_all[start_idx:end_idx].mean(dim=1)  # Average across agents
            else:  # [total_steps * num_agents]
                ep_values = values_all[start_idx*self.config.num_agents:(end_idx)*self.config.num_agents]
                ep_values = ep_values.view(T_ep, self.config.num_agents).mean(dim=1)
            
            final_value = all_final_values[ep_idx].mean()
            
            # Compute GAE for this episode
            values_with_bootstrap = torch.cat([ep_values, final_value.unsqueeze(0)])
            ep_advantages = Policy.gae(
                ep_rewards, ep_dones, values_with_bootstrap,
                gamma=self.config.gamma,
                lambda_=self.config.gae_lambda
            )
            ep_returns = ep_advantages + ep_values
            
            all_advantages.append(ep_advantages)
            all_returns.append(ep_returns)
        
        advantages_all = torch.cat(all_advantages, dim=0)  # [total_steps]
        returns_all = torch.cat(all_returns, dim=0)  # [total_steps]
        
        # Expand to match agent counts
        advantages_expanded = advantages_all.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)
        returns_expanded = returns_all.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)
        
        return {
            'observations': all_observations,
            'agent_indices': all_agent_indices,
            'actions': actions_all,
            'log_probs': log_probs_all,
            'rewards': rewards_all,
            'dones': dones_all,
            'values': values_all,
            'advantages': advantages_expanded,
            'returns': returns_expanded,
            'episode_returns': all_episode_returns,
            'episode_stats': all_episode_stats,
            'num_episodes': num_rollouts,
            'num_transitions': actions_all.size(0),
        }
    
    
    def _idx_to_action_str(self, action_idx: int, valid_actions: List[str]) -> str:
        """Map action index to string."""
        if action_idx == 0:
            return "wait"
        elif action_idx == 1:
            return "search"
        else:
            move_actions = [a for a in valid_actions if a.startswith("move_")]
            if move_actions:
                return move_actions[min(action_idx - 2, len(move_actions) - 1)]
            return "wait"
    
    
    def print_iteration_metrics(
        self,
        iteration: int,
        rollout: Dict,
        batch: Dict = None,
        losses: Dict = None,
        advantages: torch.Tensor = None,
    ):
        """
        ===== NEW METHOD: Comprehensive iteration diagnostics =====
        Prints detailed metrics including:
        - Agent actions and positions
        - Value loss and policy loss
        - Agent state information
        - Episode statistics
        - Advantage distribution
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} - DETAILED DIAGNOSTICS")
        print(f"{'='*80}")
        
        # ===== AGENT ACTIONS =====
        print(f"\n{'-'*80}")
        print(f"AGENT ACTIONS")
        print(f"{'-'*80}")
        
        actions = rollout['actions']  # [T, num_agents]
        for agent_id in range(self.config.num_agents):
            action_indices = actions[:, agent_id].cpu().numpy()
            
            # Get valid actions for first step (proxy for available actions)
            valid_actions_list = [self.env.get_valid_actions(agent_id) for _ in range(len(action_indices))]
            
            # Convert indices to action strings
            action_strings = []
            for t, action_idx in enumerate(action_indices[:min(10, len(action_indices))]):
                try:
                    action_str = self._idx_to_action_str(int(action_idx), valid_actions_list[0])
                    action_strings.append(action_str)
                except:
                    action_strings.append(f"idx_{int(action_idx)}")
            
            action_counts = {}
            for action_str in action_strings:
                action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            print(f"\n  Agent {agent_id}:")
            print(f"    Total steps: {len(action_indices)}")
            print(f"    First 10 actions: {' → '.join(action_strings)}")
            print(f"    Action distribution: {action_counts}")
            print(f"    Unique actions: {len(set(action_strings))}")
        
        # ===== AGENT STATE =====
        print(f"\n{'-'*80}")
        print(f"AGENT STATE")
        print(f"{'-'*80}")
        
        for agent_id in range(self.config.num_agents):
            node_idx = self.env.get_agent_node_index(agent_id)
            node_name = list(self.env.node_to_idx.keys())[node_idx] if hasattr(self.env, 'node_to_idx') else f"Node_{node_idx}"
            
            # Get agent position history
            print(f"\n  Agent {agent_id}:")
            print(f"    Current position: {node_name} (node_idx={node_idx})")
            print(f"    Valid actions: {self.env.get_valid_actions(agent_id)}")
        
        # ===== LOSS METRICS =====
        print(f"\n{'-'*80}")
        print(f"LOSS METRICS")
        print(f"{'-'*80}")
        
        if losses:
            print(f"\n  Policy Loss:     {losses['policy_loss']:>10.6f}")
            print(f"  Value Loss:      {losses['value_loss']:>10.6f}")
            print(f"  Entropy Bonus:   {losses['entropy']:>10.6f}")
            
            # Combined loss
            combined = (losses['policy_loss'] + 
                       self.config.value_loss_coef * losses['value_loss'] - 
                       self.config.entropy_coef * losses['entropy'])
            print(f"  Combined Loss:   {combined:>10.6f}")
            
            # ===== 6 HARD DIAGNOSTIC METRICS =====
            print(f"\n{'-'*80}")
            print(f"6 PPO DIAGNOSTIC METRICS (HARD INDICATORS)")
            print(f"{'-'*80}")
            
            # 1. Approximate KL divergence
            approx_kl = losses.get('approx_kl', 0.0)
            kl_status = "✓ OK" if 0.001 <= approx_kl <= 0.02 else ("✗ TOO HIGH" if approx_kl > 0.02 else "✗ TOO LOW")
            print(f"\n  1. Approx KL divergence: {approx_kl:>10.6f}  {kl_status}")
            print(f"     (Normal range: 0.001-0.02; >0.02=step too large; ≈0 with reward drop=adv/logp bug)")
            
            # 2. Clipping fraction
            clip_frac = losses.get('clip_fraction', 0.0)
            clip_status = "✓ OK" if 0.1 <= clip_frac <= 0.4 else ("✗ TOO HIGH" if clip_frac > 0.4 else "✗ TOO LOW")
            print(f"\n  2. Clip fraction:       {clip_frac:>10.4f}  {clip_status}")
            print(f"     (Normal range: 0.1-0.4; 0=can't learn; 1=completely clipped)")
            
            # 3. Entropy
            entropy = losses.get('entropy', 0.0)
            entropy_status = "✓ OK" if entropy > 0.1 else "✗ LOW - check entropy_coef"
            print(f"\n  3. Entropy:             {entropy:>10.6f}  {entropy_status}")
            print(f"     (Should decrease smoothly; high entropy + reward drop = entropy_coef too large)")
            
            # 4. Explained variance
            explained_var = losses.get('explained_var', 0.0)
            var_status = "✓ OK" if explained_var > 0.1 else ("✗ CRASH" if explained_var < 0 else "✗ WEAK")
            print(f"\n  4. Explained variance:  {explained_var:>10.4f}  {var_status}")
            print(f"     (<0 or ~0 = critic collapsed; fix: lower lr_value, add value_clip, reduce value_loss_coef)")
            
            # 5. Advantage distribution
            adv_mean = losses.get('adv_mean', 0.0)
            adv_std = losses.get('adv_std', 1.0)
            adv_status = "✓ OK" if abs(adv_mean) < 0.1 and 0.9 <= adv_std <= 1.1 else "⚠ NOT NORMALIZED"
            print(f"\n  5. Advantage stats:     mean={adv_mean:>8.4f}, std={adv_std:>8.4f}  {adv_status}")
            print(f"     (Should be: mean≈0, std≈1 after normalization)")
            
            # 6. Old vs new logp
            old_logp_mean = losses.get('old_logp_mean', 0.0)
            new_logp_mean = losses.get('new_logp_mean', 0.0)
            logp_status = "✓ OK" if abs(old_logp_mean - new_logp_mean) < 1.0 else "✗ DIVERGED"
            print(f"\n  6. LogP sanity check:   old={old_logp_mean:>8.4f}, new={new_logp_mean:>8.4f}  {logp_status}")
            print(f"     (Shouldn't diverge; if old≈new+KL≠0, check ratio calculation)")
        
        # ===== REWARD METRICS =====
        print(f"\n{'-'*80}")
        print(f"REWARD METRICS")
        print(f"{'-'*80}")
        
        rewards = rollout['rewards'].cpu().numpy()
        print(f"\n  Total return:    {rollout['episode_return']:>10.2f}")
        print(f"  Mean reward:     {np.mean(rewards):>10.4f}")
        print(f"  Std reward:      {np.std(rewards):>10.4f}")
        print(f"  Min reward:      {np.min(rewards):>10.4f}")
        print(f"  Max reward:      {np.max(rewards):>10.4f}")
        print(f"  Episode length:  {len(rewards):>10d} steps")
        
        # ===== VALUE FUNCTION METRICS =====
        print(f"\n{'-'*80}")
        print(f"VALUE FUNCTION")
        print(f"{'-'*80}")
        
        values = rollout['values']
        if values.dim() > 1:
            values_mean = values.mean(dim=1).cpu().numpy()
        else:
            values_mean = values.cpu().numpy()
        
        print(f"\n  Value mean:      {np.mean(values_mean):>10.4f}")
        print(f"  Value std:       {np.std(values_mean):>10.4f}")
        print(f"  Value min:       {np.min(values_mean):>10.4f}")
        print(f"  Value max:       {np.max(values_mean):>10.4f}")
        
        # ===== ADVANTAGE METRICS =====
        print(f"\n{'-'*80}")
        print(f"ADVANTAGE ESTIMATION")
        print(f"{'-'*80}")
        
        if advantages is not None:
            adv_np = advantages.cpu().numpy()
            print(f"\n  Advantage mean:  {np.mean(adv_np):>10.4f}")
            print(f"  Advantage std:   {np.std(adv_np):>10.4f}")
            print(f"  Advantage min:   {np.min(adv_np):>10.4f}")
            print(f"  Advantage max:   {np.max(adv_np):>10.4f}")
            
            # Normalize check
            if np.std(adv_np) > 0:
                normalized_adv = (adv_np - np.mean(adv_np)) / (np.std(adv_np) + 1e-8)
                print(f"  Normalized mean: {np.mean(normalized_adv):>10.4f} (should be ~0)")
                print(f"  Normalized std:  {np.std(normalized_adv):>10.4f} (should be ~1)")
        
        # ===== EPISODE STATISTICS =====
        print(f"\n{'-'*80}")
        print(f"EPISODE STATISTICS")
        print(f"{'-'*80}")
        
        stats = rollout['episode_stats']
        print(f"\n  People rescued:           {stats['people_rescued']:>6d}")
        print(f"  People found:             {stats['people_found']:>6d}")
        print(f"  People alive:             {stats['people_alive']:>6d}")
        print(f"  Nodes swept:              {stats['nodes_swept']:>6d}")
        print(f"  High-risk redundancy:     {stats['high_risk_redundancy']:>6.2f}")
        print(f"  Sweep complete:           {stats['sweep_complete']:>6} (1=yes, 0=no)")
        print(f"  Time steps:               {stats['time_step']:>6d}")
        
        # ===== BATCH STATISTICS (if applicable) =====
        if batch:
            print(f"\n{'-'*80}")
            print(f"BATCH STATISTICS (K={batch['num_episodes']} episodes)")
            print(f"{'-'*80}")
            
            print(f"\n  Total transitions:  {batch['num_transitions']:>6d}")
            print(f"  Episode returns:    {[f'{r:.2f}' for r in batch['episode_returns']]}")
            print(f"  Mean return:        {np.mean(batch['episode_returns']):>10.2f}")
            print(f"  Std return:         {np.std(batch['episode_returns']):>10.2f}")
            
            # Batch episode stats
            batch_stats = batch['episode_stats']
            print(f"  Mean people rescued: {np.mean([s['people_rescued'] for s in batch_stats]):>9.1f}")
            print(f"  Mean nodes swept:    {np.mean([s['nodes_swept'] for s in batch_stats]):>9.1f}")
        
        print(f"\n{'='*80}\n")

    
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        final_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages.
        
        CRITICAL FIX: Handle reward/value shape mismatch
        - rewards: [T] (shared across all agents)
        - values: [T, num_agents] (per-agent value estimates)
        
        Solution: Replicate rewards across agents for GAE computation
        """
        T = rewards.size(0)
        values_reshaped = values.view(T, self.config.num_agents)
        
        # Replicate shared reward across all agents (since reward is shared)
        rewards_per_agent = rewards.unsqueeze(1).expand(-1, self.config.num_agents)  # [T, num_agents]
        
        # Take average value for bootstrapping (since reward is shared)
        final_value_avg = final_value.mean()
        values_avg = values_reshaped.mean(dim=1)  # [T]
        
        values_with_bootstrap = torch.cat([values_avg, final_value_avg.unsqueeze(0)])
        
        # Compute GAE on averaged values with shared rewards
        advantages = Policy.gae(
            rewards, dones, values_with_bootstrap,
            gamma=self.config.gamma,
            lambda_=self.config.gae_lambda
        )  # [T]
        
        returns = advantages + values_avg  # [T]
        
        # Replicate advantages and returns across agents to match action/log_prob dimensions
        advantages = advantages.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)  # [T*num_agents]
        returns = returns.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)  # [T*num_agents]
        
        return advantages, returns
    
    
    def update_policy(
        self,
        observations: List,
        agent_indices_list: List[torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        valid_actions_per_step: List[List[List[str]]] = None,  # NEW: Valid actions for masking
    ) -> Dict[str, float]:
        """Update policy and value networks with diagnostic metrics."""
        T = actions.size(0)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        total_explained_var = 0.0
        
        # Normalize advantages to stabilize training
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten returns to match dimensions
        returns_flat = returns.reshape(-1)
        old_log_probs_flat = old_log_probs.reshape(-1)
        
        first_epoch_metrics = {}
        
        for epoch in range(self.config.num_ppo_epochs):
            all_log_probs = []
            all_action_probs = []
            all_values = []
            
            for t in range(T):
                obs = observations[t]
                agent_indices = agent_indices_list[t]
                
                # Move observation to GPU
                obs_gpu = obs.to(self.device) if hasattr(obs, 'to') else obs
                
                # Use automatic mixed precision for forward pass
                with autocast(enabled=self.use_amp):
                    action_logits, _ = self.policy(obs_gpu, agent_indices)
                    
                    # CRITICAL FIX: Apply same masking as during action selection!
                    if valid_actions_per_step is not None and t < len(valid_actions_per_step):
                        valid_actions_list = valid_actions_per_step[t]
                        
                        # Create action masks
                        action_masks = torch.zeros_like(action_logits, dtype=torch.bool, device=self.device)
                        for i, valid_actions in enumerate(valid_actions_list):
                            for action_str in valid_actions:
                                action_idx = self.policy._action_str_to_idx(action_str)
                                if action_idx < self.policy.max_actions:
                                    action_masks[i, action_idx] = True
                        
                        # Apply masking BEFORE softmax (same as select_actions)
                        masked_logits = action_logits.clone()
                        masked_logits[~action_masks] = float('-inf')
                        action_probs = F.softmax(masked_logits, dim=-1)
                    else:
                        # Fallback if no valid actions provided (shouldn't happen)
                        action_probs = F.softmax(action_logits, dim=-1)
                    
                    log_probs = torch.log(
                        action_probs[torch.arange(self.config.num_agents, device=self.device), actions[t]] + 1e-8
                    )
                    
                    all_log_probs.append(log_probs)
                    all_action_probs.append(action_probs)
                    
                    value = self.value(obs_gpu, agent_indices)
                    all_values.append(value)
            
            new_log_probs = torch.cat(all_log_probs).reshape(-1)
            new_action_probs = torch.cat(all_action_probs)
            new_values = torch.cat(all_values).reshape(-1)
            
            # COMPUTE FIRST EPOCH DIAGNOSTICS ONLY (before weights change)
            if epoch == 0:
                # 1. Approximate KL divergence
                first_epoch_metrics['approx_kl'] = (old_log_probs_flat - new_log_probs).mean().item()
                
                # 2. Clipping fraction
                ratio = torch.exp(new_log_probs - old_log_probs_flat)
                clipped = torch.abs(ratio - 1.0) > self.config.clip_epsilon
                first_epoch_metrics['clip_fraction'] = clipped.float().mean().item()
                
                # 3. Entropy (from action probs)
                first_epoch_metrics['entropy_val'] = Policy.entropy_bonus(new_action_probs).item()
                
                # 4. Explained variance (critic performance)
                var_returns = torch.var(returns_flat).item()
                var_residual = torch.var(returns_flat - new_values).item()
                first_epoch_metrics['explained_var'] = 1.0 - (var_residual / (var_returns + 1e-8))
                
                # 5. Advantage statistics
                first_epoch_metrics['adv_mean'] = advantages_normalized.mean().item()
                first_epoch_metrics['adv_std'] = advantages_normalized.std().item()
                
                # 6. Old vs new logp statistics (sanity check)
                first_epoch_metrics['old_logp_mean'] = old_log_probs_flat.mean().item()
                first_epoch_metrics['new_logp_mean'] = new_log_probs.mean().item()
            
            # Compute losses with mixed precision
            with autocast(enabled=self.use_amp):
                policy_loss = Policy.policy_loss(
                    advantages_normalized, old_log_probs_flat, new_log_probs,
                    clip_epsilon=self.config.clip_epsilon
                )
                
                # VALUE CLIPPING: Prevent critic from changing too fast
                # Clip new values to be close to old value predictions
                value_clip_margin = 0.2
                old_values_for_clip = torch.cat([
                    self.value(obs.to(self.device) if hasattr(obs, 'to') else obs, 
                               agent_indices_list[t]).detach()
                    for t, obs in enumerate(observations)
                ]).reshape(-1)
                
                values_clipped = old_values_for_clip + torch.clamp(
                    new_values - old_values_for_clip, 
                    -value_clip_margin, 
                    value_clip_margin
                )
                
                # Use clipped values for MSE loss
                value_loss = Value.value_loss(values_clipped, returns_flat)
                entropy = Policy.entropy_bonus(new_action_probs)
                
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss -
                    self.config.entropy_coef * entropy
                )
            
            # Backward pass with gradient scaling
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.policy_optimizer)
                self.scaler.unscale_(self.value_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.policy_optimizer)
                self.scaler.step(self.value_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value.parameters(),
                    self.config.max_grad_norm
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Compute averages
        num_epochs = self.config.num_ppo_epochs
        avg_policy_loss = total_policy_loss / num_epochs
        avg_value_loss = total_value_loss / num_epochs
        avg_entropy = total_entropy / num_epochs
        
        # Use first epoch metrics for diagnostics
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'approx_kl': first_epoch_metrics.get('approx_kl', 0.0),
            'clip_fraction': first_epoch_metrics.get('clip_fraction', 0.0),
            'explained_var': first_epoch_metrics.get('explained_var', 0.0),
            'adv_mean': first_epoch_metrics.get('adv_mean', 0.0),
            'adv_std': first_epoch_metrics.get('adv_std', 1.0),
            'old_logp_mean': first_epoch_metrics.get('old_logp_mean', 0.0),
            'new_logp_mean': first_epoch_metrics.get('new_logp_mean', 0.0),
        }
    
    
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """
        Run evaluation episodes on held-out layouts.
        
        ===== CHANGE 10: Separate evaluation mode =====
        Uses deterministic actions and eval layout seeds.
        
        Args:
            num_episodes: Number of eval episodes (default from config)
            
        Returns:
            summary: Aggregated evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.config.num_eval_episodes
        
        returns = []
        stats_list = []
        
        print(f"Running evaluation ({num_episodes} episodes)...)\n")
        
        for ep in range(num_episodes):
            try:
                # Use eval layout seeds
                layout_seed = self.eval_layout_seeds[ep % len(self.eval_layout_seeds)]
                
                rollout = self.collect_rollout(
                    num_steps=self.config.steps_per_rollout,
                    deterministic=True,  # Greedy actions for evaluation
                    layout_seed=layout_seed
                )
                
                returns.append(rollout['episode_return'])
                stats_list.append(rollout['episode_stats'])
            
            except Exception as e:
                print(f"⚠️  Warning: Evaluation episode {ep} failed: {e}")
                print("   Continuing with remaining episodes...")
                continue
        
        # Aggregate statistics
        summary = {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'people_rescued_mean': np.mean([s['people_rescued'] for s in stats_list]),
            'people_found_mean': np.mean([s['people_found'] for s in stats_list]),
            'people_alive_mean': np.mean([s['people_alive'] for s in stats_list]),
            'nodes_swept_mean': np.mean([s['nodes_swept'] for s in stats_list]),
            'high_risk_redundancy_mean': np.mean([s['high_risk_redundancy'] for s in stats_list]),
            'sweep_complete_rate': np.mean([s['sweep_complete'] for s in stats_list]),
            'episode_length_mean': np.mean([s['time_step'] for s in stats_list]),
        }
        
        print(f"Eval complete: return={summary['return_mean']:.2f}±{summary['return_std']:.2f}")
        
        return summary
    
    
    def save_checkpoint(self, iteration: int, is_best: bool = False, extra: Dict = None):
        """
        Save model checkpoint.
        
        ===== CHANGE 11: Model checkpointing =====
        Saves policy, value, optimizer states, and metadata.
        """
        checkpoint = {
            'iteration': iteration,
            'policy_state': self.policy.state_dict(),
            'value_state': self.value.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'config': self.config.to_dict(),
            'best_eval_return': self.best_eval_return,
            'scaler_state': self.scaler.state_dict() if self.scaler else None,
            'extra': extra or {},
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_iter{iteration}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
    
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.value.load_state_dict(checkpoint['value_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        self.best_eval_return = checkpoint['best_eval_return']
        if self.scaler and checkpoint.get('scaler_state'):
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        print(f"Loaded checkpoint from {path}")
        return checkpoint
    
    
    def train(self):
        """
        Main training loop with logging, eval, and checkpointing.
        
        ===== CHANGE 12: Complete training pipeline =====
        Integrates all improvements: logging, eval, checkpointing.
        
        ===== CHANGE 13: Batch rollout collection =====
        Can now collect multiple rollouts before policy updates using batch_rollout_size.
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Total iterations: {self.config.num_iterations}")
        print(f"Batch rollout size: {self.config.batch_rollout_size}")
        print(f"{'='*60}\n")
        
        for iteration in range(self.config.num_iterations):
            try:
                # ===== CHANGE: Collect batch of rollouts =====
                if self.config.batch_rollout_size > 1:
                    # Collect multiple rollouts and aggregate
                    batch = self.collect_batch_rollouts(
                        num_rollouts=self.config.batch_rollout_size,
                        num_steps=self.config.steps_per_rollout
                    )
                    
                    # Update policy using batched data
                    losses = self.update_policy(
                        batch['observations'],
                        batch['agent_indices'],
                        batch['actions'],
                        batch['log_probs'],
                        batch['advantages'],
                        batch['returns'],
                        batch.get('valid_actions_per_step', None)  # Pass valid actions
                    )
                    
                    # Log average metrics across batch
                    avg_return = np.mean(batch['episode_returns'])
                    avg_stats = {}
                    for key in batch['episode_stats'][0].keys():
                        avg_stats[key] = np.mean([s[key] for s in batch['episode_stats']])
                    
                    if iteration % self.config.log_interval == 0:
                        self.logger.log_iteration(
                            iteration,
                            losses,
                            avg_stats,
                            avg_return
                        )
                        
                        print(f"Iter {iteration:4d} | "
                              f"Batch Return: {avg_return:7.2f} (n={batch['num_episodes']}) | "
                              f"Policy Loss: {losses['policy_loss']:7.4f} | "
                              f"Rescued: {avg_stats['people_rescued']:2.1f} | "
                              f"Redundancy: {avg_stats['high_risk_redundancy']:.2f}")
                        
                        # ===== DETAILED DIAGNOSTICS FOR FIRST ROLLOUT IN BATCH =====
                        first_rollout = {
                            'actions': batch['actions'][:batch['actions'].size(0)//batch['num_episodes']],
                            'rewards': batch['rewards'][:batch['rewards'].size(0)//batch['num_episodes']],
                            'values': batch['values'][:batch['values'].size(0)//batch['num_episodes']],
                            'episode_return': batch['episode_returns'][0],
                            'episode_stats': batch['episode_stats'][0],
                        }
                        self.print_iteration_metrics(
                            iteration,
                            first_rollout,
                            batch=batch,
                            losses=losses,
                            advantages=batch['advantages'][:batch['advantages'].size(0)//batch['num_episodes']]
                        )
                else:
                    # Original single rollout per iteration
                    layout_seed = random.choice(self.train_layout_seeds)
                    
                    rollout = self.collect_rollout(
                        num_steps=self.config.steps_per_rollout,
                        layout_seed=layout_seed
                    )
                    
                    # Compute advantages
                    advantages, returns = self.compute_advantages(
                        rollout['rewards'],
                        rollout['dones'],
                        rollout['values'],
                        rollout['final_value']
                    )
                    
                    # Update policy
                    losses = self.update_policy(
                        rollout['observations'],
                        rollout['agent_indices'],
                        rollout['actions'],
                        rollout['log_probs'],
                        advantages,
                        returns,
                        rollout.get('valid_actions_per_step', None)  # Pass valid actions
                    )
                    
                    # Log training metrics
                    if iteration % self.config.log_interval == 0:
                        self.logger.log_iteration(
                            iteration,
                            losses,
                            rollout['episode_stats'],
                            rollout['episode_return']
                        )
                        
                        print(f"Iter {iteration:4d} | "
                              f"Return: {rollout['episode_return']:7.2f} | "
                              f"Policy Loss: {losses['policy_loss']:7.4f} | "
                              f"Rescued: {rollout['episode_stats']['people_rescued']:2d} | "
                              f"Redundancy: {rollout['episode_stats']['high_risk_redundancy']:.2f}")
                        
                        # ===== DETAILED DIAGNOSTICS =====
                        self.print_iteration_metrics(
                            iteration,
                            rollout,
                            batch=None,
                            losses=losses,
                            advantages=advantages
                        )
                
                # Evaluate and checkpoint
                if (iteration + 1) % self.config.eval_interval == 0:
                    print(f"\n{'='*60}")
                    print(f"Evaluation at iteration {iteration + 1}/{self.config.num_iterations}")
                    print(f"Progress: {(iteration + 1) / self.config.num_iterations * 100:.1f}%")
                    print(f"{'='*60}")
                    eval_summary = self.evaluate()
                    self.logger.log_eval(iteration, eval_summary)
                    
                    # Check if best model
                    is_best = eval_summary['return_mean'] > self.best_eval_return
                    if is_best:
                        self.best_eval_return = eval_summary['return_mean']
                    
                    self.save_checkpoint(
                        iteration,
                        is_best=is_best,
                        extra=eval_summary
                    )
                
                # Regular checkpoint
                if (iteration + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(iteration)
            
            except KeyboardInterrupt:
                print("\n\n⚠️ Training interrupted by user at iteration", iteration)
                print("Saving checkpoint before exit...")
                self.save_checkpoint(iteration, extra={'interrupted': True})
                print("Checkpoint saved. Exiting.")
                return
            
            except Exception as e:
                print(f"\n\n Error at iteration {iteration}: {type(e).__name__}: {e}")
                print("Saving emergency checkpoint...")
                try:
                    self.save_checkpoint(iteration, extra={'error': str(e)})
                    print("Emergency checkpoint saved.")
                except:
                    print("Failed to save emergency checkpoint.")
                raise  # Re-raise the exception for debugging
        
        # Final evaluation
        print("\n" + "="*60)
        print("Training complete! Running final evaluation...")
        print("="*60 + "\n")
        
        final_eval = self.evaluate(num_episodes=50)
        self.logger.log_eval(self.config.num_iterations, final_eval)
        
        print(f"\nFinal Results:")
        print(f"   Return: {final_eval['return_mean']:.2f} ± {final_eval['return_std']:.2f}")
        print(f"   People Rescued: {final_eval['people_rescued_mean']:.1f}")
        print(f"   High-Risk Redundancy: {final_eval['high_risk_redundancy_mean']:.2%}")
        print(f"   Sweep Complete Rate: {final_eval['sweep_complete_rate']:.2%}")
        
        self.logger.close()


if __name__ == "__main__":
    # Example: Train on daycare (babycare) with comprehensive diagnostics
    config = PPOConfig.get_default("daycare")
    config.log_interval = 10  # Print comprehensive diagnostics every 10 iterations
    trainer = EnhancedPPOTrainer(config)
    trainer.train()