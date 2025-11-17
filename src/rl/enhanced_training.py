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
    
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        final_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages."""
        T = rewards.size(0)
        values_reshaped = values.view(T, self.config.num_agents)
        values_avg = values_reshaped.mean(dim=1)
        final_value_avg = final_value.mean()
        
        values_with_bootstrap = torch.cat([values_avg, final_value_avg.unsqueeze(0)])
        
        advantages = Policy.gae(
            rewards, dones, values_with_bootstrap,
            gamma=self.config.gamma,
            lambda_=self.config.gae_lambda
        )
        
        returns = advantages + values_avg
        
        advantages = advantages.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)
        returns = returns.unsqueeze(1).expand(-1, self.config.num_agents).reshape(-1)
        
        return advantages, returns
    
    
    def update_policy(
        self,
        observations: List,
        agent_indices_list: List[torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy and value networks."""
        T = actions.size(0)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
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
                    action_probs = F.softmax(action_logits, dim=-1)
                    log_probs = torch.log(
                        action_probs[torch.arange(self.config.num_agents, device=self.device), actions[t]] + 1e-8
                    )
                    
                    all_log_probs.append(log_probs)
                    all_action_probs.append(action_probs)
                    
                    value = self.value(obs_gpu, agent_indices)
                    all_values.append(value)
            
            new_log_probs = torch.cat(all_log_probs)
            new_action_probs = torch.cat(all_action_probs)
            new_values = torch.cat(all_values)
            old_log_probs_flat = old_log_probs.reshape(-1)
            
            # Compute losses with mixed precision
            with autocast(enabled=self.use_amp):
                policy_loss = Policy.policy_loss(
                    advantages, old_log_probs_flat, new_log_probs,
                    clip_epsilon=self.config.clip_epsilon
                )
                
                value_loss = Value.value_loss(new_values, returns)
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
        
        return {
            'policy_loss': total_policy_loss / self.config.num_ppo_epochs,
            'value_loss': total_value_loss / self.config.num_ppo_epochs,
            'entropy': total_entropy / self.config.num_ppo_epochs,
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
        checkpoint = torch.load(path, map_location=self.device)
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
                        batch['returns']
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
                        returns
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
                print("\n\n⚠️  Training interrupted by user at iteration", iteration)
                print("Saving checkpoint before exit...")
                self.save_checkpoint(iteration, extra={'interrupted': True})
                print("Checkpoint saved. Exiting.")
                return
            
            except Exception as e:
                print(f"\n\n❌ Error at iteration {iteration}: {type(e).__name__}: {e}")
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
    # Example: Train on office with default config
    config = PPOConfig.get_default("office")
    trainer = EnhancedPPOTrainer(config)
    trainer.train()