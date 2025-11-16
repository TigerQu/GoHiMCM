"""
Experiment logging utilities for building evacuation RL.

===== NEW FILE: Implements logging for experiments =====

This module provides:
1. CSV logging for training/evaluation metrics
2. Optional TensorBoard integration
3. Plotting utilities for analysis
"""

import os
import csv
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentLogger:
    """
    Logs training and evaluation metrics to CSV and optionally TensorBoard.
    
    ===== CHANGE 1: NEW logging infrastructure =====
    Addresses paper requirement: "I want to see training curves"
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = False,
    ):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Base directory for logs (e.g., "logs/")
            experiment_name: Name of experiment (e.g., "office_baseline")
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize CSV files
        self.train_csv_path = os.path.join(self.exp_dir, "train_metrics.csv")
        self.eval_csv_path = os.path.join(self.exp_dir, "eval_metrics.csv")
        
        # Initialize CSV headers
        self._init_train_csv()
        self._init_eval_csv()
        
        # TensorBoard writer
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(self.exp_dir, "tensorboard")
            self.tb_writer = SummaryWriter(tb_dir)
        
        print(f"📊 Experiment logger initialized: {self.exp_dir}")
    
    
    def _init_train_csv(self):
        """Initialize training metrics CSV with headers."""
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration',
                'policy_loss',
                'value_loss',
                'entropy',
                'avg_return',
                'episode_length',
                'people_rescued',
                'people_found',
                'people_alive',
                'nodes_swept',
                'high_risk_redundancy',
                'active_agents',
                'sweep_complete',
            ])
    
    
    def _init_eval_csv(self):
        """Initialize evaluation metrics CSV with headers."""
        with open(self.eval_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration',
                'eval_return_mean',
                'eval_return_std',
                'eval_people_rescued_mean',
                'eval_people_found_mean',
                'eval_people_alive_mean',
                'eval_nodes_swept_mean',
                'eval_high_risk_redundancy_mean',
                'eval_sweep_complete_rate',
                'eval_episode_length_mean',
            ])
    
    
    def log_iteration(
        self,
        iteration: int,
        losses: Dict[str, float],
        episode_stats: Dict[str, float],
        episode_return: float,
    ):
        """
        Log training metrics for one iteration.
        
        ===== MAIN METHOD: Called after each training iteration =====
        
        Args:
            iteration: Training iteration number
            losses: Dict with 'policy_loss', 'value_loss', 'entropy'
            episode_stats: Dict from reward_shaper.get_episode_summary()
            episode_return: Total shaped reward for episode
        """
        # Write to CSV
        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                losses['policy_loss'],
                losses['value_loss'],
                losses['entropy'],
                episode_return,
                episode_stats.get('time_step', 0),
                episode_stats.get('people_rescued', 0),
                episode_stats.get('people_found', 0),
                episode_stats.get('people_alive', 0),
                episode_stats.get('nodes_swept', 0),
                episode_stats.get('high_risk_redundancy', 0.0),
                episode_stats.get('active_agents', 0),
                int(episode_stats.get('sweep_complete', False)),
            ])
        
        # Write to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Loss/policy', losses['policy_loss'], iteration)
            self.tb_writer.add_scalar('Loss/value', losses['value_loss'], iteration)
            self.tb_writer.add_scalar('Loss/entropy', losses['entropy'], iteration)
            self.tb_writer.add_scalar('Episode/return', episode_return, iteration)
            self.tb_writer.add_scalar('Episode/length', episode_stats.get('time_step', 0), iteration)
            self.tb_writer.add_scalar('Rescue/people_rescued', episode_stats.get('people_rescued', 0), iteration)
            self.tb_writer.add_scalar('Coverage/nodes_swept', episode_stats.get('nodes_swept', 0), iteration)
            self.tb_writer.add_scalar('Coverage/high_risk_redundancy', episode_stats.get('high_risk_redundancy', 0.0), iteration)
    
    
    def log_eval(
        self,
        iteration: int,
        eval_summary: Dict[str, float],
    ):
        """
        Log evaluation metrics.
        
        Args:
            iteration: Training iteration when eval was performed
            eval_summary: Dict from evaluate() method with aggregated stats
        """
        # Write to CSV
        with open(self.eval_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                eval_summary['return_mean'],
                eval_summary['return_std'],
                eval_summary['people_rescued_mean'],
                eval_summary['people_found_mean'],
                eval_summary['people_alive_mean'],
                eval_summary['nodes_swept_mean'],
                eval_summary['high_risk_redundancy_mean'],
                eval_summary['sweep_complete_rate'],
                eval_summary['episode_length_mean'],
            ])
        
        # Write to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar('Eval/return_mean', eval_summary['return_mean'], iteration)
            self.tb_writer.add_scalar('Eval/people_rescued_mean', eval_summary['people_rescued_mean'], iteration)
            self.tb_writer.add_scalar('Eval/high_risk_redundancy_mean', eval_summary['high_risk_redundancy_mean'], iteration)
            self.tb_writer.add_scalar('Eval/sweep_complete_rate', eval_summary['sweep_complete_rate'], iteration)
    
    
    def close(self):
        """Close logger and save final plots."""
        if self.tb_writer:
            self.tb_writer.close()
        
        # Generate final plots
        self.plot_training_curves()
        print(f"✅ Logs saved to: {self.exp_dir}")
    
    
    def plot_training_curves(self):
        """
        Generate training curve plots from logged data.
        
        ===== CHANGE 2: Automatic plot generation =====
        Creates publication-ready figures for paper.
        """
        # Read training data
        train_data = []
        with open(self.train_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            train_data = list(reader)
        
        if not train_data:
            return
        
        # Convert to arrays
        iterations = [int(row['iteration']) for row in train_data]
        returns = [float(row['avg_return']) for row in train_data]
        people_rescued = [int(row['people_rescued']) for row in train_data]
        high_risk_redundancy = [float(row['high_risk_redundancy']) for row in train_data]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Curves: {self.experiment_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Returns
        axes[0, 0].plot(iterations, returns, linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].set_title('Episode Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: People Rescued
        axes[0, 1].plot(iterations, people_rescued, linewidth=2, color='green')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('People Rescued')
        axes[0, 1].set_title('Rescue Performance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: High-Risk Redundancy
        axes[1, 0].plot(iterations, high_risk_redundancy, linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('High-Risk Redundancy')
        axes[1, 0].set_title('Redundancy Coverage')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: Policy Loss
        policy_losses = [float(row['policy_loss']) for row in train_data]
        axes[1, 1].plot(iterations, policy_losses, linewidth=2, color='red')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Policy Loss')
        axes[1, 1].set_title('Policy Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved to: {plot_path}")