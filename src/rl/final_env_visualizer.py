#!/usr/bin/env python3
"""
Visualization and diagnostics for final environment training.

Supports:
1. Agent trajectory mapping (shows agent paths on warehouse grid)
2. Time series metrics (coverage, rescue, hazards over time)
3. Heatmaps of node visitation frequency
4. Reward component breakdown per episode
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from scripts.visualize_layouts import compute_positions_by_floor


class FinalEnvironmentVisualizer:
    """Visualization tools for final warehouse environment."""
    
    @staticmethod
    def plot_agent_trajectories(
        trainer,
        episode_idx: int = 0,
        max_steps: int = 300,
        deterministic: bool = True,
        save_dir: str = None
    ) -> Dict:
        """
        Record agent trajectories and create visualization.
        
        Args:
            trainer: EnhancedPPOTrainer instance
            episode_idx: Episode number (for labeling)
            max_steps: Max steps per episode
            deterministic: Use greedy policy
            save_dir: Where to save PNG
            
        Returns:
            Dict with trajectory data and stats
        """
        env = trainer.env
        obs = env.reset()
        trainer.reward_shaper.reset()
        
        # Track trajectories
        agent_paths = {i: [] for i in range(trainer.config.num_agents)}
        node_visits = defaultdict(int)
        stats = env.get_statistics()
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Record agent positions
            for i in range(trainer.config.num_agents):
                node_id = env.agents[i].node_id
                agent_paths[i].append(node_id)
                node_visits[node_id] += 1
            
            # Get valid actions
            agent_indices = torch.tensor([
                env.get_agent_node_index(i)
                for i in range(trainer.config.num_agents)
            ])
            valid_actions_list = [
                env.get_valid_actions(i)
                for i in range(trainer.config.num_agents)
            ]
            
            # Select actions
            with torch.no_grad():
                actions, _, _ = trainer.policy.select_actions(
                    obs, agent_indices, valid_actions_list, deterministic
                )
            
            # Convert to action dict
            action_dict = {}
            for i in range(trainer.config.num_agents):
                action_idx = actions[i].item()
                action_str = trainer._idx_to_action_str(
                    action_idx,
                    valid_actions_list[i]
                )
                action_dict[i] = action_str
            
            # Step environment
            obs, _, done, _ = env.do_action(action_dict)
            step += 1
        
        # Final stats
        final_stats = env.get_statistics()
        trajectory_data = {
            'agent_paths': agent_paths,
            'node_visits': dict(node_visits),
            'steps': step,
            'coverage': final_stats.get('nodes_swept', 0),
            'rescued': final_stats.get('people_rescued', 0),
            'alive': final_stats.get('people_alive', 0),
        }
        
        # Create visualization if save_dir provided
        if save_dir:
            FinalEnvironmentVisualizer._draw_trajectories(
                env, agent_paths, node_visits, episode_idx, save_dir
            )
        
        return trajectory_data
    
    
    @staticmethod
    def _draw_trajectories(
        env, 
        agent_paths: Dict[int, List[str]], 
        node_visits: Dict[str, int],
        episode_idx: int,
        save_dir: str
    ):
        """Draw agent trajectories on warehouse layout."""
        
        # Get node positions
        positions = compute_positions_by_floor(env)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw nodes
        for nid, node in env.nodes.items():
            x, y = positions.get(nid, (0, 0))
            
            # Color by type
            if node.ntype == 'exit':
                color = 'lime'
                size = 200
            elif node.ntype == 'room':
                # Color by risk level
                color = 'red' if getattr(node, 'risk_level', 'normal') == 'high' else 'lightblue'
                size = 150
            else:  # hall
                color = 'lightgray'
                size = 100
            
            ax.scatter(x, y, s=size, c=color, edgecolors='black', linewidth=1.5, zorder=3)
            
            # Label node
            label = nid.replace('H_', '').replace('R_', '').replace('EXIT_WH_', 'EX_')
            ax.text(x, y, label, ha='center', va='center', fontsize=7, fontweight='bold', zorder=4)
        
        # Draw edges (building connectivity)
        for u_id, v_id in env.edges.keys():
            if u_id in positions and v_id in positions:
                x1, y1 = positions[u_id]
                x2, y2 = positions[v_id]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Draw agent trajectories with colors
        colors = ['blue', 'orange', 'green', 'purple', 'brown']
        for agent_id, path in agent_paths.items():
            color = colors[agent_id % len(colors)]
            
            # Draw path as connected line
            if len(path) > 1:
                for i in range(len(path) - 1):
                    node1, node2 = path[i], path[i+1]
                    if node1 in positions and node2 in positions:
                        x1, y1 = positions[node1]
                        x2, y2 = positions[node2]
                        ax.arrow(
                            x1, y1, x2-x1, y2-y1,
                            head_width=0.3, head_length=0.2,
                            fc=color, ec=color, alpha=0.4, linewidth=2, zorder=2
                        )
            
            # Mark start and end
            if path:
                start = positions[path[0]]
                ax.scatter(*start, s=100, marker='o', c=color, edgecolors='black', linewidth=2, zorder=5)
                
                end = positions[path[-1]]
                ax.scatter(*end, s=100, marker='x', c=color, edgecolors='black', linewidth=2, zorder=5)
        
        # Visit heatmap overlay
        max_visits = max(node_visits.values()) if node_visits else 1
        for nid, visits in node_visits.items():
            if nid in positions:
                x, y = positions[nid]
                alpha = 0.3 * (visits / max_visits)
                circle = patches.Circle((x, y), 0.5, color='yellow', alpha=alpha, zorder=1)
                ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_title(f'Agent Trajectories - Episode {episode_idx}\n' + 
                    f'Coverage: {sum(1 for v in node_visits.values())} nodes | ' +
                    f'Steps: {len(agent_paths[0])}', fontsize=12, fontweight='bold')
        ax.legend(
            [plt.scatter([], [], c=colors[i]) for i in range(len(agent_paths))],
            [f'Agent {i}' for i in range(len(agent_paths))],
            loc='upper right'
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'trajectories_ep{episode_idx:04d}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved trajectory visualization: {save_path}")
    
    
    @staticmethod
    def plot_episode_metrics(
        returns: List[float],
        rescue_counts: List[int],
        coverage_counts: List[int],
        save_path: str = None
    ):
        """Plot episode metrics over training."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        iters = np.arange(len(returns))
        
        # Returns
        axes[0].plot(iters, returns, linewidth=2, color='blue')
        axes[0].fill_between(iters, returns, alpha=0.3, color='blue')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Return')
        axes[0].set_title('Episode Returns')
        axes[0].grid(True, alpha=0.3)
        
        # Rescues
        axes[1].bar(iters, rescue_counts, color='orange', alpha=0.7)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('People Rescued')
        axes[1].set_title('Rescue Performance')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Coverage
        axes[2].bar(iters, coverage_counts, color='green', alpha=0.7)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Nodes Swept')
        axes[2].set_title('Coverage Performance')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved metrics plot: {save_path}")
        plt.close()


# Standalone function for enhanced_training.py compatibility
def plot_agent_trajectories(trainer, max_steps=300, deterministic=True, save_path=None, title="Agent Trajectories"):
    """
    Standalone wrapper for FinalEnvironmentVisualizer.plot_agent_trajectories.
    This is the function that enhanced_training.py imports.
    """
    if save_path:
        save_dir = os.path.dirname(save_path)
        episode_idx = 0  # Default episode index
    else:
        save_dir = None
        episode_idx = 0
    
    return FinalEnvironmentVisualizer.plot_agent_trajectories(
        trainer=trainer,
        episode_idx=episode_idx,
        max_steps=max_steps,
        deterministic=deterministic,
        save_dir=save_dir
    )
