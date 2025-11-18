#!/usr/bin/env python3
"""
Visualization and diagnostics for final environment training.

Supports:
1. Agent trajectory mapping (shows agent paths on warehouse grid)
2. Visit heatmaps of node visitation frequency
3. Directional arrows showing agent movement
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List
from collections import defaultdict

try:
    from scripts.visualize_layouts import compute_positions_by_floor
except ImportError:
    # Fallback to networkx layout if visualize_layouts not available
    import networkx as nx
    def compute_positions_by_floor(env):
        return nx.spring_layout(env.G, seed=42, k=2, iterations=50)


def plot_agent_trajectories(
    trainer,
    max_steps: int = 300,
    deterministic: bool = True,
    save_path: str = None,
    title: str = "Agent Trajectories"
):
    """
    Record agent trajectories and create visualization with arrows and heatmaps.
    
    Args:
        trainer: EnhancedPPOTrainer instance
        max_steps: Max steps per episode
        deterministic: Use greedy policy
        save_path: Where to save PNG
        title: Plot title
        
    Returns:
        Dict with trajectory data and stats
    """
    env = trainer.env
    obs = env.reset()
    
    # Track trajectories
    agent_paths = {i: [] for i in range(trainer.config.num_agents)}
    node_visits = defaultdict(int)
    
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
        ], device=trainer.device)
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
    
    # Create visualization if save_path provided
    if save_path:
        _draw_trajectories(
            env, agent_paths, node_visits, final_stats, step, save_path, title
        )
    
    return {
        'agent_paths': agent_paths,
        'node_visits': dict(node_visits),
        'steps': step,
        'coverage': final_stats.get('nodes_swept', 0),
    }


def _draw_trajectories(
    env, 
    agent_paths: Dict[int, List[str]], 
    node_visits: Dict[str, int],
    stats: Dict,
    steps: int,
    save_path: str,
    title: str
):
    """Draw agent trajectories on warehouse layout with arrows and heatmap."""
    
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
            # Color by risk level if available
            risk = getattr(node, 'risk_level', 'normal')
            color = 'red' if risk == 'high' else 'lightblue'
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
    
    # Draw agent trajectories with directional arrows
    colors = ['blue', 'orange', 'green', 'purple', 'brown']
    for agent_id, path in agent_paths.items():
        color = colors[agent_id % len(colors)]
        
        # Draw path as connected arrows
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
            start = positions.get(path[0])
            if start:
                ax.scatter(*start, s=100, marker='o', c=color, edgecolors='black', linewidth=2, zorder=5)
            end = positions.get(path[-1])
            if end:
                ax.scatter(*end, s=100, marker='x', c=color, edgecolors='black', linewidth=2, zorder=5)
    
    # Visit heatmap overlay (yellow circles showing frequently visited nodes)
    max_visits = max(node_visits.values()) if node_visits else 1
    for nid, visits in node_visits.items():
        if nid in positions:
            x, y = positions[nid]
            alpha = 0.3 * (visits / max_visits)
            circle = patches.Circle((x, y), 0.5, color='yellow', alpha=alpha, zorder=1)
            ax.add_patch(circle)
    
    ax.set_aspect('equal')
    ax.set_title(
        f'{title}\nCoverage: {stats.get("nodes_swept", 0)} nodes | Steps: {steps} | '
        f'Rescued: {stats.get("people_rescued", 0)} | Alive: {stats.get("people_alive", 0)}',
        fontsize=12, fontweight='bold'
    )
    ax.legend(
        [plt.scatter([], [], c=colors[i]) for i in range(len(agent_paths))],
        [f'Agent {i} ({len(set(agent_paths[i]))} unique)' for i in range(len(agent_paths))],
        loc='upper right'
    )
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved trajectory visualization: {save_path}")
