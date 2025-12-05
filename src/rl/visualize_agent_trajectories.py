#!/usr/bin/env python3
"""
Agent Trajectory Visualization Tool

Visualizes agent movement patterns during training or evaluation episodes.
Creates plots showing:
- Agent paths through the building
- Room visitation heatmaps
- Movement patterns over time
- Coordination between agents
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import networkx as nx
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.layouts import build_standard_office_layout, build_babycare_layout, build_two_floor_warehouse
from rl.ppo_config import PPOConfig
from rl.enhanced_training import EnhancedPPOTrainer
from rl.train_phase2_daycare import Phase2DaycareRewardShaper, Phase2DaycareTrainer
from rl.train_phase3_warehouse import Phase3WarehouseRewardShaper, Phase3WarehouseTrainer


class TrajectoryRecorder:
    """Records agent trajectories during episodes."""
    
    def __init__(self, env):
        self.env = env
        self.reset()
    
    def reset(self):
        """Reset trajectory recording."""
        self.trajectories = defaultdict(list)  # agent_id -> [(node_id, timestep)]
        self.actions = defaultdict(list)       # agent_id -> [action_str]
        self.node_visits = defaultdict(int)    # node_id -> visit_count
        self.timestep = 0
    
    def record_step(self, actions_dict):
        """Record agent positions and actions for current timestep."""
        for agent_id, agent in self.env.agents.items():
            self.trajectories[agent_id].append((agent.node_id, self.timestep))
            self.node_visits[agent.node_id] += 1
            
            if agent_id in actions_dict:
                self.actions[agent_id].append(actions_dict[agent_id])
        
        self.timestep += 1
    
    def get_summary(self):
        """Get trajectory summary statistics."""
        summary = {
            'total_steps': self.timestep,
            'agents': len(self.trajectories),
            'unique_nodes_visited': len(self.node_visits),
            'total_visits': sum(self.node_visits.values()),
        }
        
        # Per-agent stats
        for agent_id, traj in self.trajectories.items():
            unique_nodes = len(set([node for node, _ in traj]))
            summary[f'agent_{agent_id}_unique_nodes'] = unique_nodes
            summary[f'agent_{agent_id}_total_moves'] = len(traj)
        
        return summary


def visualize_trajectories_2d(recorder, env, save_path=None, show_people=True):
    """
    Create 2D trajectory visualization showing agent paths.
    
    Args:
        recorder: TrajectoryRecorder with recorded episode
        env: Environment instance
        save_path: Path to save figure (optional)
        show_people: Whether to show people locations
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get graph layout
    pos = nx.spring_layout(env.G, seed=42, k=2, iterations=50)
    
    # Draw base graph
    node_colors = []
    for node_id in env.G.nodes():
        node = env.nodes[node_id]
        if node.ntype == 'exit':
            node_colors.append('lightgreen')
        elif node.ntype == 'room':
            node_colors.append('lightblue')
        elif node.ntype == 'hallway':
            node_colors.append('lightgray')
        else:
            node_colors.append('white')
    
    nx.draw_networkx_nodes(env.G, pos, node_color=node_colors, 
                          node_size=300, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(env.G, pos, alpha=0.3, width=1, ax=ax)
    
    # Draw node labels (abbreviated)
    labels = {nid: nid[:8] for nid in env.G.nodes()}
    nx.draw_networkx_labels(env.G, pos, labels, font_size=6, ax=ax)
    
    # Draw people locations
    if show_people:
        people_nodes = set()
        for person in env.people.values():
            people_nodes.add(person.node_id)
        
        people_pos = {nid: pos[nid] for nid in people_nodes if nid in pos}
        if people_pos:
            nx.draw_networkx_nodes(env.G, people_pos, node_color='red',
                                  node_size=150, alpha=0.7, ax=ax,
                                  node_shape='^', label='People')
    
    # Draw agent trajectories
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    
    for agent_id, trajectory in recorder.trajectories.items():
        color = colors[agent_id % len(colors)]
        
        # Extract path
        path_nodes = [node_id for node_id, _ in trajectory]
        
        # Draw path
        for i in range(len(path_nodes) - 1):
            node1 = path_nodes[i]
            node2 = path_nodes[i + 1]
            
            if node1 in pos and node2 in pos:
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                
                # Fade older paths
                alpha = 0.3 + 0.7 * (i / len(path_nodes))
                
                ax.arrow(x1, y1, x2 - x1, y2 - y1,
                        head_width=0.02, head_length=0.03,
                        fc=color, ec=color, alpha=alpha, length_includes_head=True)
        
        # Mark start and end
        if path_nodes:
            start_node = path_nodes[0]
            end_node = path_nodes[-1]
            
            if start_node in pos:
                ax.plot(*pos[start_node], 'o', color=color, markersize=12,
                       label=f'Agent {agent_id} Start', markeredgecolor='black', markeredgewidth=2)
            
            if end_node in pos:
                ax.plot(*pos[end_node], '*', color=color, markersize=20,
                       label=f'Agent {agent_id} End', markeredgecolor='black', markeredgewidth=2)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Title and info
    summary = recorder.get_summary()
    title = f"Agent Trajectories\n"
    title += f"Steps: {summary['total_steps']}, "
    title += f"Nodes Visited: {summary['unique_nodes_visited']}, "
    title += f"Total Visits: {summary['total_visits']}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved trajectory plot: {save_path}")
    
    return fig


def visualize_heatmap(recorder, env, save_path=None):
    """
    Create heatmap showing node visitation frequency.
    
    Args:
        recorder: TrajectoryRecorder with recorded episode
        env: Environment instance
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get graph layout
    pos = nx.spring_layout(env.G, seed=42, k=2, iterations=50)
    
    # Calculate visit intensity
    max_visits = max(recorder.node_visits.values()) if recorder.node_visits else 1
    
    # Draw nodes with heatmap colors
    node_colors = []
    for node_id in env.G.nodes():
        visits = recorder.node_visits.get(node_id, 0)
        intensity = visits / max_visits if max_visits > 0 else 0
        
        # Red = high visits, blue = low visits
        color = plt.cm.RdYlBu_r(intensity)
        node_colors.append(color)
    
    nx.draw_networkx_nodes(env.G, pos, node_color=node_colors,
                          node_size=500, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(env.G, pos, alpha=0.3, width=1, ax=ax)
    
    # Add visit count labels
    labels = {}
    for node_id in env.G.nodes():
        visits = recorder.node_visits.get(node_id, 0)
        if visits > 0:
            labels[node_id] = f"{visits}"
    
    nx.draw_networkx_labels(env.G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                               norm=plt.Normalize(vmin=0, vmax=max_visits))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Visits', fontsize=12)
    
    # Title
    summary = recorder.get_summary()
    title = f"Node Visitation Heatmap\n"
    title += f"Total Visits: {summary['total_visits']}, "
    title += f"Unique Nodes: {summary['unique_nodes_visited']}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved heatmap: {save_path}")
    
    return fig


def visualize_timeline(recorder, save_path=None):
    """
    Create timeline showing agent positions over time.
    
    Args:
        recorder: TrajectoryRecorder with recorded episode
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(len(recorder.trajectories), 1, 
                             figsize=(14, 3 * len(recorder.trajectories)))
    
    if len(recorder.trajectories) == 1:
        axes = [axes]
    
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    
    for idx, (agent_id, trajectory) in enumerate(recorder.trajectories.items()):
        ax = axes[idx]
        
        # Extract data
        nodes = [node_id for node_id, _ in trajectory]
        times = [t for _, t in trajectory]
        
        # Create node -> index mapping
        unique_nodes = sorted(set(nodes))
        node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
        node_indices = [node_to_idx[node] for node in nodes]
        
        # Plot
        color = colors[agent_id % len(colors)]
        ax.plot(times, node_indices, '-o', color=color, alpha=0.7, 
               markersize=4, linewidth=2, label=f'Agent {agent_id}')
        
        # Y-axis: node names
        ax.set_yticks(range(len(unique_nodes)))
        ax.set_yticklabels([node[:12] for node in unique_nodes], fontsize=8)
        
        # Labels
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Node', fontsize=10)
        ax.set_title(f'Agent {agent_id} Movement Timeline', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved timeline: {save_path}")
    
    return fig


def record_episode(env, policy, config, deterministic=True):
    """
    Record a single episode with trajectory tracking.
    
    Args:
        env: Environment instance
        policy: Trained policy
        config: PPOConfig
        deterministic: Use deterministic actions
        
    Returns:
        recorder: TrajectoryRecorder with episode data
        stats: Episode statistics
    """
    recorder = TrajectoryRecorder(env)
    
    obs = env.reset()
    done = False
    step = 0
    max_steps = 300
    
    while not done and step < max_steps:
        # Get valid actions
        valid_actions_list = [env.get_valid_actions(i) for i in range(config.num_agents)]
        agent_indices_list = [env.get_agent_node_index(i) for i in range(config.num_agents)]
        
        if None in agent_indices_list:
            print(f"Warning: Agent position error at step {step}")
            break
        
        agent_indices = torch.tensor(agent_indices_list)
        
        # Select actions
        with torch.no_grad():
            actions, _, _ = policy.select_actions(
                obs, agent_indices, valid_actions_list, deterministic=deterministic
            )
        
        # Convert to action dict
        action_dict = {}
        for i, (valid_actions, action_idx) in enumerate(zip(valid_actions_list, actions)):
            sorted_actions = sorted(valid_actions)
            if action_idx.item() < len(sorted_actions):
                action_dict[i] = sorted_actions[action_idx.item()]
            else:
                action_dict[i] = 'wait'
        
        # Record before step
        recorder.record_step(action_dict)
        
        # Execute
        obs, _, done, info = env.do_action(action_dict)
        step += 1
    
    # Get final stats
    stats = env.get_statistics()
    
    return recorder, stats


def visualize_trained_model(scenario='office', checkpoint_path=None, 
                           num_episodes=3, output_dir='trajectory_visualizations'):
    """
    Load trained model and visualize agent trajectories.
    
    Args:
        scenario: 'office', 'daycare', or 'warehouse'
        checkpoint_path: Path to checkpoint (if None, looks for best_model.pt)
        num_episodes: Number of episodes to visualize
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*70)
    print(f"VISUALIZING AGENT TRAJECTORIES - {scenario.upper()}")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load appropriate trainer
    if scenario == 'office':
        config = PPOConfig(scenario='office', num_agents=2, seed=42)
        trainer = EnhancedPPOTrainer(config)
        env = build_standard_office_layout()
    elif scenario == 'daycare':
        config = PPOConfig(scenario='daycare', num_agents=3, seed=42)
        reward_shaper = Phase2DaycareRewardShaper()
        trainer = Phase2DaycareTrainer(config, reward_shaper)
        env = build_babycare_layout()
    elif scenario == 'warehouse':
        config = PPOConfig(scenario='warehouse', num_agents=4, seed=42)
        reward_shaper = Phase3WarehouseRewardShaper()
        trainer = Phase3WarehouseTrainer(config, reward_shaper)
        env = build_two_floor_warehouse()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Find checkpoint
    if checkpoint_path is None:
        # Look for best model
        log_dirs = sorted(Path('logs').glob(f'*{scenario}*'), key=lambda x: x.stat().st_mtime)
        if log_dirs:
            checkpoint_path = log_dirs[-1] / 'checkpoints' / 'best_model.pt'
            print(f"📂 Found checkpoint: {checkpoint_path}")
        else:
            print(f"❌ No checkpoint found for {scenario}")
            return
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        trainer.policy.load_state_dict(checkpoint['policy_state'])
        trainer.value.load_state_dict(checkpoint['value_state'])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"   Iteration: {checkpoint.get('iteration', 'unknown')}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Record episodes
    print(f"\n🎬 Recording {num_episodes} episodes...\n")
    
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}...")
        
        # Record episode
        env.reset(seed=42 + ep)
        recorder, stats = record_episode(env, trainer.policy, config, deterministic=True)
        
        # Print summary
        summary = recorder.get_summary()
        print(f"  Steps: {summary['total_steps']}")
        print(f"  Nodes visited: {summary['unique_nodes_visited']}")
        print(f"  People rescued: {stats.get('people_rescued', 0)}")
        print(f"  Rooms swept: {stats.get('nodes_swept', 0)}")
        
        # Create visualizations
        ep_dir = output_dir / scenario / f"episode_{ep + 1}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Trajectory plot
        visualize_trajectories_2d(recorder, env, 
                                 save_path=ep_dir / "trajectories.png")
        
        # 2. Heatmap
        visualize_heatmap(recorder, env,
                         save_path=ep_dir / "heatmap.png")
        
        # 3. Timeline
        visualize_timeline(recorder,
                          save_path=ep_dir / "timeline.png")
        
        print(f"  ✅ Saved visualizations to {ep_dir}\n")
    
    print("="*70)
    print(f"✅ ALL VISUALIZATIONS COMPLETE!")
    print(f"📁 Saved to: {output_dir / scenario}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize agent trajectories')
    parser.add_argument('--scenario', type=str, default='warehouse',
                       choices=['office', 'daycare', 'warehouse'],
                       help='Scenario to visualize (default: warehouse)')
    parser.add_argument('--checkpoint', type=str, default='logs/phase3_warehouse_4agents_20251119_040402/checkpoints/best_model.pt',
                       help='Path to checkpoint file (default: warehouse best_model)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize (default: 5)')
    parser.add_argument('--output', type=str, default='trajectory_visualizations',
                       help='Output directory (default: trajectory_visualizations)')
    
    args = parser.parse_args()
    
    visualize_trained_model(
        scenario=args.scenario,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        output_dir=args.output
    )
