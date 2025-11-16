"""
Visualization and analysis utilities.

===== NEW FILE: Creates paper-ready figures =====

Provides:
1. Attention heatmap visualization (GAT analysis)
2. Time series plots (occupants, hazards, coverage)
3. Comparative performance plots
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

from rl.new_gat import GAT
from rl.new_ppo import Policy, Value
from rl.enhanced_training import EnhancedPPOTrainer
from rl.ppo_config import PPOConfig


class AttentionVisualizer:
    """
    Visualize GAT attention weights for interpretability.
    
    ===== CHANGE 16: Attention heatmap analysis =====
    Shows which building connections the policy focuses on.
    """
    
    def __init__(self, gat: GAT):
        """
        Initialize with trained GAT.
        
        Args:
            gat: Trained GAT network
        """
        self.gat = gat
    
    
    def get_attention_weights(self, data) -> Tuple:
        """
        Forward pass with attention weight extraction.
        
        ===== Extract attention from GAT layers =====
        PyG's GATConv can return attention weights.
        """
        x = data.x
        edge_index = data.edge_index
        
        # Layer 1 with attention
        out1, (e1, alpha1) = self.gat.gat1(
            x, edge_index,
            return_attention_weights=True
        )
        out1 = self.gat.norm1(out1)
        out1 = self.gat.elu(out1)
        
        # Layer 2 with attention
        out2, (e2, alpha2) = self.gat.gat2(
            out1, edge_index,
            return_attention_weights=True
        )
        out2 = self.gat.norm2(out2)
        out2 = self.gat.elu(out2)
        
        # Layer 3 (final)
        out3 = self.gat.gat3(out2, edge_index)
        
        return out3, (e2, alpha2)  # Return layer 2 attention for visualization
    
    
    def plot_attention_heatmap(
        self,
        env,
        obs,
        save_path: str = "attention_heatmap.png"
    ):
        """
        Plot attention weights on building graph.
        
        ===== CHANGE 17: Create attention visualization =====
        
        Args:
            env: Environment instance (for graph structure)
            obs: Current observation (Data object)
            save_path: Where to save figure
        """
        # Get attention weights
        _, (edge_index, alpha) = self.get_attention_weights(obs)
        
        # Average attention across heads
        alpha = alpha.mean(dim=1).detach().numpy()
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        node_labels = {}
        
        # Add nodes
        for nid, node in env.nodes.items():
            G.add_node(nid)
            # Label format: room type + hazard status
            label = nid
            if node.on_fire:
                label += " 🔥"
            elif node.smoky:
                label += " 💨"
            node_labels[nid] = label
        
        # Add edges with attention weights
        edge_weights = {}
        edge_index_np = edge_index.numpy()
        node_ids = list(env.nodes.keys())
        
        for i in range(edge_index_np.shape[1]):
            u_idx = edge_index_np[0, i]
            v_idx = edge_index_np[1, i]
            u = node_ids[u_idx]
            v = node_ids[v_idx]
            
            if (u, v) not in edge_weights and (v, u) not in edge_weights:
                edge_weights[(u, v)] = alpha[i]
                G.add_edge(u, v, weight=alpha[i])
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        node_colors = []
        for nid in G.nodes():
            node = env.nodes[nid]
            if node.on_fire:
                node_colors.append('red')
            elif node.smoky:
                node_colors.append('orange')
            elif node.agent_here:
                node_colors.append('blue')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges with attention-based width
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        weights_normalized = [(w - min(weights)) / (max(weights) - min(weights) + 1e-8) for w in weights]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 5 for w in weights_normalized],
            alpha=0.6,
            edge_color=weights_normalized,
            edge_cmap=plt.cm.Reds
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
        
        plt.title("GAT Attention Weights\n(Thicker edges = higher attention)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Attention heatmap saved to: {save_path}")


class TimeSeriesAnalyzer:
    """
    Analyze environment dynamics over time.
    
    ===== CHANGE 18: Time series diagnostic plots =====
    Shows how coverage, occupants, and hazards evolve.
    """
    
    @staticmethod
    def run_diagnostic_episode(
        trainer: EnhancedPPOTrainer,
        max_steps: int = 200,
        deterministic: bool = True
    ) -> Dict[str, List]:
        """
        Run episode and collect time series data.
        
        Args:
            trainer: Trained PPO trainer
            max_steps: Maximum episode length
            deterministic: Use greedy actions
            
        Returns:
            log: Dict of time series data
        """
        env = trainer.env
        obs = env.reset()
        trainer.reward_shaper.reset()
        
        # Initialize log
        log = {
            't': [],
            'fraction_rooms_swept': [],
            'people_alive': [],
            'people_in_fire': [],
            'people_in_smoke': [],
            'people_rescued': [],
            'high_risk_redundancy': [],
            'active_agents': [],
        }
        
        # Count total sweep targets
        total_rooms = sum(
            1 for n in env.nodes.values()
            if n.ntype in env.config.get("sweep_node_types", {"room"})
        )
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Get agent positions
            agent_indices = torch.tensor([
                env.get_agent_node_index(i)
                for i in range(trainer.config.num_agents)
            ])
            
            # Get valid actions
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
            
            # Collect statistics
            stats = env.get_statistics()
            
            # Count people in hazards
            people_in_fire = 0
            people_in_smoke = 0
            for person in env.people.values():
                if person.is_alive and person.node_id:
                    node = env.nodes.get(person.node_id)
                    if node:
                        if node.on_fire:
                            people_in_fire += 1
                        elif node.smoky:
                            people_in_smoke += 1
            
            # Log timestep
            log['t'].append(stats['time_step'])
            log['fraction_rooms_swept'].append(stats['nodes_swept'] / max(total_rooms, 1))
            log['people_alive'].append(stats['people_alive'])
            log['people_in_fire'].append(people_in_fire)
            log['people_in_smoke'].append(people_in_smoke)
            log['people_rescued'].append(stats['people_rescued'])
            log['high_risk_redundancy'].append(stats['high_risk_redundancy'])
            log['active_agents'].append(stats['active_agents'])
            
            step += 1
        
        return log
    
    
    @staticmethod
    def plot_time_series(
        log: Dict[str, List],
        save_path: str = "time_series.png",
        title: str = "Episode Dynamics"
    ):
        """
        Plot time series of key metrics.
        
        Args:
            log: Time series data from run_diagnostic_episode
            save_path: Where to save figure
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        t = log['t']
        
        # Plot 1: Coverage over time
        axes[0, 0].plot(t, log['fraction_rooms_swept'], linewidth=2, color='blue')
        axes[0, 0].fill_between(t, 0, log['fraction_rooms_swept'], alpha=0.3)
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Fraction of Rooms Swept')
        axes[0, 0].set_title('Sweep Coverage Progress')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: People status
        axes[0, 1].plot(t, log['people_alive'], linewidth=2, color='green', label='Alive')
        axes[0, 1].plot(t, log['people_rescued'], linewidth=2, color='blue', label='Rescued')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Number of People')
        axes[0, 1].set_title('Occupant Status')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: People in hazards
        axes[1, 0].plot(t, log['people_in_fire'], linewidth=2, color='red', label='In Fire')
        axes[1, 0].plot(t, log['people_in_smoke'], linewidth=2, color='orange', label='In Smoke')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Number of People')
        axes[1, 0].set_title('People Exposed to Hazards')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: High-risk redundancy
        axes[1, 1].plot(t, log['high_risk_redundancy'], linewidth=2, color='purple')
        axes[1, 1].fill_between(t, 0, log['high_risk_redundancy'], alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('High-Risk Redundancy')
        axes[1, 1].set_title('High-Risk Room Coverage')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Time series plot saved to: {save_path}")


def analyze_trained_model(checkpoint_path: str, scenario: str = "office"):
    """
    Complete analysis of a trained model.
    
    ===== CHANGE 19: One-stop analysis script =====
    Generates all visualizations for a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        scenario: Scenario name
    """
    print(f"\n🔍 Analyzing model: {checkpoint_path}\n")
    
    # Load config and create trainer
    config = PPOConfig.get_default(scenario)
    trainer = EnhancedPPOTrainer(config)
    trainer.load_checkpoint(checkpoint_path)
    
    # Create output directory
    analysis_dir = "analysis_results"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Run diagnostic episode and get time series
    print("Running diagnostic episode...")
    log = TimeSeriesAnalyzer.run_diagnostic_episode(trainer)
    
    # 2. Plot time series
    TimeSeriesAnalyzer.plot_time_series(
        log,
        save_path=os.path.join(analysis_dir, f"{scenario}_time_series.png"),
        title=f"{scenario.capitalize()} Scenario: Episode Dynamics"
    )
    
    # 3. Get observation at interesting timestep (e.g., mid-episode)
    mid_step = len(log['t']) // 2
    obs = trainer.env.get_observation()
    
    # 4. Plot attention heatmap
    print("Generating attention heatmap...")
    attention_viz = AttentionVisualizer(trainer.policy.gat)
    attention_viz.plot_attention_heatmap(
        trainer.env,
        obs,
        save_path=os.path.join(analysis_dir, f"{scenario}_attention.png")
    )
    
    print(f"\n Analysis complete! Results saved to: {analysis_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trained PPO model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='office',
        choices=['office', 'daycare', 'warehouse'],
        help="Scenario name"
    )
    
    args = parser.parse_args()
    
    analyze_trained_model(args.checkpoint, args.scenario)