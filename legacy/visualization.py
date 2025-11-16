import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Wedge
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Import environment (assumes single_floor.py is in same directory)
from Env_sim.env import BuildingFireEnvironment
from Env_sim.layouts import build_standard_office_layout


# =============================================================================
# PLOTTING CONFIGURATION (Paper-ready defaults)
# =============================================================================

PLOT_CONFIG = {
    # Figure sizing (inches)
    'single_panel_size': (3.5, 3.0),  # Single column width
    'double_panel_size': (7.0, 3.0),  # Double column width
    'multi_panel_size': (7.0, 6.0),   # Full page width
    
    # Fonts (consistent with LaTeX documents)
    'font_family': 'serif',
    'font_size_title': 10,
    'font_size_label': 9,
    'font_size_tick': 8,
    'font_size_legend': 8,
    
    # Line weights
    'line_width_thin': 0.5,
    'line_width_medium': 1.0,
    'line_width_thick': 1.5,
    
    # Colors (colorblind-friendly, grayscale-safe)
    'color_fire': '#d62728',      # Red
    'color_smoke': '#7f7f7f',     # Gray
    'color_agent': '#1f77b4',     # Blue
    'color_person': '#2ca02c',    # Green
    'color_exit': '#9467bd',      # Purple
    'color_swept': '#e377c2',     # Pink
    'color_trajectory': '#ff7f0e', # Orange
    
    # Alpha values
    'alpha_smoke': 0.3,
    'alpha_fire': 0.5,
    'alpha_swept': 0.2,
    
    # DPI
    'dpi_save': 300,
    'dpi_display': 100,
}


# =============================================================================
# GRAPH LAYOUT COMPUTATION
# =============================================================================

def compute_layout(env: BuildingFireEnvironment) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D positions for nodes based on building structure.
    
    Layout strategy:
    - Exits on far left/right
    - Hallway segments in center (horizontal line)
    - Top rooms above hallway
    - Bottom rooms below hallway
    
    Returns:
        Dict mapping node_id to (x, y) coordinates
    """
    pos = {}
    
    # Hallway segments (horizontal spine)
    for i in range(3):
        pos[f"H{i}"] = (i * 3.0, 0.0)
    
    # Top rooms
    for i in range(3):
        pos[f"RT{i}"] = (i * 3.0, 2.0)
    
    # Bottom rooms
    for i in range(3):
        pos[f"RB{i}"] = (i * 3.0, -2.0)
    
    # Exits
    pos["EXIT_LEFT"] = (-1.5, 0.0)
    pos["EXIT_RIGHT"] = (7.5, 0.0)
    
    return pos


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class FireEvacVisualizer:
    """
    Publication-quality visualization for fire evacuation simulations.
    
    Generates static figures suitable for scientific papers (PDF/SVG).
    """
    
    def __init__(self, env: BuildingFireEnvironment, config: Optional[Dict] = None):
        """
        Initialize visualizer.
        
        Args:
            env: Fire evacuation environment
            config: Optional plotting configuration (overrides defaults)
        """
        self.env = env
        self.config = {**PLOT_CONFIG, **(config or {})}
        self.layout = compute_layout(env)
        
        # Set matplotlib style
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['font.size'] = self.config['font_size_label']
        plt.rcParams['axes.linewidth'] = self.config['line_width_thin']
        plt.rcParams['xtick.major.width'] = self.config['line_width_thin']
        plt.rcParams['ytick.major.width'] = self.config['line_width_thin']
    
    def plot_snapshot(self, ax: plt.Axes, title: Optional[str] = None,
                     show_legend: bool = True, show_scale: bool = True) -> None:
        """
        Plot single snapshot of environment state.
        
        Args:
            ax: Matplotlib axes
            title: Panel title (e.g., "t = 0s")
            show_legend: Whether to show legend
            show_scale: Whether to show scale bar
        """
        # Draw edges (building structure)
        for u, v in self.env.G.edges():
            x1, y1 = self.layout[u]
            x2, y2 = self.layout[v]
            
            edge_meta = self.env.G.edges[u, v]['meta']
            linestyle = '--' if edge_meta.door else '-'
            linewidth = self.config['line_width_thick'] if edge_meta.fire_door else self.config['line_width_medium']
            
            ax.plot([x1, x2], [y1, y2], 'k-', 
                   linestyle=linestyle, linewidth=linewidth, alpha=0.3, zorder=1)
        
        # Draw nodes
        for nid, node in self.env.nodes.items():
            x, y = self.layout[nid]
            
            # Node size based on type
            if node.ntype == 'exit':
                size = 400
                marker = 's'
            elif node.ntype == 'hall':
                size = 300
                marker = 'o'
            else:  # room
                size = 500
                marker = 'o'
            
            # Node color based on state
            if node.on_fire:
                color = self.config['color_fire']
                alpha = self.config['alpha_fire']
                edgecolor = 'darkred'
            elif node.smoky:
                color = self.config['color_smoke']
                alpha = self.config['alpha_smoke']
                edgecolor = 'gray'
            elif node.swept:
                color = self.config['color_swept']
                alpha = self.config['alpha_swept']
                edgecolor = 'purple'
            else:
                color = 'white'
                alpha = 1.0
                edgecolor = 'black'
            
            ax.scatter(x, y, s=size, c=color, alpha=alpha, 
                      edgecolors=edgecolor, linewidths=self.config['line_width_medium'],
                      marker=marker, zorder=2)
            
            # Label
            label_y_offset = -0.4 if node.ntype == 'room' else -0.3
            ax.text(x, y + label_y_offset, nid, 
                   ha='center', va='top', fontsize=self.config['font_size_tick'],
                   zorder=3)
        
        # Draw people
        for person in self.env.people.values():
            if person.rescued or not person.is_alive:
                continue
            
            if person.on_edge:
                # Interpolate position on edge
                x1, y1 = self.layout[person.edge_u]
                x2, y2 = self.layout[person.edge_v]
                # Progress along edge (eta decreases, so invert)
                edge_meta = self.env.G.edges[person.edge_u, person.edge_v]['meta']
                total_time = edge_meta.length / person.v_class
                progress = 1.0 - (person.edge_eta / max(total_time, 1.0))
                x = x1 + progress * (x2 - x1)
                y = y1 + progress * (y2 - y1)
            else:
                x, y = self.layout[person.node_id]
                # Offset slightly for visibility
                x += np.random.uniform(-0.2, 0.2)
                y += np.random.uniform(-0.2, 0.2)
            
            # Person marker
            marker_size = 80 if person.mobility == 'child' else 100
            marker_shape = '^' if person.mobility == 'child' else 'o'
            color = self.config['color_person'] if person.seen else 'gray'
            
            ax.scatter(x, y, s=marker_size, c=color, marker=marker_shape,
                      edgecolors='black', linewidths=self.config['line_width_thin'],
                      zorder=4, alpha=0.8)
        
        # Draw agents
        for agent in self.env.agents.values():
            x, y = self.layout[agent.node_id]
            
            # Agent marker (star)
            ax.scatter(x, y, s=300, c=self.config['color_agent'], marker='*',
                      edgecolors='black', linewidths=self.config['line_width_medium'],
                      zorder=5)
            
            # Agent ID label
            ax.text(x, y + 0.5, f"A{agent.agent_id}", 
                   ha='center', va='bottom', fontsize=self.config['font_size_tick'],
                   fontweight='bold', zorder=6)
        
        # Formatting
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=self.config['font_size_title'], pad=10)
        
        # Legend
        if show_legend:
            legend_elements = [
                mpatches.Patch(facecolor=self.config['color_fire'], alpha=self.config['alpha_fire'],
                              edgecolor='darkred', label='Fire'),
                mpatches.Patch(facecolor=self.config['color_smoke'], alpha=self.config['alpha_smoke'],
                              edgecolor='gray', label='Smoke'),
                mpatches.Patch(facecolor=self.config['color_swept'], alpha=self.config['alpha_swept'],
                              edgecolor='purple', label='Swept'),
                plt.Line2D([0], [0], marker='*', color='w', 
                          markerfacecolor=self.config['color_agent'], markersize=10,
                          markeredgecolor='black', label='Agent'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=self.config['color_person'], markersize=8,
                          markeredgecolor='black', label='Person (aware)'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='gray', markersize=8,
                          markeredgecolor='black', label='Person (unaware)'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', 
                     fontsize=self.config['font_size_legend'],
                     frameon=True, fancybox=False, shadow=False)
        
        # Scale bar
        if show_scale:
            scale_length = 3.0  # meters
            scale_x = -1.0
            scale_y = -3.5
            ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
                   'k-', linewidth=self.config['line_width_thick'])
            ax.text(scale_x + scale_length / 2, scale_y - 0.3, f'{scale_length:.0f}m',
                   ha='center', va='top', fontsize=self.config['font_size_tick'])
    
    def plot_temporal_evolution(self, snapshots: List[int], 
                               output_path: str, format: str = 'pdf') -> None:
        """
        Create multi-panel figure showing temporal evolution.
        
        Args:
            snapshots: List of time steps to capture
            output_path: Output file path
            format: 'pdf' or 'svg'
        """
        num_snapshots = len(snapshots)
        ncols = min(3, num_snapshots)
        nrows = (num_snapshots + ncols - 1) // ncols
        
        fig = plt.figure(figsize=self.config['multi_panel_size'])
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, 
                              hspace=0.3, wspace=0.2,
                              left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Store initial state
        initial_state = {
            'time_step': self.env.time_step,
            'stats': dict(self.env.stats),
        }
        
        for idx, t in enumerate(snapshots):
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col])
            
            # Reset and simulate to time t
            self.env.reset(seed=42, fire_node="RT1")
            for _ in range(t):
                # Simple sweep strategy
                actions = {}
                for agent_id in self.env.agents.keys():
                    valid = self.env.get_valid_actions(agent_id)
                    # Prioritize search, then move
                    if "search" in valid and self.env.agents[agent_id].node_id != "EXIT_LEFT" and self.env.agents[agent_id].node_id != "EXIT_RIGHT":
                        actions[agent_id] = "search"
                    else:
                        move_actions = [a for a in valid if a.startswith("move_")]
                        if move_actions:
                            actions[agent_id] = move_actions[0]
                        else:
                            actions[agent_id] = "wait"
                
                self.env.do_action(actions)
            
            # Plot snapshot
            title = f"t = {t}s"
            show_legend = (idx == 0)  # Only first panel
            show_scale = (idx == num_snapshots - 1)  # Only last panel
            self.plot_snapshot(ax, title=title, show_legend=show_legend, show_scale=show_scale)
        
        # Overall title
        fig.suptitle('Temporal Evolution of Fire Evacuation Sweep', 
                    fontsize=self.config['font_size_title'] + 2, fontweight='bold')
        
        # Save
        plt.savefig(output_path, format=format, dpi=self.config['dpi_save'],
                   bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Saved temporal evolution figure: {output_path}")
        plt.close()
    
    def plot_heatmap_with_trajectories(self, num_steps: int,
                                      output_path: str, format: str = 'pdf') -> None:
        """
        Create hazard heatmap with agent trajectory overlay.
        
        Args:
            num_steps: Number of simulation steps
            output_path: Output file path
            format: 'pdf' or 'svg'
        """
        fig, (ax_heat, ax_traj) = plt.subplots(1, 2, figsize=self.config['double_panel_size'])
        
        # Initialize
        self.env.reset(seed=42, fire_node="RT1")
        
        # Track agent trajectories
        trajectories = {agent_id: [] for agent_id in self.env.agents.keys()}
        
        # Track hazard intensity over time
        fire_intensity = {nid: [] for nid in self.env.nodes.keys()}
        smoke_intensity = {nid: [] for nid in self.env.nodes.keys()}
        
        # Simulate
        for t in range(num_steps):
            # Record agent positions
            for agent_id, agent in self.env.agents.items():
                pos = self.layout[agent.node_id]
                trajectories[agent_id].append((t, pos[0], pos[1]))
            
            # Record hazards
            for nid, node in self.env.nodes.items():
                fire_intensity[nid].append(1.0 if node.on_fire else 0.0)
                smoke_intensity[nid].append(1.0 if node.smoky else 0.0)
            
            # Step
            actions = {}
            for agent_id in self.env.agents.keys():
                valid = self.env.get_valid_actions(agent_id)
                if "search" in valid:
                    actions[agent_id] = "search"
                else:
                    move_actions = [a for a in valid if a.startswith("move_")]
                    actions[agent_id] = move_actions[0] if move_actions else "wait"
            
            self.env.do_action(actions)
        
        # Plot 1: Hazard heatmap
        ax_heat.set_title('Cumulative Hazard Exposure', 
                         fontsize=self.config['font_size_title'])
        
        # Compute cumulative exposure
        for nid, node in self.env.nodes.items():
            x, y = self.layout[nid]
            fire_cum = sum(fire_intensity[nid])
            smoke_cum = sum(smoke_intensity[nid])
            total_exposure = fire_cum * 2.0 + smoke_cum  # Weight fire more
            
            if total_exposure > 0:
                size = 300 + total_exposure * 50
                alpha = min(0.3 + total_exposure * 0.05, 0.9)
                color = plt.cm.Reds(min(total_exposure / num_steps, 1.0))
                
                ax_heat.scatter(x, y, s=size, c=[color], alpha=alpha,
                              edgecolors='darkred', linewidths=self.config['line_width_medium'],
                              zorder=2)
        
        # Draw building structure
        for u, v in self.env.G.edges():
            x1, y1 = self.layout[u]
            x2, y2 = self.layout[v]
            ax_heat.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, linewidth=0.5, zorder=1)
        
        ax_heat.set_aspect('equal')
        ax_heat.axis('off')
        
        # Plot 2: Agent trajectories
        ax_traj.set_title('Agent Trajectories',
                         fontsize=self.config['font_size_title'])
        
        # Draw building
        for u, v in self.env.G.edges():
            x1, y1 = self.layout[u]
            x2, y2 = self.layout[v]
            ax_traj.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Draw nodes
        for nid in self.env.nodes.keys():
            x, y = self.layout[nid]
            ax_traj.scatter(x, y, s=100, c='lightgray', edgecolors='gray',
                          linewidths=0.5, zorder=2)
        
        # Draw trajectories
        colors = [self.config['color_agent'], self.config['color_trajectory']]
        for agent_id, traj in trajectories.items():
            if not traj:
                continue
            
            xs = [p[1] for p in traj]
            ys = [p[2] for p in traj]
            
            # Plot trajectory with arrows
            ax_traj.plot(xs, ys, '-', color=colors[agent_id % len(colors)],
                        linewidth=self.config['line_width_thick'], alpha=0.7,
                        label=f'Agent {agent_id}', zorder=3)
            
            # Add arrow at end
            if len(xs) > 1:
                ax_traj.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                               arrowprops=dict(arrowstyle='->', color=colors[agent_id % len(colors)],
                                             lw=self.config['line_width_thick']),
                               zorder=4)
            
            # Mark start and end
            ax_traj.scatter(xs[0], ys[0], s=150, marker='o', 
                          c=colors[agent_id % len(colors)], edgecolors='black',
                          linewidths=self.config['line_width_medium'], zorder=5)
            ax_traj.scatter(xs[-1], ys[-1], s=150, marker='s',
                          c=colors[agent_id % len(colors)], edgecolors='black',
                          linewidths=self.config['line_width_medium'], zorder=5)
        
        ax_traj.legend(fontsize=self.config['font_size_legend'])
        ax_traj.set_aspect('equal')
        ax_traj.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, format=format, dpi=self.config['dpi_save'],
                   bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Saved heatmap figure: {output_path}")
        plt.close()
    
    def plot_coverage_progression(self, num_steps: int,
                                 output_path: str, format: str = 'pdf') -> None:
        """
        Create coverage progression figure with statistical insets.
        
        Args:
            num_steps: Number of simulation steps
            output_path: Output file path
            format: 'pdf' or 'svg'
        """
        fig = plt.figure(figsize=self.config['multi_panel_size'])
        gs = gridspec.GridSpec(2, 2, figure=fig,
                              hspace=0.3, wspace=0.3,
                              left=0.1, right=0.95, top=0.92, bottom=0.1)
        
        # Main plot: final state
        ax_main = fig.add_subplot(gs[:, 0])
        
        # Statistics plots
        ax_swept = fig.add_subplot(gs[0, 1])
        ax_people = fig.add_subplot(gs[1, 1])
        
        # Simulate
        self.env.reset(seed=42, fire_node="RT1")
        
        stats_history = {
            'time': [],
            'swept': [],
            'people_found': [],
            'people_alive': [],
        }
        
        for t in range(num_steps):
            stats = self.env.get_statistics()
            stats_history['time'].append(t)
            stats_history['swept'].append(stats['nodes_swept'])
            stats_history['people_found'].append(stats['people_found'])
            stats_history['people_alive'].append(stats['people_alive'])
            
            # Step
            actions = {}
            for agent_id in self.env.agents.keys():
                valid = self.env.get_valid_actions(agent_id)
                if "search" in valid:
                    actions[agent_id] = "search"
                else:
                    move_actions = [a for a in valid if a.startswith("move_")]
                    actions[agent_id] = move_actions[0] if move_actions else "wait"
            
            self.env.do_action(actions)
        
        # Plot final state
        self.plot_snapshot(ax_main, title=f'Final State (t={num_steps}s)',
                         show_legend=True, show_scale=True)
        
        # Plot swept rooms over time
        ax_swept.plot(stats_history['time'], stats_history['swept'],
                     linewidth=self.config['line_width_medium'],
                     color=self.config['color_swept'], marker='o', markersize=3)
        ax_swept.set_xlabel('Time (s)', fontsize=self.config['font_size_label'])
        ax_swept.set_ylabel('Rooms Swept', fontsize=self.config['font_size_label'])
        ax_swept.set_title('Coverage Progress', fontsize=self.config['font_size_title'])
        ax_swept.grid(True, alpha=0.3, linewidth=self.config['line_width_thin'])
        ax_swept.tick_params(labelsize=self.config['font_size_tick'])
        
        # Plot people found vs alive
        ax_people.plot(stats_history['time'], stats_history['people_found'],
                      linewidth=self.config['line_width_medium'],
                      color=self.config['color_person'], label='Found', marker='s', markersize=3)
        ax_people.plot(stats_history['time'], stats_history['people_alive'],
                      linewidth=self.config['line_width_medium'],
                      color='red', label='Alive', marker='^', markersize=3)
        ax_people.set_xlabel('Time (s)', fontsize=self.config['font_size_label'])
        ax_people.set_ylabel('Number of People', fontsize=self.config['font_size_label'])
        ax_people.set_title('Rescue Statistics', fontsize=self.config['font_size_title'])
        ax_people.legend(fontsize=self.config['font_size_legend'], loc='best')
        ax_people.grid(True, alpha=0.3, linewidth=self.config['line_width_thin'])
        ax_people.tick_params(labelsize=self.config['font_size_tick'])
        
        fig.suptitle('Coverage Progression Analysis',
                    fontsize=self.config['font_size_title'] + 2, fontweight='bold')
        
        plt.savefig(output_path, format=format, dpi=self.config['dpi_save'],
                   bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Saved coverage progression figure: {output_path}")
        plt.close()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-ready visualizations for fire evacuation simulation'
    )
    parser.add_argument('--mode', type=str, required=True,
                       choices=['temporal', 'heatmap', 'coverage', 'all'],
                       help='Visualization mode')
    parser.add_argument('--output', type=str, default='figure',
                       help='Output file name (without extension)')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'svg'],
                       help='Output format')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Build environment
    print("Building environment...")
    env = build_standard_office_layout()
    viz = FireEvacVisualizer(env)
    
    # Generate figures
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    if args.mode in ['temporal', 'all']:
        snapshots = [0, 5, 10, 15, 20, 25]
        output_path = output_dir / f"{args.output}_temporal.{args.format}"
        print(f"\nGenerating temporal evolution figure...")
        viz.plot_temporal_evolution(snapshots, str(output_path), format=args.format)
    
    if args.mode in ['heatmap', 'all']:
        output_path = output_dir / f"{args.output}_heatmap.{args.format}"
        print(f"\nGenerating heatmap figure...")
        viz.plot_heatmap_with_trajectories(args.steps, str(output_path), format=args.format)
    
    if args.mode in ['coverage', 'all']:
        output_path = output_dir / f"{args.output}_coverage.{args.format}"
        print(f"\nGenerating coverage progression figure...")
        viz.plot_coverage_progression(args.steps, str(output_path), format=args.format)
    
    print(f"\n{'='*60}")
    print("✓ All figures generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()