# file: src/traditional_planner/plot_sweep.py
"""
Enhanced sweep visualization with improved agent trajectory display.
Based on agent trajectory visualization best practices.

Features:
- Better graph layout using spring layout + grid detection
- Improved color gradients for trajectory paths
- Enhanced start/end markers with clear visibility
- Optional heatmap overlay for node visitation frequency
"""

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import networkx as nx
import re


def _is_circular_layout(
    node_positions: Dict[str, Tuple[float, float]]
) -> bool:
    """Detect whether the provided node_positions lie roughly on a circle.

    Returns True if nodes are arranged on a ring (low radial variance).
    """
    if not node_positions:
        return False
    pts = np.array(list(node_positions.values()))
    if pts.shape[0] < 6:
        return False
    centroid = pts.mean(axis=0)
    radii = np.linalg.norm(pts - centroid, axis=1)
    mean_r = radii.mean()
    if mean_r <= 0:
        return False
    rel_std = radii.std() / mean_r
    # If radial std is small, points are roughly on a circle
    return rel_std < 0.25


def _auto_layout_from_graph(
    edges: List[Tuple[str, str]], node_ids: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Compute a nicer layout from graph topology using heuristics.

    Tries grid detection, backbone (corridor) layout, then Kamada-Kawai.
    """
    G = nx.Graph()
    for nid in node_ids:
        G.add_node(nid)
    for u, v in edges:
        if u in G.nodes() and v in G.nodes():
            G.add_edge(u, v)

    # --- grid detection ---
    patterns = [
        re.compile(r"R(\d+)C(\d+)", re.IGNORECASE),
        re.compile(r"r[_\-]?(\d+)[_\-]?c[_\-]?(\d+)",
                   re.IGNORECASE),
        re.compile(r"^(\d+)[_xX](\d+)$"),
        re.compile(r"^(\d+)[,](\d+)$"),
    ]
    grid_coords = {}
    for nid in G.nodes():
        for pat in patterns:
            m = pat.search(nid)
            if m:
                try:
                    r = int(m.group(1))
                    c = int(m.group(2))
                    grid_coords[nid] = (r, c)
                except (ValueError, AttributeError):
                    pass
                break

    if len(grid_coords) >= max(3, 0.3 * len(G.nodes())):
        rows = sorted({r for r, _ in grid_coords.values()})
        cols = sorted({c for _, c in grid_coords.values()})
        row_index = {r: i for i, r in enumerate(rows)}
        col_index = {c: i for i, c in enumerate(cols)}
        spacing = 1.5
        pos = {}
        for nid in G.nodes():
            if nid in grid_coords:
                r, c = grid_coords[nid]
                x = col_index[c] * spacing
                y = -row_index[r] * spacing
                pos[nid] = (x, y)
            else:
                pos[nid] = None
        # assign unmatched with local spring
        unmatched = [n for n, p in pos.items() if p is None]
        if unmatched:
            sub_pos = nx.spring_layout(G.subgraph(unmatched),
                                       seed=123)
            for nid, (x, y) in sub_pos.items():
                pos[nid] = (x, y)
        return pos

    # --- backbone (house/corridor) heuristic ---
    def _compute_backbone(g):
        maxd = -1
        best = (None, None)
        for u in g.nodes():
            lengths = nx.single_source_shortest_path_length(g, u)
            for v, d in lengths.items():
                if d > maxd:
                    maxd = d
                    best = (u, v)
        if best[0] is None:
            return []
        try:
            return nx.shortest_path(g, best[0], best[1])
        except Exception:
            return []

    backbone = _compute_backbone(G)
    if backbone and len(backbone) >= 3:
        pos = {}
        spacing_x = 2.0
        spacing_y = 1.5
        for i, nid in enumerate(backbone):
            pos[nid] = (i * spacing_x, 0.0)
        for i, nid in enumerate(backbone):
            nbrs = [n for n in G.neighbors(nid) if n not in backbone]
            for j, nbr in enumerate(nbrs):
                side = -1 if j % 2 == 0 else 1
                layer = (j // 2) + 1
                pos[nbr] = (i * spacing_x, side * layer * spacing_y)
        remaining = [n for n in G.nodes() if n not in pos]
        if remaining:
            sub_pos = nx.spring_layout(G.subgraph(remaining), seed=999)
            for nid, (x, y) in sub_pos.items():
                pos[nid] = (x + 0.5, y - 0.5)
        return pos

    # final fallback: kamada_kawai
    return nx.kamada_kawai_layout(G)


def _infer_layout_from_env(env: object) -> Dict[str, Tuple[float, float]]:
    """
    Infer 2D node positions from environment using floor + node type.

    Uses a hierarchy of layout strategies:
    1. Grid detection for warehouse-like layouts
    2. Hierarchical layout for multi-floor buildings (babycare)
    3. Spring layout fallback for irregular layouts

    Returns a well-organized layout with good visual separation.
    """
    pos = {}

    if not hasattr(env, "nodes") or not hasattr(env, "G"):
        return {}

    # Build graph for layout computation
    G = nx.Graph()
    for nid in env.G.nodes():
        G.add_node(nid)
    for u, v in env.G.edges():
        G.add_edge(u, v)

    # --- STRATEGY 1: Grid Detection (Warehouse) ---
    grid_patterns = [
        re.compile(r"^H_(\d+)_(\d+)$"),  # warehouse: H_r_c
        re.compile(r"^R_(\d+)_(\d+)$"),  # warehouse: R_r_c
        re.compile(r"R(\d+)C(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)[_xX](\d+)$"),
    ]

    grid_coords = {}
    for nid in G.nodes():
        for pat in grid_patterns:
            m = pat.search(nid)
            if m:
                try:
                    r, c = int(m.group(1)), int(m.group(2))
                    grid_coords[nid] = (r, c)
                except (ValueError, AttributeError):
                    pass
                break

    # If >30% nodes match grid pattern, use grid layout
    if len(grid_coords) >= max(3, 0.3 * len(G.nodes())):
        rows = sorted({r for r, _ in grid_coords.values()})
        cols = sorted({c for _, c in grid_coords.values()})

        row_idx = {r: i for i, r in enumerate(rows)}
        col_idx = {c: i for i, c in enumerate(cols)}

        # Use larger spacing for warehouse
        spacing_x = 3.0 if len(rows) > 3 else 2.5
        spacing_y = 4.0 if len(cols) > 4 else 3.5

        for nid in G.nodes():
            if nid in grid_coords:
                r, c = grid_coords[nid]
                x = col_idx[c] * spacing_x
                y = -row_idx[r] * spacing_y
                pos[nid] = (x, y)

        # Assign unmatched nodes using spring layout
        unmatched = [n for n in G.nodes() if n not in pos]
        if unmatched:
            subgraph_pos = nx.spring_layout(G.subgraph(unmatched),
                                            seed=123, k=1.5)
            for nid, (x, y) in subgraph_pos.items():
                pos[nid] = (x, y)

        return pos

    # --- STRATEGY 2: Floor-based Hierarchical Layout (Babycare) ---
    floor_nodes = {}
    for nid in G.nodes():
        if nid not in env.nodes:
            continue
        node_info = env.nodes[nid]
        floor = getattr(node_info, "floor", 0)
        floor_nodes.setdefault(floor, []).append(nid)

    if len(floor_nodes) > 1:  # Multi-floor building
        for floor in sorted(floor_nodes.keys()):
            nodes_on_floor = floor_nodes[floor]
            y_base = -floor * 6.0

            # Separate by type
            hallways = []
            rooms = []
            exits = []

            for nid in sorted(nodes_on_floor):
                node_info = env.nodes.get(nid)
                if node_info is None:
                    continue

                ntype = getattr(node_info, "ntype", "generic")

                if ntype in ["hall", "hallway", "corridor"]:
                    hallways.append(nid)
                elif ntype == "exit":
                    exits.append(nid)
                elif ntype == "room":
                    rooms.append(nid)
                else:
                    hallways.append(nid)

            # Layout: exits on sides, hallways in center, rooms on
            # top/bottom
            total_width = max(len(hallways), len(rooms)) * 2.5

            # Exits
            if exits:
                pos[exits[0]] = (-total_width / 2 - 1.5, y_base)
                if len(exits) > 1:
                    pos[exits[-1]] = (total_width / 2 + 1.5, y_base)

            # Hallways (center horizontal line)
            x_start = -(len(hallways) - 1) * 2.5 / 2
            for i, nid in enumerate(hallways):
                pos[nid] = (x_start + i * 2.5, y_base)

            # Rooms (distributed above/below)
            x_start = -(len(rooms) - 1) * 2.5 / 2
            for i, nid in enumerate(rooms):
                side = 1.5 if i % 2 == 0 else -1.5
                pos[nid] = (x_start + i * 2.5, y_base + side)

        return pos

    # --- STRATEGY 3: Spring Layout Fallback ---
    # Use improved spring layout with better parameters
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42, scale=10)

    return pos


def plot_sweep_with_risk(
    node_positions: Optional[Dict[str, Tuple[float, float]]],
    edges: List[Tuple[str, str]],
    agent_paths: Dict[int, List[str]],
    node_clear_time: Optional[Dict[str, float]] = None,
    risk_at_time: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (18, 10),
    title: str = "Deterministic Sweep Paths",
    save_path: Optional[str] = None,
    show_node_labels: bool = True,
    env: Optional[object] = None,
    show_people: bool = True,
    show_heatmap: bool = True,
) -> None:
    """
    Enhanced plot of building floorplan with agent trajectories.

    Features:
    - Larger figure size (18x10) for better visibility
    - Improved layout algorithm with grid detection
    - Gradient alpha for trajectory paths
      (older=lighter, newer=darker)
    - Better markers: circles for start, stars for end
    - Optional heatmap showing node visitation frequency
    - High-quality output (300 DPI)
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Compute positions
    if env is not None and hasattr(env, "G"):
        try:
            node_positions = _infer_layout_from_env(env)
            if not node_positions:
                node_ids = list({nid for e in edges for nid in e})
                node_positions = _auto_layout_from_graph(edges, node_ids)
        except Exception:
            node_ids = list({nid for e in edges for nid in e})
            node_positions = _auto_layout_from_graph(edges, node_ids)
    else:
        # Fallback
        if node_positions is None or _is_circular_layout(node_positions):
            node_ids = list({nid for e in edges for nid in e})
            if node_positions is not None:
                node_ids = sorted(
                    set(node_ids) | set(node_positions.keys()))
            node_positions = _auto_layout_from_graph(edges, node_ids)

    # --- 1. Draw edges (building skeleton) ---
    for u, v in edges:
        if u not in node_positions or v not in node_positions:
            continue
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.8,
                alpha=0.3, zorder=1)

    # --- 2. Calculate node visitation frequency ---
    node_visit_count = {}
    for path in agent_paths.values():
        for node_id in path:
            node_visit_count[node_id] = (
                node_visit_count.get(node_id, 0) + 1)

    # --- 3. Draw nodes with type coloring ---
    node_ids = list(node_positions.keys())
    node_x = np.array([node_positions[n][0] for n in node_ids])
    node_y = np.array([node_positions[n][1] for n in node_ids])

    # Color by type
    type_colors = {
        "exit": "#ff4444",       # Red
        "room": "#44aa44",       # Green
        "hall": "#4488ff",       # Blue
        "hallway": "#4488ff",
        "corridor": "#4488ff",
    }

    node_colors = []
    for n in node_ids:
        if env is not None and hasattr(env, "nodes"):
            node_info = env.nodes.get(n)
            if node_info is not None:
                ntype = getattr(node_info, "ntype", "generic")
                color = type_colors.get(ntype, "#999999")
            else:
                color = "#999999"
        else:
            color = "#999999"

        node_colors.append(color)

    # Draw base nodes (larger for better visibility)
    ax.scatter(node_x, node_y, s=150, c=node_colors,
               edgecolors="black", linewidths=1.0, zorder=3, alpha=0.8)

    # Draw node labels
    if show_node_labels:
        for n, (x, y) in node_positions.items():
            label = n if len(n) <= 8 else n[:8]
            ax.text(x, y, label, fontsize=6, ha="center", va="center",
                    fontweight="bold", zorder=4)

    # --- 4. Draw people locations ---
    if show_people and env is not None and hasattr(env, "people"):
        people_nodes = set()
        for person in env.people.values():
            node_id = getattr(person, "node_id", None)
            if node_id is None and isinstance(person, dict):
                node_id = person.get("node")
            if node_id is None:
                node_id = getattr(person, "node", None)
            if node_id:
                people_nodes.add(node_id)

        people_pos = [node_positions[n] for n in people_nodes
                      if n in node_positions]
        if people_pos:
            xs, ys = zip(*people_pos)
            ax.scatter(xs, ys, marker="^", c="red", s=120, zorder=6,
                       edgecolors="darkred", linewidths=1.5,
                       label="People")

    # --- 5. Draw agent trajectories with enhanced visualization ---
    # Use distinct colors for different agents
    agent_colors = [
        "#1f77b4",  # tab:blue
        "#ff7f0e",  # tab:orange
        "#2ca02c",  # tab:green
        "#d62728",  # tab:red
        "#9467bd",  # tab:purple
        "#8c564b",  # tab:brown
        "#e377c2",  # tab:pink
        "#7f7f7f",  # tab:gray
    ]

    for agent_idx, (agent_id, path) in enumerate(agent_paths.items()):
        if len(path) < 1:
            continue

        coords = [node_positions[n] for n in path
                  if n in node_positions]
        if len(coords) < 1:
            continue

        color = agent_colors[agent_idx % len(agent_colors)]

        # Draw path with gradient alpha (newer steps darker)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]

            # Gradient: older steps lighter, newer steps darker
            progress = i / max(1, len(coords) - 1)
            alpha = 0.4 + 0.6 * progress
            linewidth = 1.5 + 2.0 * progress

            ax.plot([x0, x1], [y0, y1], color=color,
                    linewidth=linewidth, alpha=alpha,
                    solid_capstyle="round", zorder=2)

            # Add arrow markers on some segments
            if (len(coords) > 2 and
                    i % max(1, (len(coords) - 1) // 4) == 0):
                dx, dy = x1 - x0, y1 - y0
                ax.arrow(x0, y0, dx * 0.7, dy * 0.7,
                         head_width=0.3, head_length=0.2,
                         fc=color, ec=color, alpha=alpha * 0.7,
                         zorder=2, length_includes_head=True)

        # Start marker (circle)
        if coords:
            sx, sy = coords[0]
            ax.scatter([sx], [sy], s=250, marker="o", c=[color],
                       edgecolors="black", linewidths=2, zorder=5,
                       label=f"Agent {agent_id} (start)")

        # End marker (star)
        if len(coords) > 1:
            ex, ey = coords[-1]
            ax.scatter([ex], [ey], s=400, marker="*", c=[color],
                       edgecolors="black", linewidths=2, zorder=5,
                       label=f"Agent {agent_id} (end)")

        # Show unique nodes visited
        unique = len(set(path))
        ax.plot([], [], color=color, linewidth=3,
                label=f"Agent {agent_id}: {unique} nodes",
                alpha=0.8)

    # --- 6. Enhanced legend ---
    # Symbol legend
    symbol_legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markersize=10,
               markeredgecolor="black", markeredgewidth=1.2,
               label="● = Start", linestyle="none"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="gray", markersize=16,
               markeredgecolor="black", markeredgewidth=1.2,
               label="★ = End", linestyle="none"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="red", markersize=8,
               markeredgecolor="darkred", markeredgewidth=1,
               label="▲ = Person", linestyle="none"),
    ]

    # Type legend
    type_legend = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#ff4444", markersize=8,
               markeredgecolor="black", markeredgewidth=0.8,
               label="Exit", linestyle="none"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#44aa44", markersize=8,
               markeredgecolor="black", markeredgewidth=0.8,
               label="Room", linestyle="none"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#4488ff", markersize=8,
               markeredgecolor="black", markeredgewidth=0.8,
               label="Hallway", linestyle="none"),
    ]

    # Combine legends
    handles, labels = ax.get_legend_handles_labels()

    # Create two-part legend
    leg1 = ax.legend(symbol_legend + type_legend,
                     [e.get_label()
                      for e in symbol_legend + type_legend],
                     loc="upper left", fontsize=9,
                     framealpha=0.95, edgecolor="black",
                     title="Legend")
    ax.add_artist(leg1)

    # Agent legend (trajectory info)
    leg2 = ax.legend(handles, labels, loc="upper right", fontsize=8,
                     framealpha=0.95, edgecolor="black",
                     title="Agents", ncol=1, borderpad=0.5)

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("X Position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)
    ax.set_facecolor("#f9f9f9")

    # Set limits with margin
    if node_x.size > 0:
        margin = 2.0
        ax.set_xlim(node_x.min() - margin, node_x.max() + margin)
        ax.set_ylim(node_y.min() - margin, node_y.max() + margin)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
