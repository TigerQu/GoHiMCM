from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_sweep_with_risk(
    node_positions: Optional[Dict[str, Tuple[float, float]]],
    edges: List[Tuple[str, str]],
    agent_paths: Dict[int, List[str]],
    node_clear_time: Dict[str, float],
    risk_at_time: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    save_path: str = "sweep.png",
) -> None:
    """
    Simple visualization helper for sweep runs.

    Parameters
    - node_positions: mapping node_id -> (x,y)
    - edges: list of (u,v) pairs
    - agent_paths: dict agent_id -> list of node_ids visited (in time order)
    - node_clear_time: dict node_id -> time when first cleared
    - risk_at_time: optional dict node_id -> risk value (ignored for now)
    - title: plot title
    - save_path: output PNG path

    The function is intentionally lightweight and has no external deps
    beyond matplotlib and networkx.
    """
    # Build the graph and determine positions. If node_positions is None or
    # incomplete, try to produce a sensible layout automatically.
    G = nx.Graph()
    all_node_ids = set()
    for u, v in edges:
        all_node_ids.add(u)
        all_node_ids.add(v)
    if node_positions:
        for nid in node_positions.keys():
            all_node_ids.add(nid)

    if not all_node_ids:
        raise ValueError("No nodes found to plot")

    for nid in sorted(all_node_ids):
        G.add_node(nid)
    for u, v in edges:
        if u in G.nodes() and v in G.nodes():
            G.add_edge(u, v)

    # Decide positions: prefer provided node_positions, otherwise compute.
    pos: Dict[str, Tuple[float, float]]
    if node_positions and set(node_positions.keys()) >= set(G.nodes()):
        pos = node_positions
    else:
        # Heuristic: detect floor-prefixed IDs like 'F0_' and layout floors
        # as horizontal layers stacked vertically. This often matches multi-
        # floor building naming conventions used in layouts (F0_...)
        import re

        floor_groups = {}
        for nid in G.nodes():
            m = re.match(r"^F(\d+)_", nid)
            if m:
                f = int(m.group(1))
                floor_groups.setdefault(f, []).append(nid)

        if floor_groups:
            # Layout each floor using spring_layout then stack
            pos = {}
            ys = {}
            floors = sorted(floor_groups.keys())
            for i, f in enumerate(floors):
                sub_nodes = floor_groups[f]
                subG = G.subgraph(sub_nodes)
                sub_pos = nx.spring_layout(subG, seed=100 + f, k=0.5)
                # vertical offset: higher floor -> higher y
                y_offset = float(i) * 5.0
                for nid, (x, y) in sub_pos.items():
                    pos[nid] = (x, y + y_offset)
        else:
            # If no floor grouping, try to detect a grid-like naming scheme
            # commonly used in warehouse layouts. Patterns handled include:
            #   R{row}C{col}, r{row}_c{col}, r-row-c-col, or plain "{row}_{col}".
            import re as _re

            grid_coords = {}
            patterns = [
                _re.compile(r"R(\d+)C(\d+)", _re.IGNORECASE),
                _re.compile(r"r[_\-]?(\d+)[_\-]?c[_\-]?(\d+)", _re.IGNORECASE),
                _re.compile(r"^(\d+)[_xX](\d+)$"),
                _re.compile(r"^(\d+)[,](\d+)$"),
            ]

            for nid in G.nodes():
                for pat in patterns:
                    m = pat.search(nid)
                    if m:
                        try:
                            r = int(m.group(1))
                            c = int(m.group(2))
                            grid_coords[nid] = (r, c)
                        except Exception:
                            continue
                        break

            # If a majority of nodes match a grid pattern, lay them out on a grid.
            # Lower the detection threshold so partially-encoded grids
            # (common in some warehouse names) are still recognized.
            if len(grid_coords) >= max(3, 0.3 * len(G.nodes())):
                # normalize rows/cols to start at 0
                rows = sorted({r for r, _ in grid_coords.values()})
                cols = sorted({c for _, c in grid_coords.values()})
                row_index = {r: i for i, r in enumerate(rows)}
                col_index = {c: i for i, c in enumerate(cols)}
                pos = {}
                spacing = 1.5
                for nid in G.nodes():
                    if nid in grid_coords:
                        r, c = grid_coords[nid]
                        x = col_index[c] * spacing
                        y = -row_index[r] * spacing
                        pos[nid] = (x, y)
                    else:
                        # fallback: place unmatched nodes using spring layout
                        # later
                        pos[nid] = None
                # For nodes with None positions, compute local spring positions
                unmatched = [n for n, p in pos.items() if p is None]
                if unmatched:
                    sub_pos = nx.spring_layout(G.subgraph(unmatched), seed=123)
                    for nid, (x, y) in sub_pos.items():
                        pos[nid] = (x, y)
            else:
                # Try a house-like backbone layout: find a long path in the
                # graph (the backbone/corridor) and arrange rooms along it.
                def _compute_backbone(g):
                    # Find pair of nodes with max shortest-path distance
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
                    return nx.shortest_path(g, best[0], best[1])

                backbone = _compute_backbone(G)
                if backbone and len(backbone) >= 3:
                    pos = {}
                    spacing_x = 2.0
                    spacing_y = 1.5
                    # place backbone along x axis
                    for i, nid in enumerate(backbone):
                        pos[nid] = (i * spacing_x, 0.0)
                    # attach neighbors perpendicularly
                    for i, nid in enumerate(backbone):
                        nbrs = [
                            n for n in G.neighbors(nid)
                            if n not in backbone
                        ]
                        for j, nbr in enumerate(nbrs):
                            side = -1 if j % 2 == 0 else 1
                            layer = (j // 2) + 1
                            pos[nbr] = (
                                i * spacing_x,
                                side * layer * spacing_y,
                            )
                    # remaining nodes -> spring layout localized
                    remaining = [n for n in G.nodes() if n not in pos]
                    if remaining:
                        sub_pos = nx.spring_layout(
                            G.subgraph(remaining), seed=999
                        )
                        # shift remaining positions to avoid overlaps
                        for nid, (x, y) in sub_pos.items():
                            pos[nid] = (x + 0.5, y - 0.5)
                    else:
                        # Fall back to Kamada-Kawai layout for readability
                        pos = nx.kamada_kawai_layout(G)

    # Color nodes by clear time (earlier=blue, later=yellow).
    # Unseen nodes -> light gray
    all_times = [t for t in node_clear_time.values()]
    max_t = max(all_times) if all_times else 1.0

    node_colors = []
    for nid in G.nodes():
        if nid in node_clear_time:
            # normalize to [0,1]
            val = node_clear_time[nid] / max_t if max_t > 0 else 0.0
            node_colors.append(val)
        else:
            node_colors.append(None)

    cmap = plt.cm.viridis
    node_color_mapped = [
        cmap(c) if c is not None else (0.8, 0.8, 0.8, 1.0)
        for c in node_colors
    ]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#999999", width=1.0)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=300,
        node_color=node_color_mapped,
        linewidths=0.5,
        edgecolors="#333333",
        ax=ax,
    )

    # Draw labels (shortened)
    labels = {nid: nid for nid in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # Draw agent paths with distinct colors and arrows
    colors = plt.cm.tab10
    for aid, path in agent_paths.items():
        coords = [pos[nid] for nid in path if nid in pos]
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        col = colors(aid % 10)
        ax.plot(xs, ys, color=col, linewidth=2.0, alpha=0.9, zorder=3)
        # draw arrow heads and time indices
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
            )
            # small time label near start point
            ax.text(x0, y0, f"{i}", color=col, fontsize=6, alpha=0.7)

        # last point label
        lx, ly = coords[-1]
        ax.text(lx, ly, f"A{aid}", color=col, fontsize=9, fontweight="bold")

    # Colorbar for clear time
    if any(c is not None for c in node_colors):
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_t)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("clear time (timesteps)")

    ax.set_title(title or "Sweep visualization")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
