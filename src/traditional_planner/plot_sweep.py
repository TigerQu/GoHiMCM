# file: src/visualization/plot_sweeps.py

from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        re.compile(r"r[_\-]?(\d+)[_\-]?c[_\-]?(\d+)", re.IGNORECASE),
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
                except Exception:
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
            sub_pos = nx.spring_layout(G.subgraph(unmatched), seed=123)
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


def plot_sweep_with_risk(
    node_positions: Optional[Dict[str, Tuple[float, float]]],
    edges: List[Tuple[str, str]],
    agent_paths: Dict[int, List[str]],
    node_clear_time: Optional[Dict[str, float]] = None,
    risk_at_time: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "Deterministic Sweep Paths",
    save_path: Optional[str] = None,
    show_node_labels: bool = True,
    env: Optional[object] = None,
    show_people: bool = True,
) -> None:
    """
    Plot building floorplan with agent trajectories and optional
    risk/clear-time coloring.
    """

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # If an environment is provided prefer its graph layout (spring_layout)
    if env is not None and hasattr(env, "G"):
        try:
            node_positions = nx.spring_layout(env.G, seed=42, k=2, iterations=50)
        except Exception:
            node_positions = _auto_layout_from_graph(edges, list({nid for e in edges for nid in e}))
    else:
        # If provided positions look ring-like, replace with an automatic
        # layout computed from topology to avoid circular embeddings.
        if node_positions is None or _is_circular_layout(node_positions):
            node_ids = list({nid for e in edges for nid in e})
            # preserve any provided node ids as superset
            if node_positions is not None:
                node_ids = sorted(set(node_ids) | set(node_positions.keys()))
            node_positions = _auto_layout_from_graph(edges, node_ids)

    # --- 1. Draw edges (building skeleton) ---
    for u, v in edges:
        if u not in node_positions or v not in node_positions:
            continue
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        ax.plot([x1, x2], [y1, y2], linewidth=1.0, alpha=0.4)

    # --- 2. Compute node colors: risk or clear_time or default ---
    node_ids = list(node_positions.keys())
    node_x = np.array([node_positions[n][0] for n in node_ids])
    node_y = np.array([node_positions[n][1] for n in node_ids])

    if risk_at_time is not None:
        risk_vals = np.array([risk_at_time.get(n, 0.0) for n in node_ids])
        if risk_vals.max() > 0:
            norm_risk = (
                (risk_vals - risk_vals.min())
                / (risk_vals.max() - risk_vals.min())
            )
        else:
            norm_risk = risk_vals
        colors = cm.get_cmap("Reds")(norm_risk)
        cbar_label = "Risk score"
    elif node_clear_time is not None:
        clear_vals = np.array(
            [node_clear_time.get(n, np.nan) for n in node_ids]
        )
        finite_mask = np.isfinite(clear_vals)
        colors = np.full(
            (len(node_ids), 4), (0.7, 0.7, 0.7, 1.0)
        )
        if finite_mask.any():
            vals = clear_vals[finite_mask]
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            cmap = cm.get_cmap("viridis")
            colors[finite_mask] = cmap(norm_vals)
        cbar_label = "Clear time"
    else:
        colors = cm.get_cmap("Blues")(np.zeros(len(node_ids)))
        cbar_label = None

    # --- 3. Draw nodes ---
    sc = ax.scatter(
        node_x,
        node_y,
        s=80,
        c=colors,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )

    if cbar_label is not None:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    if show_node_labels:
        for n, (x, y) in node_positions.items():
            label = n if len(n) <= 10 else n[:10]
            ax.text(x, y + 0.05, label, fontsize=7, ha="center", va="bottom")

    # Draw people locations (triangles) when env supplies people
    if show_people and env is not None and hasattr(env, "people"):
        people_nodes = set()
        for person in env.people.values():
            node_id = getattr(person, "node_id", None) or (
                person.get("node") if isinstance(person, dict) else getattr(person, "node", None)
            )
            if node_id:
                people_nodes.add(node_id)

        people_pos = [node_positions[n] for n in people_nodes if n in node_positions]
        if people_pos:
            xs, ys = zip(*people_pos)
            ax.scatter(xs, ys, marker="^", c="red", s=120, zorder=6)

    # --- 4. Draw agent trajectories ---
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]

    # Draw agent trajectories with faded arrows and start/end markers
    for idx, (agent_id, path) in enumerate(agent_paths.items()):
        if len(path) < 1:
            continue

        coords = [node_positions[n] for n in path if n in node_positions]
        if len(coords) < 1:
            continue

        color = cm.get_cmap("tab10")(idx % 10)
        # draw short arrows with increasing alpha (older -> fainter)
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            alpha = 0.2 + 0.8 * (i / max(1, len(coords) - 2))
            dx, dy = x1 - x0, y1 - y0
            ax.arrow(x0, y0, dx, dy, head_width=0.03, head_length=0.04, fc=color, ec=color, alpha=alpha, length_includes_head=True, linewidth=1.6)

        # start / end markers
        sx, sy = coords[0]
        ex, ey = coords[-1]
        ax.plot(sx, sy, marker="o", color=color, markersize=8, markeredgecolor="black", zorder=5)
        ax.plot(ex, ey, marker="*", color=color, markersize=12, markeredgecolor="black", zorder=5)

        ax.plot([c[0] for c in coords], [c[1] for c in coords], linestyle=linestyles[idx % len(linestyles)], linewidth=1.4, alpha=0.7, label=f"Agent {agent_id}", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x (m or grid)")
    ax.set_ylabel("y (m or grid)")
    ax.legend(loc="best", fontsize=8)

    margin = 0.5
    ax.set_xlim(node_x.min() - margin, node_x.max() + margin)
    ax.set_ylim(node_y.min() - margin, node_y.max() + margin)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
