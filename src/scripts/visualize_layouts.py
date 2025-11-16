"""Visualize layouts (office, babycare, warehouse) and save PNGs.

Usage:
    python3 src/scripts/visualize_layouts.py

Dependencies:
    pip install networkx matplotlib

The script draws simple 2D layouts by grouping nodes by floor and ordering
nodes within each floor for readability. It colors node types and writes
images to `src/scripts/outputs/`.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import networkx as nx
import matplotlib.pyplot as plt
from Env_sim.layouts import (
    build_standard_office_layout,
    build_babycare_layout,
    build_two_floor_warehouse,
)

NODE_COLOR_MAP = {
    'room': '#87CEFA',        # light sky blue
    'hall': '#D3D3D3',        # light gray
    'exit': '#7CFC00',        # lawn green
    'nursery': '#FFF68F',     # light yellow
    'nurse_room': '#FFA500',  # orange
    'play_area': '#DA70D6',   # orchid
    'kitchen': '#FFC0CB',     # pink
    'cargo_room': '#DEB887',  # burlywood
    'nurse_hall': '#98FB98',  # pale green
    'roof_exit': '#FF4500',   # orange red
    'elevator': '#B0E0E6',    # powder blue (if present)
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def ensure_out_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_positions_by_floor(env):
    """Compute positions for nodes by floor.

    This uses naming conventions produced by the layout builders to create
    meaningful 2D placements instead of a single straight line. Known
    patterns handled:
    - Warehouse halls: `F{f}_H_{r}_{c}` -> grid by (r,c)
    - Warehouse rooms: `F{f}_R_{r}_{c}` -> placed between halls
    - Babycare corridors: `F{f}_C{i}` and related rooms `F{f}_NUR{r}`,
      `F{f}_NURSE`, `F{f}_PLAY`, `F{f}_KITCHEN`, and nurse connectors
      `F{f}_NCON_*` -> arranged with nurse station at center
    - Fallback: deterministic multi-column packing to avoid long lines
    """
    import re

    positions = {}
    floors = {}
    processed_floors = set()  # track which floors have been handled
    for nid, meta in env.nodes.items():
        f = getattr(meta, 'floor', 0)
        floors.setdefault(f, []).append((nid, meta))

    # Layout parameters
    floor_gap = 20.0  # much larger for babycare vertical separation
    hall_x_gap = 8.0  # much larger for better horizontal spacing
    hall_y_gap = 7.0  # much larger for room separation

    # Regex patterns
    wh_hall_re = re.compile(r"^H_(?P<r>\d+)_(?P<c>\d+)$")  # Single floor warehouse: H_{r}_{c}
    wh_room_re = re.compile(r"^R_(?P<r>\d+)_(?P<c>\d+)$")  # Single floor warehouse: R_{r}_{c}
    baby_corridor_re = re.compile(r"^F(?P<f>\d+)_C(?P<i>\d+)$")
    baby_nur_re = re.compile(r"^F(?P<f>\d+)_NUR(?P<r>\d+)$")
    office_hall_re = re.compile(r"^H(?P<i>\d+)$")
    office_room_top_re = re.compile(r"^RT(?P<i>\d+)$")
    office_room_bot_re = re.compile(r"^RB(?P<i>\d+)$")

    # First handle office-style simple layouts (H0-H2, RT0-RT2, RB0-RB2, EXIT_LEFT/RIGHT)
    for f, nodes in floors.items():
        # detect office pattern
        has_office_halls = any(office_hall_re.match(nid) for nid, _ in nodes)
        if has_office_halls:
            # place hallway segments horizontally in the center
            hall_positions = {}
            for nid, meta in nodes:
                m = office_hall_re.match(nid)
                if m:
                    i = int(m.group('i'))
                    x = (i - 1) * hall_x_gap * 2.5  # center around 0, H0 left, H1 center, H2 right
                    y = - (f * floor_gap)
                    positions[nid] = (x, y)
                    hall_positions[i] = x
            
            # place top rooms above hallway - aligned with their corresponding hallway segment
            for nid, meta in nodes:
                m = office_room_top_re.match(nid)
                if m:
                    i = int(m.group('i'))
                    x = hall_positions.get(i, i * hall_x_gap * 2.5)
                    y = - (f * floor_gap) + hall_y_gap * 2.0
                    positions[nid] = (x, y)
            
            # place bottom rooms below hallway - aligned with their corresponding hallway segment
            for nid, meta in nodes:
                m = office_room_bot_re.match(nid)
                if m:
                    i = int(m.group('i'))
                    x = hall_positions.get(i, i * hall_x_gap * 2.5)
                    y = - (f * floor_gap) - hall_y_gap * 2.0
                    positions[nid] = (x, y)
            
            # place exits at the far left and far right, aligned with hallway
            if 'EXIT_LEFT' in env.nodes:
                left_most = min(hall_positions.values()) if hall_positions else 0
                positions['EXIT_LEFT'] = (left_most - hall_x_gap * 2.5, - (f * floor_gap))
            if 'EXIT_RIGHT' in env.nodes:
                right_most = max(hall_positions.values()) if hall_positions else 0
                positions['EXIT_RIGHT'] = (right_most + hall_x_gap * 2.5, - (f * floor_gap))
            processed_floors.add(f)
            continue

    # Handle warehouse-style grids explicitly
    for f, nodes in floors.items():
        if f in processed_floors:
            continue
        # detect if this floor looks like a warehouse grid by checking for any H_ nodes
        has_wh_halls = any(wh_hall_re.match(nid) for nid, _ in nodes)
        if has_wh_halls:
            # collect all halls and compute grid bounds
            halls = []
            for nid, meta in nodes:
                m = wh_hall_re.match(nid)
                if m:
                    r = int(m.group('r'))
                    c = int(m.group('c'))
                    halls.append((nid, r, c))

            if halls:
                max_r = max(r for _, r, _ in halls)
                max_c = max(c for _, _, c in halls)
                # center the grid horizontally
                center_offset_x = -max_c * hall_x_gap / 2
                # place halls at grid positions with much better spacing
                for nid, r, c in halls:
                    x = center_offset_x + c * hall_x_gap * 1.8
                    y = - (f * floor_gap + r * hall_y_gap * 1.8)
                    positions[nid] = (x, y)

            # place cargo rooms (between halls) with matching spacing
            for nid, meta in nodes:
                m = wh_room_re.match(nid)
                if m:
                    r = int(m.group('r'))
                    c = int(m.group('c'))
                    x = center_offset_x + (c + 0.5) * hall_x_gap * 1.8
                    y = - (f * floor_gap + (r + 0.5) * hall_y_gap * 1.8)
                    positions[nid] = (x, y)
            
            # place warehouse exits aligned with the halls they connect to
            if halls:
                max_r = max(r for _, r, _ in halls)
                min_c = min(c for _, _, c in halls)
                max_c = max(c for _, _, c in halls)
                
                if 'EXIT_WH_LEFT' in env.nodes:
                    # place exit to the left of the leftmost column
                    x = center_offset_x + min_c * hall_x_gap * 1.8 - hall_x_gap * 2.5
                    y = - (f * floor_gap + max_r * hall_y_gap * 1.8)  # align with bottom row
                    positions['EXIT_WH_LEFT'] = (x, y)
                
                if 'EXIT_WH_RIGHT' in env.nodes:
                    # place exit to the right of the rightmost column
                    x = center_offset_x + max_c * hall_x_gap * 1.8 + hall_x_gap * 2.5
                    y = - (f * floor_gap + max_r * hall_y_gap * 1.8)  # align with bottom row
                    positions['EXIT_WH_RIGHT'] = (x, y)
            
            processed_floors.add(f)
            continue

        # Handle babycare floor heuristics
        # Find any corridor nodes like F{f}_C0..C2
        corridors = { }
        for nid, meta in nodes:
            m = baby_corridor_re.match(nid)
            if m and int(m.group('f')) == f:
                i = int(m.group('i'))
                corridors[i] = nid

        if corridors:
            # order corridor indices
            idxs = sorted(corridors.keys())
            center_x = 0.0
            # place corridors horizontally - main central axis with generous spacing
            for pos_i, i in enumerate(idxs):
                nid = corridors[i]
                x = center_x + (pos_i - len(idxs)/2 + 0.5) * hall_x_gap * 6
                y = - (f * floor_gap)
                positions[nid] = (x, y)

            # place nurse station at center top - key focal point
            nurse_id = f"F{f}_NURSE"
            if nurse_id in env.nodes:
                positions[nurse_id] = (center_x, - (f * floor_gap) + hall_y_gap * 3.5)

            # place nurseries in a row above corridors, aligned and well-spaced
            for nid, meta in nodes:
                m = baby_nur_re.match(nid)
                if m and int(m.group('f')) == f:
                    r = int(m.group('r'))
                    cor = corridors.get(r)
                    if cor and cor in positions:
                        cx, cy = positions[cor]
                        # align nurseries above their corresponding corridors
                        positions[nid] = (cx, cy + hall_y_gap * 2.0)

            # play area: place to the far right
            play = f"F{f}_PLAY"
            kit = f"F{f}_KITCHEN"
            if play in env.nodes:
                last_c = corridors.get(idxs[-1])
                if last_c and last_c in positions:
                    cx, cy = positions[last_c]
                    positions[play] = (cx + hall_x_gap * 5, cy)
            
            # kitchen: place to the far left
            if kit in env.nodes:
                first_c = corridors.get(idxs[0])
                if first_c and first_c in positions:
                    cx, cy = positions[first_c]
                    positions[kit] = (cx - hall_x_gap * 5, cy)

            # nurse connector halls: place strategically between rooms and nurse station
            for nid, meta in nodes:
                if nid.startswith(f"F{f}_NCON_"):
                    target = None
                    if "NUR" in nid and "_NUR" in nid:
                        target = nid.replace("NCON_", "")
                    elif "PLAY" in nid:
                        target = f"F{f}_PLAY"
                    elif "KIT" in nid:
                        target = f"F{f}_KITCHEN"
                    if target and target in positions and nurse_id in positions:
                        tx, ty = positions[target]
                        nxp, nyp = positions[nurse_id]
                        # place connector closer to the room (2/3 of the way from nurse to room)
                        positions[nid] = ((tx * 2 + nxp) / 3.0, (ty * 2 + nyp) / 3.0)
            processed_floors.add(f)
            continue

        # Fallback: deterministic multi-column packing to avoid long lines
        # place nodes in a grid row by row
        if f not in processed_floors:
            sorted_nodes = sorted([nid for nid, _ in nodes])
            cols = min(8, max(3, int(len(sorted_nodes) ** 0.5)))
            for idx, nid in enumerate(sorted_nodes):
                row = idx // cols
                col = idx % cols
                x = (col - cols / 2) * hall_x_gap * 1.5
                y = - (f * floor_gap + row * hall_y_gap)
                positions[nid] = (x, y)
            processed_floors.add(f)

    # Ensure every node has a position: place any remaining nodes sensibly
    for f, nodes in floors.items():
        # collect existing x coordinates for this floor
        xs = [positions[nid][0] for nid, _ in nodes if nid in positions]
        if xs:
            min_x, max_x = min(xs), max(xs)
        else:
            min_x, max_x = -4.0, 4.0

        # place unpositioned nodes: exits at extremes, others near center
        exit_left = min_x - hall_x_gap
        exit_right = max_x + hall_x_gap
        center_x = (min_x + max_x) / 2.0
        placed_exits = 0

        for nid, meta in nodes:
            if nid in positions:
                continue
            ntype = getattr(meta, 'ntype', '')
            if ntype == 'exit' or nid.lower().startswith('exit'):
                # alternate placing exits left/right
                if placed_exits % 2 == 0:
                    positions[nid] = (exit_left, - (f * floor_gap) - 0.5)
                else:
                    positions[nid] = (exit_right, - (f * floor_gap) - 0.5)
                placed_exits += 1
            else:
                # place near center, stacking vertically if many
                # count how many already placed near center
                similar = [p for p in positions.values() if abs(p[0] - center_x) < 1.0]
                offset_row = len(similar)
                positions[nid] = (center_x + 0.5 * offset_row, - (f * floor_gap) - 1.0 - offset_row * 0.6)

    return positions


def draw_env(env, out_path, title=None):
    G = env.G
    pos = compute_positions_by_floor(env)

    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    ax.set_title(title or out_path)
    ax.set_facecolor('#f8f8f8')

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    labels = {}
    for nid, meta in env.nodes.items():
        node_type = getattr(meta, 'ntype', 'room')
        node_colors.append(NODE_COLOR_MAP.get(node_type, '#CCCCCC'))
        # Size by area if available (larger baseline for clarity)
        area = getattr(meta, 'area', 10.0)
        node_sizes.append(max(300, min(2500, area * 20)))
        labels[nid] = nid

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.3)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='#444444', linewidths=1.0)
    # Optional: draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=6)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ensure_out_dir()

    # Office
    office = build_standard_office_layout()
    office_path = os.path.join(OUTPUT_DIR, 'layout_office.png')
    draw_env(office, office_path, title='Standard Office Layout')
    print('Saved', office_path)

    # Babycare
    baby = build_babycare_layout()
    baby_path = os.path.join(OUTPUT_DIR, 'layout_babycare.png')
    draw_env(baby, baby_path, title='Babycare 5-floor Layout')
    print('Saved', baby_path)

    # Warehouse
    wh = build_two_floor_warehouse()
    wh_path = os.path.join(OUTPUT_DIR, 'layout_warehouse.png')
    draw_env(wh, wh_path, title='Large Warehouse Layout')
    print('Saved', wh_path)


if __name__ == '__main__':
    main()
