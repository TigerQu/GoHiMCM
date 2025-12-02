#!/usr/bin/env python3
"""Quick test to check the new layout algorithm."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.layouts import (
    build_standard_office_layout,
    build_babycare_layout,
)
from traditional_planner.plot_sweep import _infer_layout_from_env

# Build office layout
env = build_standard_office_layout()
pos = _infer_layout_from_env(env)

print("Office layout nodes and positions:")
for nid in sorted(pos.keys()):
    x, y = pos[nid]
    ntype = (
        env.nodes[nid].ntype if nid in env.nodes else "unknown"
    )
    floor = env.nodes[nid].floor if nid in env.nodes else 0
    print(
        f"  {nid:15s} | type={ntype:8s} | floor={floor} | "
        f"pos=({x:6.2f}, {y:6.2f})"
    )

print("\n" + "="*60)
print("Babycare layout nodes and positions (first 20):")
env2 = build_babycare_layout()
pos2 = _infer_layout_from_env(env2)
for i, nid in enumerate(sorted(pos2.keys())[:20]):
    x, y = pos2[nid]
    ntype = (
        env2.nodes[nid].ntype if nid in env2.nodes else "unknown"
    )
    floor = env2.nodes[nid].floor if nid in env2.nodes else 0
    print(
        f"  {nid:15s} | type={ntype:8s} | floor={floor} | "
        f"pos=({x:6.2f}, {y:6.2f})"
    )
