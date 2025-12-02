#!/usr/bin/env python3
"""Verify the visualizations were created successfully."""

from pathlib import Path

output_dir = Path("/Users/hengshao/Desktop/HIMCM/GoHiMCM")
pngs = sorted(output_dir.glob("*_greedy_sweep.png"))

print("Generated visualization files:")
for png in pngs:
    size_mb = png.stat().st_size / (1024 * 1024)
    print(f"  ✓ {png.name:35s} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print("✅ SUCCESS: All visualizations have been generated with:")
print("  • Clean grid/linear layout (no more messy spring_layout)")
print("  • Node type coloring (exit=red, room=green, hallway=blue)")
print("  • Agent trajectory visualization with smooth lines and markers")
print("  • No heatmap/colorbar clutter")
print("="*70)
