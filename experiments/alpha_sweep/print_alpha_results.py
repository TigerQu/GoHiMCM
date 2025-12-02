#!/usr/bin/env python3
"""Print summary of alpha sweep experiment results."""

from pathlib import Path

print("\n" + "=" * 80)
print("🔬 ALPHA PARAMETER SWEEP EXPERIMENT - RESULTS SUMMARY")
print("=" * 80)

print("\n📋 EXPERIMENT CONFIGURATION:")
print("   Layout: Babycare (3 floors, 41 nodes)")
print("   Alpha range: 0.05 ~ 2.0 (23 values)")
print("   Seeds per alpha: 10")
print("   Total episodes: 230")

print("\n📊 KEY FINDINGS:")
print("   ✓ Maximum rescue: α = 0.05-0.25 → 35.7 people")
print("   ✓ Fastest clearing: α = 2.0 → 62.8 steps")
print("   ✓ Best efficiency: α = 0.15 → 0.5434 rescued/step")

print("\n📈 GENERATED VISUALIZATIONS:")
output_dir = Path("/Users/hengshao/Desktop/HIMCM/GoHiMCM")
pngs = sorted(output_dir.glob("daycare_alpha_*.png"))

for png in pngs:
    size_kb = png.stat().st_size / 1024
    name = png.name.replace("daycare_alpha_", "").replace(".png", "")
    print(f"   • {name:15s} → {size_kb:6.1f} KB")

print("\n💾 DOCUMENTED RESULTS:")
results_file = output_dir / "ALPHA_SWEEP_RESULTS.md"
if results_file.exists():
    print(f"   ✓ {results_file.name}")
    print("     Full detailed analysis with tables and recommendations")

print("\n🎯 RECOMMENDATION:")
print("   Use α = 0.15 for best overall balance:")
print("   • High rescue efficiency (0.5434 rescued/step)")
print("   • Good rescue count (35.7 people)")
print("   • Reasonable clear time (65.7 steps)")
print("   • Low standard deviation (stable results)")

print("\n🔧 EXTENSIBILITY:")
print("   The experiment framework can easily be extended to:")
print("   • Test different layouts (office, warehouse)")
print("   • Sweep other parameters (beta, gamma, weights)")
print("   • Compare different planning strategies")
print("   • Analyze trade-offs in multi-objective optimization")

print("\n" + "=" * 80 + "\n")
