"""Analyze babycare layout structure"""
import sys
sys.path.insert(0, "/Users/qzy/Projects/GoHiMCM")

from src.Env_sim.layouts import build_babycare_layout

env = build_babycare_layout()

print("=" * 70)
print("BABYCARE LAYOUT ANALYSIS")
print("=" * 70)

for f in range(3):
    nodes = [(n, env.nodes[n]) for n in env.nodes.keys() if env.nodes[n].floor == f]
    halls = [n for n, meta in nodes if meta.ntype == 'hall']
    rooms = [n for n, meta in nodes if meta.ntype == 'room']
    
    print(f"\nFloor {f}: {len(nodes)} nodes total")
    print(f"  Corridors (C0-C2): {[n for n in halls if '_C' in n]}")
    print(f"  Nurse connectors: {[n for n in halls if 'NCON' in n]}")
    print(f"  Nurseries: {[n for n in rooms if 'NUR' in n]}")
    print(f"  Nurse station: {[n for n in rooms if n == f'F{f}_NURSE']}")
    print(f"  Play area: {[n for n in rooms if 'PLAY' in n]}")
    print(f"  Kitchen: {[n for n in rooms if 'KITCHEN' in n]}")

print("\n" + "=" * 70)
print("CONNECTIONS:")
print("=" * 70)

# Check nurse station connections
for f in range(3):
    nurse = f"F{f}_NURSE"
    neighbors = list(env.G.neighbors(nurse))
    print(f"\nFloor {f} Nurse Station connections:")
    print(f"  {nurse} -> {neighbors}")

# Check stairs
print("\nStairs between floors:")
for f in range(2):
    c1_f0 = f"F{f}_C1"
    c1_f1 = f"F{f+1}_C1"
    if c1_f1 in env.G.neighbors(c1_f0):
        print(f"  {c1_f0} <-> {c1_f1}")
