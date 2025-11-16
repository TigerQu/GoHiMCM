#!/usr/bin/env python3
"""
Movement time units validation test.

Validates that Fix #2 (civilian traversal time units) works correctly:
- Person edge costs are in timesteps (matching agent movement)
- Hazard penalties use multiplicative factors (realistic routing)
- Civilians and agents move at consistent speeds
"""

import sys
sys.path.insert(0, '/Users/Admin/Desktop/HIMCM 26/GoHiMCM')

from src.environment.env import BuildingFireEnvironment
from src.environment.entities import Person, EdgeMeta
from src.environment.occupants import _edge_cost_for_person

def test_movement_time_units():
    """Test that movement times are in consistent units."""
    
    print("\n" + "="*70)
    print("MOVEMENT TIME UNITS TEST")
    print("="*70)
    
    # Create a simple test environment
    env = BuildingFireEnvironment()
    
    # Add simple topology: two nodes connected by 10m edge
    env.add_node("N0", "room", area=20.0, length=5.0, floor=0)
    env.add_node("N1", "hall", area=10.0, length=5.0, floor=0)
    env.add_edge("N0", "N1", length=10.0, width=2.0, door=False)
    
    # Spawn a test person (adult speed = 0.8 m/s)
    pid = env.spawn_person("N0", age=25, mobility="adult")
    person = env.people[pid]
    
    env.reset(fire_node="none", seed=42)
    
    print("\n[Test 1] Person effective speed matches config")
    print(f"  Adult base speed: {person.v_class} m/s")
    print(f"  Assisted multiplier: {person.assisted_multiplier}")
    print(f"  Panic multiplier: {person.panic_multiplier}")
    
    normal_speed = person.v_class
    print(f"  Normal effective speed: {normal_speed} m/s")
    assert normal_speed == person.effective_speed
    print(f"  ✓ Effective speed matches v_class")
    
    print("\n[Test 2] Edge cost calculation returns timesteps")
    
    # Get the edge
    edge_meta = env.G.edges["N0", "N1"]["meta"]
    print(f"  Edge length: {edge_meta.length} m")
    print(f"  Person speed: {person.v_class} m/s")
    
    # Calculate cost
    cost = _edge_cost_for_person(env, person, "N0", "N1", edge_meta)
    print(f"  Edge cost: {cost} timesteps")
    
    # Manual calculation:
    # - Base time: 10m / 0.8 m/s = 12.5 seconds
    # - Timesteps: 12.5 sec / 5 sec/step = 2.5 timesteps
    # - No congestion, no hazards
    # - Expected: ~2.5 timesteps
    expected_cost = (edge_meta.length / person.v_class) / 5.0
    print(f"  Expected: {expected_cost} timesteps")
    
    assert abs(cost - expected_cost) < 0.01, f"Expected {expected_cost}, got {cost}"
    print(f"  ✓ Cost is in timesteps (not seconds)")
    
    print("\n[Test 3] Agent movement time calculation (for comparison)")
    
    # Agent speed from config
    agent_speed = env.config['agent_speed']  # 1.5 m/s
    print(f"  Agent speed: {agent_speed} m/s")
    
    # Agent traversal calculation (from env.move_agent):
    agent_traversal_seconds = edge_meta.length / agent_speed  # 10/1.5 = 6.67 sec
    agent_traversal_timesteps = agent_traversal_seconds / 5.0  # 6.67/5 = 1.33 timesteps
    print(f"  Agent traversal time: {agent_traversal_seconds:.2f} seconds = {agent_traversal_timesteps:.2f} timesteps")
    
    # Person should take longer (slower speed)
    person_ratio = cost / agent_traversal_timesteps
    print(f"  Person/Agent time ratio: {person_ratio:.2f}x")
    print(f"  Speed ratio: {agent_speed / person.v_class:.2f}x")
    assert abs(person_ratio - (agent_speed / person.v_class)) < 0.01
    print(f"  ✓ Time ratio matches speed ratio (correct units)")
    
    print("\n[Test 4] Hazard penalties use multiplicative factors")
    
    # Fire up the destination node
    env.nodes["N1"].on_fire = True
    
    cost_with_fire = _edge_cost_for_person(env, person, "N0", "N1", edge_meta)
    print(f"  Cost without fire: {cost:.3f} timesteps")
    print(f"  Cost with fire: {cost_with_fire:.3f} timesteps")
    
    ratio = cost_with_fire / cost
    print(f"  Fire hazard multiplier: {ratio:.2f}x")
    assert 1.4 < ratio < 1.6, f"Expected ~1.5x multiplier, got {ratio}x"
    print(f"  ✓ Fire penalty is multiplicative (1.5x) not absolute")
    
    # Test smoke
    env.nodes["N1"].on_fire = False
    env.nodes["N1"].smoky = True
    
    cost_with_smoke = _edge_cost_for_person(env, person, "N0", "N1", edge_meta)
    print(f"  Cost with smoke: {cost_with_smoke:.3f} timesteps")
    
    ratio = cost_with_smoke / cost
    print(f"  Smoke hazard multiplier: {ratio:.2f}x")
    assert 1.05 < ratio < 1.15, f"Expected ~1.1x multiplier, got {ratio}x"
    print(f"  ✓ Smoke penalty is multiplicative (1.1x) not absolute")
    
    print("\n[Test 5] Congestion affects cost additively with hazard")
    
    env.nodes["N1"].smoky = False
    
    # Add congestion to the edge
    env._edge_load[("N0", "N1")] = 5  # 5 people on edge
    
    cost_congested = _edge_cost_for_person(env, person, "N0", "N1", edge_meta)
    print(f"  Cost without congestion: {cost:.3f} timesteps")
    print(f"  Cost with 5 people congestion: {cost_congested:.3f} timesteps")
    
    # Congestion: density = 5 / 10m = 0.5
    # Multiplier: 1 + 0.8 * 0.5 = 1.4
    # Expected: 2.5 * 1.4 = 3.5 timesteps
    expected_congested = cost * 1.4
    assert abs(cost_congested - expected_congested) < 0.01
    print(f"  ✓ Congestion applies as multiplicative factor")
    
    print("\n" + "="*70)
    print("✓ All movement time unit tests PASSED")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_movement_time_units()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n✗ Test ERROR: {e}\n")
        traceback.print_exc()
        sys.exit(1)
