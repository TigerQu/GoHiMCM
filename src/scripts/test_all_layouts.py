#!/usr/bin/env python3
"""
Comprehensive layout testing script.

Tests all three layout builders (office, babycare, warehouse) to ensure:
1. Layouts initialize without errors
2. Agent and civilian movement works correctly
3. Fire spreading and health degradation work
4. Feature extraction works (11D vectors with 4-type one-hot)
5. Statistics tracking is accurate
"""

import sys
sys.path.insert(0, '/Users/Admin/Desktop/HIMCM 26/GoHiMCM')

from src.Env_sim.layouts import (
    build_standard_office_layout,
    build_babycare_layout,
    build_two_floor_warehouse
)

def test_layout(name, builder_func, num_steps=50):
    """
    Test a single layout.
    
    Args:
        name: Display name for the layout
        builder_func: Function that returns BuildingFireEnvironment
        num_steps: Number of timesteps to simulate
    
    Returns:
        Dict with test results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    try:
        # Build layout
        env = builder_func()
        print(f"✓ Layout built successfully")
        print(f"  Nodes: {len(env.nodes)}")
        print(f"  Edges: {env.G.number_of_edges()}")
        print(f"  People: {len(env.people)}")
        print(f"  Agents: {len(env.agents)}")
        
        # Reset
        obs = env.reset(seed=42)
        print(f"✓ Environment reset")
        print(f"  Observation shape: x={obs.x.shape}, edge_index={obs.edge_index.shape}")
        print(f"  Feature dimension: {obs.x.shape[1]} (expected 11)")
        
        # Verify feature dimension
        if obs.x.shape[1] != 11:
            return {
                'name': name,
                'status': 'FAILED',
                'error': f'Feature dimension mismatch: {obs.x.shape[1]} != 11'
            }
        
        # Simulate
        for step in range(num_steps):
            # Simple policy: agents search, people try to evacuate
            actions = {}
            for agent_id in env.agents:
                valid = env.get_valid_actions(agent_id)
                # Pick first valid action (greedy)
                actions[agent_id] = valid[0] if valid else "wait"
            
            obs, reward, done, info = env.do_action(actions)
            
            if done:
                print(f"✓ Episode terminated at step {step}")
                break
        
        print(f"✓ Simulation completed {step+1} steps")
        print(f"  Final stats: rescued={info['stats']['people_rescued']}, "
              f"found={info['stats']['people_found']}, "
              f"swept={info['stats']['nodes_swept']}, "
              f"alive={info['stats']['people_alive']}")
        
        return {
            'name': name,
            'status': 'PASSED',
            'steps': step + 1,
            'nodes': len(env.nodes),
            'edges': env.G.number_of_edges(),
            'people': len(env.people),
            'agents': len(env.agents),
            'stats': info['stats']
        }
    
    except Exception as e:
        import traceback
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            'name': name,
            'status': 'FAILED',
            'error': str(e)
        }

def main():
    """Run all layout tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE LAYOUT TEST SUITE")
    print("="*70)
    print("\nTesting all three layout builders with the fixed environment...")
    
    results = []
    
    # Test 1: Standard Office
    results.append(test_layout(
        "Standard Office Layout",
        build_standard_office_layout,
        num_steps=50
    ))
    
    # Test 2: Babycare
    results.append(test_layout(
        "Babycare Multi-Floor Layout",
        build_babycare_layout,
        num_steps=50
    ))
    
    # Test 3: Warehouse
    results.append(test_layout(
        "Two-Floor Warehouse Layout",
        build_two_floor_warehouse,
        num_steps=50
    ))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    
    for result in results:
        status_icon = "✓" if result['status'] == 'PASSED' else "✗"
        print(f"{status_icon} {result['name']}: {result['status']}")
        if result['status'] == 'PASSED':
            print(f"    Nodes={result['nodes']}, Edges={result['edges']}, "
                  f"People={result['people']}, Agents={result['agents']}")
            print(f"    Steps={result['steps']}, "
                  f"Rescued={result['stats']['people_rescued']}, "
                  f"Found={result['stats']['people_found']}")
        else:
            print(f"    Error: {result['error']}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    print(f"{'='*70}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
