#!/usr/bin/env python3
"""
Configuration override test.

Validates that per-environment config overrides work correctly,
specifically testing Fix #3: Person.effective_speed uses per-env config.
"""

import sys
sys.path.insert(0, '/Users/Admin/Desktop/HIMCM 26/GoHiMCM')

from src.Env_sim.layouts import build_standard_office_layout

def test_config_overrides():
    """Test that custom config multipliers are respected by Person.effective_speed."""
    
    print("\n" + "="*70)
    print("CONFIG OVERRIDE TEST")
    print("="*70)
    
    # Test 1: Default config
    print("\n[Test 1] Default config multipliers")
    env_default = build_standard_office_layout()
    env_default.reset(fire_node="none", seed=42)
    
    person_default = env_default.people[0]
    default_assisted_mult = person_default.assisted_multiplier
    default_panic_mult = person_default.panic_multiplier
    
    print(f"  Default assisted_multiplier: {default_assisted_mult}")
    print(f"  Default panic_multiplier: {default_panic_mult}")
    assert default_assisted_mult == 1.8, f"Expected 1.8, got {default_assisted_mult}"
    assert default_panic_mult == 0.7, f"Expected 0.7, got {default_panic_mult}"
    print(f"  ✓ Defaults match DEFAULT_CONFIG")
    
    # Test 2: Custom config with different multipliers
    print("\n[Test 2] Custom config with different multipliers")
    custom_config = {
        'assisted_speed_multiplier': 2.5,  # 2.5x faster when assisted
        'panic_speed_reduction': 0.5,      # 0.5x speed when panicked
    }
    
    # Build layout first, then create with custom config before spawning people
    from src.Env_sim.env import BuildingFireEnvironment
    from src.Env_sim.config import DEFAULT_CONFIG
    
    env_custom = BuildingFireEnvironment(config=custom_config)
    
    # Manually build the same office layout but with custom config
    env_custom.add_node("H0", "hall", length=6.0, area=12.0, floor=0)
    env_custom.add_node("H1", "hall", length=6.0, area=12.0, floor=0)
    env_custom.add_node("H2", "hall", length=6.0, area=12.0, floor=0)
    
    env_custom.add_edge("H0", "H1", length=6.0, width=2.0, door=False)
    env_custom.add_edge("H1", "H2", length=6.0, width=2.0, door=False)
    
    for i in range(3):
        env_custom.add_node(f"RT{i}", "room", length=4.0, area=16.0, floor=0)
        env_custom.add_edge(f"RT{i}", f"H{i}", length=1.0, width=0.9, door=True)
        
        env_custom.add_node(f"RB{i}", "room", length=4.0, area=16.0, floor=0)
        env_custom.add_edge(f"RB{i}", f"H{i}", length=1.0, width=0.9, door=True)
    
    env_custom.add_node("EXIT_LEFT", "exit", length=2.0, area=4.0, floor=0)
    env_custom.add_node("EXIT_RIGHT", "exit", length=2.0, area=4.0, floor=0)
    env_custom.add_edge("EXIT_LEFT", "H0", length=1.0, width=1.2, door=True)
    env_custom.add_edge("EXIT_RIGHT", "H2", length=1.0, width=1.2, door=True)
    
    # Now spawn people (which will use the custom config)
    env_custom.spawn_person("RT0", age=34, mobility="adult", hp=95.0)
    env_custom.spawn_person("RT2", age=55, mobility="limited", hp=90.0)
    env_custom.spawn_person("RB0", age=28, mobility="adult", hp=100.0)
    env_custom.spawn_person("RB2", age=6, mobility="child", hp=100.0)
    env_custom.spawn_person("RB1", age=32, mobility="adult", hp=100.0)
    env_custom.spawn_person("RB1", age=41, mobility="adult", hp=97.0)
    
    env_custom.place_agent(agent_id=0, node_id="EXIT_LEFT")
    env_custom.place_agent(agent_id=1, node_id="EXIT_RIGHT")
    
    env_custom.reset(fire_node="none", seed=42)
    
    person_custom = env_custom.people[0]
    custom_assisted_mult = person_custom.assisted_multiplier
    custom_panic_mult = person_custom.panic_multiplier
    
    print(f"  Custom assisted_multiplier: {custom_assisted_mult}")
    print(f"  Custom panic_multiplier: {custom_panic_mult}")
    assert custom_assisted_mult == 2.5, f"Expected 2.5, got {custom_assisted_mult}"
    assert custom_panic_mult == 0.5, f"Expected 0.5, got {custom_panic_mult}"
    print(f"  ✓ Custom config properly applied to Person objects")
    
    # Test 3: Verify effective_speed uses the stored multipliers
    print("\n[Test 3] Verify effective_speed uses stored multipliers")
    
    # Default person with v_class = 0.8 (adult_speed)
    person_default.v_class = 1.0  # Set to 1.0 for easy math
    
    normal_speed = person_default.effective_speed
    print(f"  Normal speed: {normal_speed} (expected 1.0 * 1.0 = 1.0)")
    assert abs(normal_speed - 1.0) < 0.001, f"Expected 1.0, got {normal_speed}"
    
    person_default.being_assisted = True
    assisted_speed = person_default.effective_speed
    print(f"  Assisted speed: {assisted_speed} (expected 1.0 * 1.8 = 1.8)")
    assert abs(assisted_speed - 1.8) < 0.001, f"Expected 1.8, got {assisted_speed}"
    
    person_default.being_assisted = False
    person_default.panicked = True
    panicked_speed = person_default.effective_speed
    print(f"  Panicked speed: {panicked_speed} (expected 1.0 * 0.7 = 0.7)")
    assert abs(panicked_speed - 0.7) < 0.001, f"Expected 0.7, got {panicked_speed}"
    
    print(f"  ✓ effective_speed correctly uses stored multipliers")
    
    # Test 4: Custom config speeds
    print("\n[Test 4] Custom config speeds")
    
    person_custom.v_class = 1.0
    person_custom.being_assisted = False
    person_custom.panicked = False
    
    normal_speed_custom = person_custom.effective_speed
    print(f"  Normal speed: {normal_speed_custom} (expected 1.0 * 1.0 = 1.0)")
    assert abs(normal_speed_custom - 1.0) < 0.001
    
    person_custom.being_assisted = True
    assisted_speed_custom = person_custom.effective_speed
    print(f"  Assisted speed: {assisted_speed_custom} (expected 1.0 * 2.5 = 2.5)")
    assert abs(assisted_speed_custom - 2.5) < 0.001, f"Expected 2.5, got {assisted_speed_custom}"
    
    person_custom.being_assisted = False
    person_custom.panicked = True
    panicked_speed_custom = person_custom.effective_speed
    print(f"  Panicked speed: {panicked_speed_custom} (expected 1.0 * 0.5 = 0.5)")
    assert abs(panicked_speed_custom - 0.5) < 0.001, f"Expected 0.5, got {panicked_speed_custom}"
    
    print(f"  ✓ Custom multipliers correctly affect effective_speed")
    
    # Test 5: Reset preserves custom config
    print("\n[Test 5] Reset preserves custom config multipliers")
    
    env_custom.reset(fire_node="none", seed=99)
    person_after_reset = env_custom.people[0]
    
    reset_assisted = person_after_reset.assisted_multiplier
    reset_panic = person_after_reset.panic_multiplier
    
    print(f"  After reset - assisted: {reset_assisted}, panic: {reset_panic}")
    assert reset_assisted == 2.5, f"Expected 2.5 after reset, got {reset_assisted}"
    assert reset_panic == 0.5, f"Expected 0.5 after reset, got {reset_panic}"
    print(f"  ✓ Custom config preserved across reset()")
    
    print("\n" + "="*70)
    print("✓ All config override tests PASSED")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_config_overrides()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n✗ Test ERROR: {e}\n")
        traceback.print_exc()
        sys.exit(1)
