"""
Simple test to understand reward calculation.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.layouts import build_standard_office_layout
from rl.reward_shaper import RewardShaper


def test_rewards():
    """Test what rewards look like in a simple scenario."""
    
    print("=" * 70)
    print("REWARD CALCULATION TEST")
    print("=" * 70 + "\n")
    
    # Create environment
    env = build_standard_office_layout()
    reward_shaper = RewardShaper.for_scenario("office")
    
    print("Reward weights:")
    print(f"  coverage: {reward_shaper.w_coverage}")
    print(f"  rescue: {reward_shaper.w_rescue}")
    print(f"  hp_loss: {reward_shaper.w_hp_loss}")
    print(f"  time: {reward_shaper.w_time}")
    print(f"  redundancy: {reward_shaper.w_redundancy}")
    print()
    
    # Reset environment
    env.reset(seed=42)
    reward_shaper.reset()
    
    print("Initial state:")
    stats = env.get_statistics()
    print(f"  People: {stats.get('total_people', 'unknown')}")
    print(f"  Nodes swept: {stats['nodes_swept']}")
    print(f"  People rescued: {stats['people_rescued']}")
    print()
    
    # Take some steps and see rewards
    print("Step-by-step rewards (just waiting to see base penalties):")
    print("-" * 70)
    
    for step in range(10):
        # Just wait
        actions = {0: "wait", 1: "wait"}
        
        # Execute action
        obs, env_reward, done, info = env.do_action(actions)
        
        # Compute shaped reward
        shaped_reward = reward_shaper.compute_reward(env)
        
        stats = env.get_statistics()
        state = env.get_state()
        
        # Count alive people and their HP
        alive_count = sum(1 for p in state['people'].values() if p.is_alive)
        avg_hp = sum(p.hp for p in state['people'].values()) / len(state['people']) if state['people'] else 0
        
        print(f"\nStep {step + 1}:")
        print(f"  Shaped reward: {shaped_reward:.3f}")
        print(f"  People alive: {alive_count}/{len(state['people'])}, Avg HP: {avg_hp:.1f}")
        print(f"  Swept: {stats['nodes_swept']}, Found: {stats['people_found']}, Rescued: {stats['people_rescued']}")
        
        if done:
            print("\n  Episode ended!")
            break
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Calculate average reward per step
    total_steps = step + 1
    avg_reward = shaped_reward / total_steps if total_steps > 0 else 0
    
    print(f"\nIf episodes are ~100 steps:")
    print(f"  Time penalty per episode: {-reward_shaper.w_time * 100:.2f}")
    print(f"  Need to sweep {reward_shaper.w_time * 100 / reward_shaper.w_coverage:.1f} rooms to break even")
    print(f"  OR rescue {reward_shaper.w_time * 100 / reward_shaper.w_rescue:.2f} people to break even")
    print()
    print("Verdict:")
    if reward_shaper.w_time * 100 > reward_shaper.w_rescue * 2:
        print("  ❌ Time penalty too high! Even rescuing 2 people won't overcome it.")
    elif reward_shaper.w_time * 100 > reward_shaper.w_coverage * 5:
        print("  ⚠️  Time penalty high. Need to sweep 5+ rooms to break even.")
    else:
        print("  ✅ Reward balance looks reasonable.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_rewards()
