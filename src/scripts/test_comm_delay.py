"""
Test script for communication delay feature.

Shows how messages are delayed by N time steps.
"""

import sys
sys.path.insert(0, "/Users/qzy/Projects/GoHiMCM")

from src.Env_sim.layouts import build_standard_office_layout

def test_delay(delay):
    print(f"\n{'='*70}")
    print(f"TEST WITH COMM_DELAY = {delay}")
    print(f"{'='*70}\n")
    
    config = {"comm_delay": delay}
    env = build_standard_office_layout()
    env.config.update(config)
    env.reset()
    
    print(f"Config: comm_delay={env.config['comm_delay']}\n")
    
    # Agent 0 searches RT0
    print(f"Time {env.time_step}: Agent 0 moves to RT0 and starts searching")
    env.move_agent(0, "H0")
    env.move_agent(0, "RT0")
    env.start_search(0)
    env.step()
    
    print(f"Time {env.time_step}: Agent 0 completed search of RT0")
    print(f"  Agent 0 searched_nodes: {env.agents[0].searched_nodes}")
    print(f"  Agent 1 known_swept_nodes: {env.agents[1].known_swept_nodes}")
    print(f"  Agent 1 message_buffer: {len(env.agents[1].message_buffer)} messages")
    print(f"  Pending messages in queue: {len(env.pending_messages)}")
    print()
    
    # Advance time until message is delivered
    for i in range(delay + 2):
        env.step()
        print(f"Time {env.time_step}:")
        print(f"  Agent 1 known_swept_nodes: {env.agents[1].known_swept_nodes}")
        print(f"  Agent 1 message_buffer: {len(env.agents[1].message_buffer)} messages")
        print(f"  Pending messages: {len(env.pending_messages)}")
        
        if env.agents[1].message_buffer:
            last_msg = env.agents[1].message_buffer[-1]
            print(f"  Last message: sent at t={last_msg['timestamp']}, "
                  f"delivered at t={last_msg['delivery_time']}")
        print()

def main():
    print("=" * 70)
    print("COMMUNICATION DELAY TEST")
    print("=" * 70)
    
    # Test with no delay
    test_delay(0)
    
    # Test with 3 step delay
    test_delay(3)
    
    # Test with 5 step delay  
    test_delay(5)

if __name__ == "__main__":
    main()
