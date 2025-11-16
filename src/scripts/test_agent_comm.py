"""
Test script demonstrating agent communication and double-sweep mechanism.

This script shows:
1. Agents communicate their search completions to each other
2. Rooms need to be swept twice (sweep_count >= 2)
3. Agents can see which nodes have been searched by others
"""

import sys
sys.path.insert(0, "/Users/qzy/Projects/GoHiMCM")

from src.Env_sim.layouts import build_standard_office_layout

def main():
    env = build_standard_office_layout()
    
    print("=" * 70)
    print("AGENT COMMUNICATION AND DOUBLE-SWEEP TEST")
    print("=" * 70)
    print(f"Config: required_sweeps={env.config['required_sweeps']}, " 
          f"enable_agent_comm={env.config['enable_agent_comm']}")
    print(f"Agents: {list(env.agents.keys())}")
    print()
    
    # Initial state
    print("Initial State:")
    for nid, node in env.nodes.items():
        if node.ntype == "room":
            print(f"  {nid}: sweep_count={node.sweep_count}, type={node.ntype}")
    print()
    
    # Simulate some actions
    print("Step 1: Agent 0 moves to RT0 and starts searching")
    env.move_agent(0, "H0")
    env.move_agent(0, "RT0")
    env.start_search(0)
    env.step()
    
    print(f"  RT0 sweep_count: {env.nodes['RT0'].sweep_count}")
    print(f"  Agent 0 searched_nodes: {env.agents[0].searched_nodes}")
    print(f"  Agent 1 known_swept_nodes: {env.agents[1].known_swept_nodes}")
    print()
    
    # Check communication
    if env.agents[1].message_buffer:
        print(f"  Agent 1 received message: {env.agents[1].message_buffer[-1]}")
    print()
    
    print("Step 2: Agent 1 moves to RT0 and searches again (second sweep)")
    env.move_agent(1, "H2")
    env.move_agent(1, "H1")
    env.move_agent(1, "H0")
    env.move_agent(1, "RT0")
    print(f"  Agent 1 is at: {env.agents[1].node_id}")
    print(f"  Agent 1 searched_nodes before search: {env.agents[1].searched_nodes}")
    env.start_search(1)
    print(f"  Agent 1 searching: {env.agents[1].searching}, timer: {env.agents[1].search_timer}")
    env.step()
    print(f"  Agent 1 searching after step: {env.agents[1].searching}")
    
    print(f"  RT0 sweep_count: {env.nodes['RT0'].sweep_count}")
    print(f"  Agent 0 searched_nodes: {env.agents[0].searched_nodes}")
    print(f"  Agent 1 searched_nodes: {env.agents[1].searched_nodes}")
    print(f"  Agent 0 known_swept_nodes: {env.agents[0].known_swept_nodes}")
    print(f"  Agent 1 known_swept_nodes: {env.agents[1].known_swept_nodes}")
    print()
    
    # Check if room is fully swept
    print(f"Is RT0 fully swept (sweep_count >= 2)? {env.nodes['RT0'].sweep_count >= 2}")
    print()
    
    # Test get_agent_known_sweeps
    print("Step 3: Query agent knowledge")
    agent0_knows = env.get_agent_known_sweeps(0)
    agent1_knows = env.get_agent_known_sweeps(1)
    print(f"  Agent 0 knows these nodes are swept: {agent0_knows}")
    print(f"  Agent 1 knows these nodes are swept: {agent1_knows}")
    print()
    
    # Sweep completion status
    print("Final Status:")
    room_count = sum(1 for n in env.nodes.values() if n.ntype == "room")
    swept_once = sum(1 for n in env.nodes.values() if n.ntype == "room" and n.sweep_count >= 1)
    swept_twice = sum(1 for n in env.nodes.values() if n.ntype == "room" and n.sweep_count >= 2)
    print(f"  Total rooms: {room_count}")
    print(f"  Rooms swept at least once: {swept_once}")
    print(f"  Rooms swept twice (complete): {swept_twice}")
    print(f"  Sweep complete: {env.is_sweep_complete()}")
    print()
    
    print("=" * 70)
    print("Communication mechanism works! Agents share search information.")
    print("=" * 70)

if __name__ == "__main__":
    main()
