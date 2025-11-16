import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx

from environment.layouts import build_standard_office_layout
from environment.env import BuildingFireEnvironment


def pick_actions(env):
    """Simple action generator compatible with the current Agent schema.

    Returns a dict {agent_id: action_str} where action_str is one of:
    - "search"
    - "move_<NEIGHBOR_NODE_ID>"
    - "wait"
    """
    actions = {}
    for aid, agent in env.agents.items():
        nid = agent.node_id
        node = env.nodes[nid]

        # If current node is a room and not swept yet -> search it
        if getattr(node, "swept", False) is False and node.ntype == "room":
            actions[aid] = "search"
            continue

        # Otherwise prefer an adjacent unswept room, then unswept hall, else move to any neighbor
        neighbors = list(env.G.neighbors(nid))
        target = None
        for nb in neighbors:
            if env.nodes[nb].ntype == "room" and not getattr(env.nodes[nb], "swept", False):
                target = nb
                break

        if target is None:
            for nb in neighbors:
                if env.nodes[nb].ntype == "hall" and not getattr(env.nodes[nb], "swept", False):
                    target = nb
                    break

        if target is None and neighbors:
            target = neighbors[0]

        if target is None:
            actions[aid] = "wait"
        else:
            actions[aid] = f"move_{target}"

    return actions


if __name__ == "__main__":
    print("Building Fire Evacuation Simulation (no-reward test)")
    print("=" * 60)

    # Build env and seed for reproducibility
    env: BuildingFireEnvironment = build_standard_office_layout()
    env.seed(42)

    # Reset (ignite a room for demonstration)
    env.reset(fire_node="RT1", seed=42)

    # Main loop: step forward a bounded number of steps
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import networkx as nx

    from environment.layouts import build_standard_office_layout
    from environment.env import BuildingFireEnvironment


    def pick_actions(env):
        """Simple action generator compatible with the current Agent schema.

        Returns a dict {agent_id: action_str} where action_str is one of:
        - "search"
        - "move_<NEIGHBOR_NODE_ID>"
        - "wait"
        """
        actions = {}
        for aid, agent in env.agents.items():
            nid = agent.node_id
            node = env.nodes[nid]

            # If current node is a room and not swept yet -> search it
            if getattr(node, "swept", False) is False and node.ntype == "room":
                actions[aid] = "search"
                continue

            # Otherwise prefer an adjacent unswept room, then unswept hall, else move to any neighbor
            neighbors = list(env.G.neighbors(nid))
            target = None
            for nb in neighbors:
                if env.nodes[nb].ntype == "room" and not getattr(env.nodes[nb], "swept", False):
                    target = nb
                    break

            if target is None:
                for nb in neighbors:
                    if env.nodes[nb].ntype == "hall" and not getattr(env.nodes[nb], "swept", False):
                        target = nb
                        break

            if target is None and neighbors:
                target = neighbors[0]

            if target is None:
                actions[aid] = "wait"
            else:
                actions[aid] = f"move_{target}"

        return actions


    if __name__ == "__main__":
        print("Building Fire Evacuation Simulation (no-reward test)")
        print("=" * 60)

        # Build env and seed for reproducibility
        env: BuildingFireEnvironment = build_standard_office_layout()
        env.seed(42)

        # Reset (ignite a room for demonstration)
        env.reset(fire_node="RT1", seed=42)

        # Main loop: step forward a bounded number of steps
        MAX_TEST_STEPS = 200
        for t in range(MAX_TEST_STEPS):
            actions = pick_actions(env)

            # Apply actions directly (avoid calling do_action to skip reward logic)
            for aid, act in actions.items():
                if act.startswith("move_"):
                    target = act[5:]
                    env.move_agent(aid, target)
                elif act == "search":
                    env.start_search(aid)
                # wait -> nothing

            env.step()

            s = env.get_statistics()
            print(f"[T={env.time_step:02d}] "
                  f"swept={s['nodes_swept']}, found={s['people_found']}, "
                  f"rescued={s['people_rescued']}, alive={s['people_alive']}, "
                  f"sweep_complete={s['sweep_complete']}")

            for aid, ag in env.agents.items():
                print(f"  - Agent {aid} @ {ag.node_id}")

            if s["sweep_complete"]:
                print("Sweep complete. Stop early.")
                break

        print("\nFinal status:")
        env.print_status()

