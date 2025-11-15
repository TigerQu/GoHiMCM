import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx

from Env_sim.layouts import build_standard_office_layout
from Env_sim.env import BuildingFireEnvironment

if __name__ == "__main__":
    print("Building Fire Evacuation Simulation (no-reward test)")
    print("=" * 60)

    # 1) 构建环境 + 固定随机种子方便复现
    env: BuildingFireEnvironment = build_standard_office_layout()
    env.seed(42)

    # 2) 重置：点火到指定房间（或 fire_node=None 随机房间）
    env.reset(fire_node="RT1", seed=42)

    # 3) 一个非常简单的“动作生成器”（只为 smoke test，不追求最优）
    def pick_actions(env):
        """
        返回 {agent_id: action_str}，action_str 只会是：
        - "search"
        - "move_<NEIGHBOR_NODE_ID>"
        - "wait"
        """
        actions = {}
        for aid, agent in env.agents.items():
            nid = agent.node_id
            node = env.nodes[nid]

            # If agent is currently assisting someone, escort them to nearest exit
            if agent.assisting_person_id is not None:
                # Find nearest exit and take the first step along shortest path
                exits = [nid for nid, m in env.nodes.items() if m.ntype == "exit"]
                if exits:
                    try:
                        nearest = min(exits, key=lambda e: nx.shortest_path_length(
                            env.G, nid, e, weight=lambda u, v, d: d['meta'].length
                        ))
                        path = nx.shortest_path(env.G, nid, nearest, weight=lambda u, v, d: d['meta'].length)
                        if len(path) > 1:
                            actions[aid] = f"move_{path[1]}"
                            continue
                    except Exception:
                        # Fall through to regular behavior if path search fails
                        pass

            # 如果当前在 room 且未扫过 → 先 search
            # If there are discovered, unassisted people here, help them first
            assisted = False
            for pid in node.people:
                p = env.people[pid]
                if p.is_alive and p.seen and not p.being_assisted and not p.rescued:
                    actions[aid] = f"assist_{pid}"
                    assisted = True
                    break
            if assisted:
                continue

            if node.ntype == "room" and not node.swept:
                actions[aid] = "search"
                continue

            # 否则优先去相邻“未扫过的 room”，找不到就去任一邻居
            neighbors = list(env.G.neighbors(nid))

            # 先挑未扫过的 room
            target = None
            for nb in neighbors:
                if env.nodes[nb].ntype == "room" and not env.nodes[nb].swept:
                    target = nb
                    break

            # 次选：未扫过的 hall
            if target is None:
                for nb in neighbors:
                    if env.nodes[nb].ntype == "hall" and not env.nodes[nb].swept:
                        target = nb
                        break

            # 实在没有就随便动一个邻居；没有邻居就 wait
            if target is None and neighbors:
                target = neighbors[0]

            if target is None:
                actions[aid] = "wait"
            else:
                actions[aid] = f"move_{target}"

        return actions

    # 4) 主循环：只推进若干步，不计算 reward
    MAX_TEST_STEPS = 200
    for t in range(MAX_TEST_STEPS):
        # 选动作（不考虑 reward，仅作演示）
        actions = pick_actions(env)

        # ——不调用 do_action（它会触发 _compute_reward）——
        # 手动执行动作：
        for aid, act in actions.items():
            if act.startswith("move_"):
                target = act[5:]
                env.move_agent(aid, target)
            elif act == "search":
                env.start_search(aid)
            # "wait" 什么也不做

        # 推进一步（含：process_searches → spread_hazards → move_civilians → degrade_health → time++)
        env.step()

        # 打印当前步的关键信息
        s = env.get_statistics()
        print(f"[T={env.time_step:02d}] "
              f"swept={s['nodes_swept']}, found={s['people_found']}, "
              f"rescued={s['people_rescued']}, alive={s['people_alive']}, "
              f"sweep_complete={s['sweep_complete']}")

        # 也可以看看 agent 位置
        for aid, ag in env.agents.items():
            print(f"  - Agent {aid} @ {ag.node_id}")

        # 所有房间扫完就提前结束
        if s["sweep_complete"]:
            print("Sweep complete. Stop early.")
            break

    print("\nFinal status:")
    env.print_status()


from Env_sim.env import BuildingFireEnvironment

env = BuildingFireEnvironment()

# Build minimal layout
env.add_node("R1", "room", length=5.0)
env.add_node("E1", "exit", length=5.0)
env.add_edge("R1", "E1", length=5.0, width=1.0)

# One person in R1
env.spawn_person("R1", age=30, mobility="adult", hp=100.0)

# Make awareness delay zero for testing
env.config["awareness_delay_mean"] = 0.0
env.config["awareness_delay_std"] = 0.0

env.reset(fire_node="none", seed=0)

for t in range(20):
    env.step()
    stats = env.get_statistics()
    print(f"t={t:2d} alive={stats['people_alive']} rescued={stats['people_rescued']}")

