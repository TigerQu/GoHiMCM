# your_solution/planner/frontier.py
from collections import deque
from typing import Dict, List, Set, Iterable, Optional

# 允许作为 frontier 目标的节点类型
DEFAULT_TARGET_TYPES = {"room"}  # 如需把 hall 也当目标：{"room", "hall"}

class FrontierExtractor:
    """
    提取“可搜索前沿（frontier）”的工具：
      - 过滤：未扫过(swept=False)、目标类型(room/hall)
      - 连通性：从任一 agent 位置可达（不穿越 on_fire）
      - 安全开关：是否直接排除 on_fire / smoky 节点
    """

    def __init__(
        self,
        target_types: Optional[Set[str]] = None,
        exclude_fire: bool = True,
        exclude_unreachable: bool = True,
        allow_smoke: bool = True,
    ):
        """
        Args:
            target_types: 作为候选的节点类型集合，默认只含 room
            exclude_fire: True 则直接排除 on_fire 节点
            exclude_unreachable: True 则仅保留从 agent 出发可达的节点
            allow_smoke: True 则保留 smoky 节点（交给打分层惩罚）；False 则滤掉
        """
        self.target_types = target_types or set(DEFAULT_TARGET_TYPES)
        self.exclude_fire = exclude_fire
        self.exclude_unreachable = exclude_unreachable
        self.allow_smoke = allow_smoke

    def extract(
        self,
        nodes: Dict[str, object],
        agents: Dict[int, object],
        neighbors_fn,
    ) -> List[str]:
        """
        主入口：返回 frontier 节点列表（node_id）

        Args:
            nodes: node_id -> NodeMeta（需要字段：ntype, swept, on_fire, smoky）
            agents: agent_id -> Agent（需要字段：node_id）
            neighbors_fn: 函数 neighbors_fn(nid) -> Iterable[str]，给出 nid 的邻居

        Returns:
            List[str]: frontier 节点 id 列表（已去重、已按连通性和安全过滤）
        """
        # 1) 初筛：类型 & 未扫过
        candidates = [
            nid for nid, meta in nodes.items()
            if getattr(meta, "ntype", None) in self.target_types
            and (not getattr(meta, "swept", False))
        ]

        if not candidates:
            return []

        # 2) 安全过滤：根据火/烟配置删掉一些候选
        safe_candidates = []
        for nid in candidates:
            m = nodes[nid]
            if self.exclude_fire and getattr(m, "on_fire", False):
                # 着火点直接剔除
                continue
            if (not self.allow_smoke) and getattr(m, "smoky", False):
                # 不允许烟雾目标
                continue
            safe_candidates.append(nid)

        if not safe_candidates:
            return []

        # 3) 连通性过滤：从任一 agent 位置出发，做 BFS/DFS，
        #    但不允许穿越 on_fire 的节点作为中转（防止“绕火穿越”）
        if self.exclude_unreachable:
            # 多源 BFS 起点：所有 agent 所在位置
            starts = [ag.node_id for ag in agents.values()]
            reachable = self._multi_source_reachable(
                starts=starts,
                neighbors_fn=neighbors_fn,
                nodes=nodes,
            )
            frontier = [nid for nid in safe_candidates if nid in reachable]
        else:
            frontier = list(safe_candidates)

        # 返回一个稳定顺序（可选：按 node_id 排个序，或保持原顺序）
        frontier.sort()
        return frontier

    def _multi_source_reachable(
        self,
        starts: Iterable[str],
        neighbors_fn,
        nodes: Dict[str, object],
    ) -> Set[str]:
        """
        从多个起点做 BFS，返回在“不穿越 on_fire 的前提下”可达的所有节点集合。
        （允许经过 smoky；是否走 smoky 的成本交给后续最短路来惩罚）

        注意：如果某个节点本身 on_fire，我们也不把它计入 reachable。
        """
        visited: Set[str] = set()
        dq = deque()

        for s in starts:
            if s is None:
                continue
            # 起点若在 fire 上，也不入队；避免 agent 起始点异常
            if getattr(nodes.get(s, None), "on_fire", False):
                continue
            visited.add(s)
            dq.append(s)

        while dq:
            u = dq.popleft()
            for v in neighbors_fn(u):
                if v in visited:
                    continue
                vm = nodes.get(v, None)
                if vm is None:
                    continue
                # 不穿越火点
                if getattr(vm, "on_fire", False):
                    continue
                visited.add(v)
                dq.append(v)

        return visited
