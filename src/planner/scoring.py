# planner/scoring.py
from typing import Dict, List, Tuple, Callable
import math

class ScoringConfig:
    def __init__(
        self,
        w_risk: float = 1.0,
        w_info: float = 0.8,
        w_access: float = 0.6,
        fire_bonus: float = 1.0,
        smoke_bonus: float = 0.5,
        childcare_bonus: float = 0.8,
        dist_to_fire_weight: float = 0.5,   # 越近越危险 => (1 - dist_norm)*权重
        people_count_weight: float = 0.7,   # 观测人数越多优先
        low_hp_weight: float = 0.6,         # 平均HP越低优先 => (1 - avg_hp_norm)
        access_eps: float = 1e-3,           # 防止除零
    ):
        self.w_risk = w_risk
        self.w_info = w_info
        self.w_access = w_access
        self.fire_bonus = fire_bonus
        self.smoke_bonus = smoke_bonus
        self.childcare_bonus = childcare_bonus
        self.dist_to_fire_weight = dist_to_fire_weight
        self.people_count_weight = people_count_weight
        self.low_hp_weight = low_hp_weight
        self.access_eps = access_eps


def compute_room_score(
    node,                             # NodeMeta
    agent_node_ids: List[str],        # 当前所有 agent 的节点ID
    travel_time_func: Callable[[str, str], float],  # 旅行时间估计 T(u->v)
    cfg: ScoringConfig
) -> float:
    """
    为一个“可作为目标的房间节点 node”计算优先级得分。
    该函数只使用 node 的“观测可见字段”和拓扑/火情，可解释、不依赖学习。
    """
    # --- 1) Risk ---
    risk = 0.0
    if node.on_fire:
        risk += cfg.fire_bonus
    elif node.smoky:
        risk += cfg.smoke_bonus

    # 距离火源越近越危险：dist_to_fire_norm ∈ [0,1]，越小越近
    risk += (1.0 - float(getattr(node, "dist_to_fire_norm", 1.0))) * cfg.dist_to_fire_weight

    # 业务标签：托儿所/老年房等
    if getattr(node, "childcare", False):
        risk += cfg.childcare_bonus

    # --- 2) Info ---
    # 观测人数多、平均HP低 => 信息/救援价值高
    ppl_cnt_norm = float(getattr(node, "obs_people_count", 0.0))  # 已经是 0..3 capped/3 的“比例”？按你的实现适配
    avg_hp_norm = float(getattr(node, "obs_avg_hp", 0.0))        # 0..1
    info = ppl_cnt_norm * cfg.people_count_weight + (1.0 - avg_hp_norm) * cfg.low_hp_weight

    # 未扫过的房间天然有信息价值；如果你希望更强，加入一个固定加成
    if not node.swept:
        info += 0.2  # 轻微鼓励

    # --- 3) Access ---
    # 取所有 agent 中到该节点的最短时间（基于 travel_time_func），然后做 1/(eps + min_T) 归一化
    min_t = math.inf
    for aid_node in agent_node_ids:
        t = travel_time_func(aid_node, node.nid)
        if t < min_t:
            min_t = t
    access = 0.0 if not math.isfinite(min_t) else 1.0 / (cfg.access_eps + min_t)

    # --- 合成 ---
    score = cfg.w_risk * risk + cfg.w_info * info + cfg.w_access * access
    return float(score)


def score_candidate_rooms(
    nodes_dict: Dict[str, object],                 # {nid: NodeMeta}
    agent_node_ids: List[str],
    travel_time_func: Callable[[str, str], float],
    cfg: ScoringConfig
) -> List[Tuple[str, float]]:
    """
    对所有“可作为目标的房间”（ntype == 'room' 且未扫）计算分数，返回 [(nid, score)]，按分数降序。
    """
    results = []
    for nid, node in nodes_dict.items():
        if getattr(node, "ntype", "") != "room":
            continue
        if getattr(node, "swept", False):
            continue

        s = compute_room_score(node, agent_node_ids, travel_time_func, cfg)
        results.append((nid, s))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
