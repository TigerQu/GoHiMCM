import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from collections import deque
import os

# ============================================================================
# ENVIRONMENT (Enhanced RescueEnv)
# ============================================================================
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
import copy

FIRE_SPREAD_PROBABILITY = 0.25
SMOKE_SPREAD_PROBABILITY = 0.55
MAX_TICKS = 50  # Increased from 40

def create_structured_building(num_rooms=12, hallway_segments=6, seed=42):
    """Creates a building graph with rooms, hallways, and exits"""
    random.seed(seed)
    G = nx.Graph()
    
    # Create hallway spine
    for i in range(hallway_segments):
        hallway_id = f"H{i}"
        G.add_node(hallway_id, type='hallway', fire=False, smoke=False, length=2)
        if i > 0:
            G.add_edge(f"H{i}", f"H{i-1}")
    
    # Create rooms connected to hallways
    for i in range(num_rooms):
        room_id = f"R{i}"
        people = []
        if random.random() > 0.4:
            num_people = random.randint(1, 3)
            for _ in range(num_people):
                people.append({'hp': 100})
        G.add_node(room_id, type='room', fire=False, smoke=False, people=people, length=4)
        hallway_connection = f"H{random.randint(0, hallway_segments-1)}"
        G.add_edge(room_id, hallway_connection)
    
    # Add exits
    G.add_node("Exit-1", type='exit', fire=False, smoke=False, length=1)
    G.add_node("Exit-2", type='exit', fire=False, smoke=False, length=1)
    G.add_edge("Exit-1", "H0")
    G.add_edge("Exit-2", f"H{hallway_segments-1}")
    
    # Select random room for fire start
    room_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'room']
    if room_nodes:
        G.graph['fire_start_node'] = random.choice(room_nodes)
    return G

def update_simulation_step(G):
    """Spreads fire/smoke and damages people"""
    next_G = copy.deepcopy(G)
    nodes_on_fire = {n for n, d in G.nodes(data=True) if d['fire']}
    nodes_with_smoke = {n for n, d in G.nodes(data=True) if d['smoke']}
    
    # Fire spread
    for node in nodes_on_fire:
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor]['type'] != 'exit' and not G.nodes[neighbor]['fire']:
                if random.random() < FIRE_SPREAD_PROBABILITY:
                    next_G.nodes[neighbor]['fire'] = True
    
    # Smoke spread
    for node in nodes_on_fire.union(nodes_with_smoke):
        for neighbor in G.neighbors(node):
            if G.nodes[neighbor]['type'] != 'exit' and not G.nodes[neighbor]['smoke']:
                if random.random() < SMOKE_SPREAD_PROBABILITY:
                    next_G.nodes[neighbor]['smoke'] = True
    
    # Damage people
    for node_id, data in next_G.nodes(data=True):
        if data['type'] == 'room' and data.get('people'):
            for person in data['people']:
                if person['hp'] > 0:
                    if data['fire']:
                        person['hp'] -= 25
                    elif data['smoke']:
                        person['hp'] -= 10
                    person['hp'] = max(0, person['hp'])
    return next_G

class RescueEnv(gym.Env):
    """Gymnasium environment for fire rescue simulation"""
    metadata = {"render_modes": []}
    
    def __init__(self, num_rooms=12, hallway_segments=6, seed=42):
        super().__init__()
        self.rng = random.Random(seed)
        self.num_rooms = num_rooms
        self.hallway_segments = hallway_segments
        self.seed_val = seed
        self.G = create_structured_building(self.num_rooms, self.hallway_segments, seed=self.seed_val)
        self.nodes = list(self.G.nodes())
        self.N = len(self.nodes)
        self.M = self.N + 2  # N moves + assist + wait
        
        # Node features: [room/hall/exit (3), fire, smoke, length, people_count, avg_hp, agent_here, distance_to_fire]
        self.F = 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # Added distance_to_fire
        
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0.0, high=1.0, shape=(self.N, self.F), dtype=np.float32),
            "adj": spaces.Box(low=0, high=1, shape=(self.N, self.N), dtype=np.int8),
            "mask": spaces.Box(low=0, high=1, shape=(self.M,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(self.M)
        
        self.tick = 0
        self.agent_pos = 0
        self.total_rescued = 0
        self._reset_graph()
    
    def _reset_graph(self):
        self.G = create_structured_building(self.num_rooms, self.hallway_segments,
                                            seed=self.rng.randint(0, 1_000_000))
        if 'fire_start_node' in self.G.graph:
            start_node = self.G.graph['fire_start_node']
            self.G.nodes[start_node]['fire'] = True
        self.nodes = list(self.G.nodes())
        assert len(self.nodes) == self.N
        start_name = "Exit-1" if "Exit-1" in self.nodes else "H0"
        self.agent_pos = self.nodes.index(start_name) if start_name in self.nodes else 0
        self.tick = 0
        self.total_rescued = 0
    
    def _node_type_onehot(self, ntype: str):
        if ntype == 'room': return [1,0,0]
        if ntype == 'hallway': return [0,1,0]
        return [0,0,1]
    
    def _count_alive(self) -> int:
        alive = 0
        for _, d in self.G.nodes(data=True):
            if d.get('type') == 'room' and d.get('people'):
                for p in d['people']:
                    if p['hp'] > 0: alive += 1
        return alive
    
    def _count_critical(self) -> int:
        """Count people with HP < 30 (critical condition)"""
        critical = 0
        for _, d in self.G.nodes(data=True):
            if d.get('type') == 'room' and d.get('people'):
                for p in d['people']:
                    if 0 < p['hp'] < 30: critical += 1
        return critical
    
    def _deaths_this_step(self, prevG) -> int:
        deaths = 0
        for n in self.nodes:
            d0 = prevG.nodes[n]
            d1 = self.G.nodes[n]
            if d1.get('type') == 'room' and d1.get('people'):
                for i in range(len(d1['people'])):
                    hp1 = d1['people'][i]['hp']
                    hp0 = d0['people'][i]['hp'] if i < len(d0.get('people', [])) else hp1
                    if hp0 > 0 and hp1 == 0:
                        deaths += 1
        return deaths
    
    def _assist_here(self) -> Tuple[bool, int]:
        """Assist people in current location. Returns (success, num_helped)"""
        node_name = self.nodes[self.agent_pos]
        d = self.G.nodes[node_name]
        if d.get('type') == 'room' and d.get('people'):
            alive_people = [p for p in d['people'] if p['hp'] > 0]
            if not alive_people: return False, 0
            # Help the most critical person
            target = min(alive_people, key=lambda p: p['hp'])
            old_hp = target['hp']
            target['hp'] = min(100, target['hp'] + 30)  # Increased from 20
            self.total_rescued += 1
            return True, 1
        return False, 0
    
    def _get_distance_to_nearest_fire(self, node_idx: int) -> float:
        """Get shortest path distance to nearest fire node"""
        node_name = self.nodes[node_idx]
        fire_nodes = [n for n, d in self.G.nodes(data=True) if d['fire']]
        if not fire_nodes:
            return 1.0  # No fire yet
        min_dist = float('inf')
        for fire_node in fire_nodes:
            try:
                dist = nx.shortest_path_length(self.G, node_name, fire_node)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                pass
        return min(1.0, min_dist / 10.0)  # Normalized
    
    def _make_adj(self) -> np.ndarray:
        A = np.zeros((self.N, self.N), dtype=np.int8)
        for i, ni in enumerate(self.nodes):
            for j, nj in enumerate(self.nodes):
                if self.G.has_edge(ni, nj):
                    A[i, j] = 1
        return A
    
    def _make_features(self) -> np.ndarray:
        X = np.zeros((self.N, self.F), dtype=np.float32)
        for i, n in enumerate(self.nodes):
            d = self.G.nodes[n]
            t = self._node_type_onehot(d['type'])
            fire = float(d['fire'])
            smoke = float(d['smoke'])
            length_norm = float(d.get('length', 1))/10.0
            people = d.get('people', [])
            cnt = len([p for p in people if p.get('hp',0)>0])
            avg = np.mean([p['hp'] for p in people if p.get('hp',0)>0]) if cnt>0 else 0.0
            people_cnt_norm = min(1.0, cnt/3.0)
            avg_hp_norm = avg/100.0
            agent_here = 1.0 if i==self.agent_pos else 0.0
            dist_to_fire = self._get_distance_to_nearest_fire(i)
            
            X[i] = np.array(t + [fire, smoke, length_norm, people_cnt_norm, 
                                 avg_hp_norm, agent_here, dist_to_fire], dtype=np.float32)
        return X
    
    def _make_mask(self) -> np.ndarray:
        mask = np.zeros((self.M,), dtype=np.int8)
        agent_node = self.nodes[self.agent_pos]
        neigh = set(self.G.neighbors(agent_node))
        for j, node in enumerate(self.nodes):
            if node in neigh: mask[j] = 1
        d = self.G.nodes[agent_node]
        can_assist = d.get('type')=='room' and d.get('people') and any(p['hp']>0 for p in d['people'])
        mask[self.N] = 1 if can_assist else 0
        mask[self.N+1] = 1
        return mask
    
    def _get_obs(self) -> Dict[str, Any]:
        return {"x": self._make_features(), "adj": self._make_adj(), "mask": self._make_mask()}
    
    def reset(self, seed=None, options=None):
        if seed is not None: self.rng.seed(seed)
        self._reset_graph()
        return self._get_obs(), {}
    
    def step(self, action: int):
        assert self.action_space.contains(action)
        prevG = copy.deepcopy(self.G)
        reward = 0.0
        
        # Action execution
        if action < self.N:  # Move action
            target = action
            agent_node = self.nodes[self.agent_pos]
            target_node = self.nodes[target]
            if self.G.has_edge(agent_node, target_node):
                self.agent_pos = target
                # Small reward for moving towards people in need
                target_data = self.G.nodes[target_node]
                if target_data.get('type') == 'room' and target_data.get('people'):
                    alive_cnt = len([p for p in target_data['people'] if p['hp'] > 0])
                    reward += 0.1 * alive_cnt
            else:
                reward -= 0.5  # Penalty for invalid move
        
        elif action == self.N:  # Assist action
            success, num_helped = self._assist_here()
            if success:
                reward += 3.0  # Increased reward for successful rescue
                critical = self._count_critical()
                reward += 1.0 if critical > 0 else 0.5  # Extra for helping critical people
            else:
                reward -= 0.5  # Penalty for failed assist
        
        else:  # Wait action
            reward -= 0.1  # Small penalty for waiting
        
        # Simulate environment
        self.G = update_simulation_step(self.G)
        self.tick += 1
        
        # Reward shaping
        alive = self._count_alive()
        reward += 0.2 * alive  # Reward for keeping people alive
        
        deaths = self._deaths_this_step(prevG)
        reward -= 5.0 * deaths  # Heavy penalty for deaths
        
        # Location penalties
        here = self.nodes[self.agent_pos]
        if self.G.nodes[here]['fire']: 
            reward -= 1.0
        elif self.G.nodes[here]['smoke']: 
            reward -= 0.3
        
        # Time penalty (encourage efficiency)
        reward -= 0.05
        
        terminated = (alive == 0)
        truncated = (self.tick >= MAX_TICKS)
        
        # Bonus for high survival rate at end
        if terminated or truncated:
            total_initial = sum(len(d.get('people', [])) for _, d in prevG.nodes(data=True) if d.get('type') == 'room')
            if total_initial > 0:
                survival_rate = alive / total_initial
                reward += 10.0 * survival_rate  # Bonus for high survival
        
        return self._get_obs(), reward, terminated, truncated, {
            "alive": alive, 
            "deaths": deaths, 
            "tick": self.tick,
            "rescued": self.total_rescued,
            "critical": self._count_critical()
        }


# ============================================================================
# GRAPH ATTENTION NETWORK (Enhanced with Residual Connections)
# ============================================================================

class GATLayer(nn.Module):
    """Graph Attention Layer with multi-head attention"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 alpha: float = 0.2, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, F = x.shape
        h = self.W(x)  # [B, N, out_dim]
        h = h.view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D]
        
        # Self-loops
        I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        A = (adj > 0).float() + I
        A = (A > 0).float()
        
        # Compute attention per head
        h_i = h.unsqueeze(2).expand(B, N, N, self.num_heads, self.head_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim)
        
        e_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, H, 2D]
        e = self.leakyrelu(self.a(e_input)).squeeze(-1)  # [B, N, N, H]
        
        # Mask and softmax
        A_expanded = A.unsqueeze(-1).expand(B, N, N, self.num_heads)
        e = e.masked_fill(A_expanded == 0, -1e9)
        alpha = torch.softmax(e, dim=2)  # [B, N, N, H]
        alpha = self.dropout(alpha)
        
        # Aggregate
        out = torch.einsum('bnjh,bnhd->bnhd', alpha, h)  # [B, N, H, D]
        out = out.reshape(B, N, self.out_dim)  # [B, N, out_dim]
        return out

class GATEncoder(nn.Module):
    """Multi-layer GAT encoder with residual connections"""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers
        
        # First layer
        self.layers.append(GATLayer(in_dim, hidden, num_heads=4))
        self.norms.append(nn.LayerNorm(hidden))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden, hidden, num_heads=4))
            self.norms.append(nn.LayerNorm(hidden))
        
        # Last layer
        self.layers.append(GATLayer(hidden, out_dim, num_heads=4))
        self.norms.append(nn.LayerNorm(out_dim))
        
        self.act = nn.ELU()
        
    def forward(self, x, adj):
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new = layer(h, adj)
            h_new = norm(h_new)
            if i < self.num_layers - 1:
                h_new = self.act(h_new)
                # Residual connection (if dimensions match)
                if h.shape[-1] == h_new.shape[-1]:
                    h_new = h_new + h
            h = h_new
        return h

class GraphActorCritic(nn.Module):
    """Actor-Critic with GAT encoder"""
    def __init__(self, node_feat_dim: int, n_actions: int, hidden=128, out_dim=128):
        super().__init__()
        self.encoder = GATEncoder(node_feat_dim, hidden, out_dim, num_layers=3)
        
        # Actor (policy) head
        self.pi = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        
        # Critic (value) head
        self.v = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, adj):
        h = self.encoder(x, adj)  # [B, N, H]
        g = h.mean(dim=1)  # [B, H] (mean-pool over nodes)
        logits = self.pi(g)  # [B, A]
        value = self.v(g).squeeze(-1)  # [B]
        return logits, value
    
    def act(self, x, adj, mask):
        logits, value = self.forward(x, adj)
        masked_logits = logits + torch.log(mask.clamp(min=1e-8))
        dist = torch.distributions.Categorical(logits=masked_logits)
        a = dist.sample()
        logprob = dist.log_prob(a)
        entropy = dist.entropy()
        return a, logprob, entropy, value
    
    def greedy(self, x, adj, mask):
        logits, value = self.forward(x, adj)
        masked_logits = logits + torch.log(mask.clamp(min=1e-8))
        a = masked_logits.argmax(dim=-1)
        return a, value
    
    def evaluate(self, x, adj, mask, actions):
        logits, value = self.forward(x, adj)
        masked_logits = logits + torch.log(mask.clamp(min=1e-8))
        dist = torch.distributions.Categorical(logits=masked_logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


# ============================================================================
# PPO TRAINING with VISUALIZATION
# ============================================================================

def compute_gae(rewards, dones, values, last_value, gamma, lam):
    """Compute GAE advantages"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterm = 1.0 - dones[t]
            next_v = last_value
        else:
            next_nonterm = 1.0 - dones[t+1]
            next_v = values[t+1]
        delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
        last_gae = delta + gamma * lam * next_nonterm * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret

def evaluate(env: RescueEnv, model: GraphActorCritic, device: torch.device, episodes: int = 10):
    """Evaluate model performance"""
    model.eval()
    metrics = {
        'return': [], 'alive': [], 'deaths': [], 'ticks': [], 
        'rescued': [], 'survival_rate': []
    }
    
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            done = False
            ep_deaths = 0
            ticks = 0
            initial_people = sum(len(d.get('people', [])) for _, d in env.G.nodes(data=True) 
                               if d.get('type') == 'room')
            
            while not done:
                xt = torch.as_tensor(obs["x"], dtype=torch.float32, device=device).unsqueeze(0)
                ad = torch.as_tensor(obs["adj"], dtype=torch.float32, device=device).unsqueeze(0)
                mk = torch.as_tensor(obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)
                a, _ = model.greedy(xt, ad, mk)
                obs, r, term, trunc, info = env.step(int(a.item()))
                ep_return += r
                done = term or trunc
                ep_deaths += info.get("deaths", 0)
                ticks += 1
            
            metrics['return'].append(ep_return)
            metrics['alive'].append(info.get("alive", 0))
            metrics['deaths'].append(ep_deaths)
            metrics['ticks'].append(ticks)
            metrics['rescued'].append(info.get("rescued", 0))
            if initial_people > 0:
                metrics['survival_rate'].append(info.get("alive", 0) / initial_people)
            else:
                metrics['survival_rate'].append(0.0)
    
    model.train()
    return {k: np.mean(v) for k, v in metrics.items()}

def ppo_train(
    total_timesteps=1_000_000,  # 10x more than before!
    num_steps=1024,  # Increased rollout buffer
    minibatch_size=128,  # Larger minibatches
    update_epochs=15,  # More epochs per update
    gamma=0.99,
    lam=0.95,
    clip_coef=0.2,
    ent_coef=0.02,  # Increased entropy for exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    lr=1e-4,  # Lower LR for stability
    seed=42,
    device="cpu",
    eval_every_updates=10,
    eval_episodes=20,
    save_dir="rescue_results"
):
    """Train PPO agent with comprehensive logging"""
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize environment and model
    env = RescueEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    N, F = obs["x"].shape
    A = env.action_space.n
    
    print(f"Environment: {N} nodes, {F} features per node, {A} actions")
    print(f"Training for {total_timesteps:,} timesteps")
    
    model = GraphActorCritic(node_feat_dim=F, n_actions=A, hidden=128, out_dim=128).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_timesteps // num_steps)
    
    num_updates = total_timesteps // num_steps
    global_step = 0
    
    # Buffers
    x_buf = np.zeros((num_steps, N, F), np.float32)
    adj_buf = np.zeros((num_steps, N, N), np.float32)
    m_buf = np.zeros((num_steps, A), np.float32)
    a_buf = np.zeros((num_steps,), np.int64)
    lp_buf = np.zeros((num_steps,), np.float32)
    r_buf = np.zeros((num_steps,), np.float32)
    d_buf = np.zeros((num_steps,), np.float32)
    v_buf = np.zeros((num_steps,), np.float32)
    
    # Metrics tracking
    training_metrics = {
        'update': [], 'steps': [], 'avg_return': [], 'avg_alive': [], 
        'avg_deaths': [], 'avg_survival_rate': [], 'avg_rescued': [],
        'policy_loss': [], 'value_loss': [], 'entropy': [], 'lr': []
    }
    
    best_survival_rate = -1e9
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    for upd in range(1, num_updates + 1):
        obs, _ = env.reset(seed=seed + upd)
        
        # Collect rollout
        for t in range(num_steps):
            global_step += 1
            x_buf[t] = obs["x"]
            adj_buf[t] = obs["adj"]
            m_buf[t] = obs["mask"]
            
            xt = torch.as_tensor(obs["x"], dtype=torch.float32, device=device).unsqueeze(0)
            ad = torch.as_tensor(obs["adj"], dtype=torch.float32, device=device).unsqueeze(0)
            mk = torch.as_tensor(obs["mask"], dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                a, logp, ent, v = model.act(xt, ad, mk)
            
            a_buf[t] = a.item()
            lp_buf[t] = logp.item()
            v_buf[t] = v.item()
            
            obs, r, term, trunc, info = env.step(a.item())
            r_buf[t] = r
            d_buf[t] = float(term or trunc)
            
            if term or trunc:
                obs, _ = env.reset()
        
        # Bootstrap
        xt = torch.as_tensor(obs["x"], dtype=torch.float32, device=device).unsqueeze(0)
        ad = torch.as_tensor(obs["adj"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, last_v = model.forward(xt, ad)
            last_v = last_v.item()
        
        # Compute advantages
        adv, ret = compute_gae(r_buf, d_buf, v_buf, last_v, gamma, lam)
        
        # Convert to tensors
        x_t = torch.as_tensor(x_buf, dtype=torch.float32, device=device)
        adj_t = torch.as_tensor(adj_buf, dtype=torch.float32, device=device)
        m_t = torch.as_tensor(m_buf, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(a_buf, dtype=torch.long, device=device)
        oldlp = torch.as_tensor(lp_buf, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
        v_t = torch.as_tensor(v_buf, dtype=torch.float32, device=device)
        
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        # PPO update
        idx = np.arange(num_steps)
        pg_losses, v_losses, entropies = [], [], []
        
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for s in range(0, num_steps, minibatch_size):
                mb = idx[s:s + minibatch_size]
                mb_x, mb_adj, mb_m = x_t[mb], adj_t[mb], m_t[mb]
                mb_a, mb_oldlp = a_t[mb], oldlp[mb]
                mb_adv, mb_ret, mb_v = adv_t[mb], ret_t[mb], v_t[mb]
                
                new_lp, ent, new_v = model.evaluate(mb_x, mb_adj, mb_m, mb_a)
                ratio = (new_lp - mb_oldlp).exp()
                
                # Policy loss
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                
                # Value loss (clipped)
                v_un = (new_v - mb_ret) ** 2
                v_cl = (mb_v + torch.clamp(new_v - mb_v, -clip_coef, clip_coef) - mb_ret) ** 2
                v_loss = 0.5 * torch.max(v_un, v_cl).mean()
                
                # Total loss
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent.mean()
                
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent.mean().item())
        
        scheduler.step()
        
        # Evaluation
        if upd % eval_every_updates == 0 or upd == num_updates:
            eval_metrics = evaluate(env, model, device, episodes=eval_episodes)
            
            training_metrics['update'].append(upd)
            training_metrics['steps'].append(global_step)
            training_metrics['avg_return'].append(eval_metrics['return'])
            training_metrics['avg_alive'].append(eval_metrics['alive'])
            training_metrics['avg_deaths'].append(eval_metrics['deaths'])
            training_metrics['avg_survival_rate'].append(eval_metrics['survival_rate'])
            training_metrics['avg_rescued'].append(eval_metrics['rescued'])
            training_metrics['policy_loss'].append(np.mean(pg_losses))
            training_metrics['value_loss'].append(np.mean(v_losses))
            training_metrics['entropy'].append(np.mean(entropies))
            training_metrics['lr'].append(scheduler.get_last_lr()[0])
            
            print(f"[Update {upd:4d}/{num_updates}] Steps: {global_step:7d} | "
                  f"Return: {eval_metrics['return']:7.2f} | "
                  f"Survival: {eval_metrics['survival_rate']*100:5.1f}% | "
                  f"Alive: {eval_metrics['alive']:4.1f} | "
                  f"Deaths: {eval_metrics['deaths']:4.1f} | "
                  f"Rescued: {eval_metrics['rescued']:4.1f}")
            
            # Save best model
            if eval_metrics['survival_rate'] > best_survival_rate:
                best_survival_rate = eval_metrics['survival_rate']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'update': upd,
                    'metrics': eval_metrics
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  >>> New best model saved! Survival rate: {best_survival_rate*100:.1f}%")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    
    # Save training log
    with open(os.path.join(save_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(training_metrics.keys())
        for i in range(len(training_metrics['update'])):
            writer.writerow([training_metrics[k][i] for k in training_metrics.keys()])
    
    # Plot results
    plot_training_results(training_metrics, save_dir)
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE! Best survival rate: {best_survival_rate*100:.1f}%")
    print(f"Results saved to: {save_dir}/")
    print("="*80)
    
    return model, training_metrics

def plot_training_results(metrics, save_dir):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('PPO+GAT Rescue Training Results', fontsize=16, fontweight='bold')
    
    steps = metrics['steps']
    
    # Row 1: Performance metrics
    axes[0, 0].plot(steps, metrics['avg_return'], color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Average Return')
    axes[0, 0].set_title('Episode Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(steps, [s*100 for s in metrics['avg_survival_rate']], 
                    color='green', linewidth=2)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Survival Rate (%)')
    axes[0, 1].set_title('Survival Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(steps, metrics['avg_alive'], color='cyan', linewidth=2, label='Alive')
    axes[0, 2].plot(steps, metrics['avg_deaths'], color='red', linewidth=2, label='Deaths')
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('People Alive vs Deaths')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Learning metrics
    axes[1, 0].plot(steps, metrics['policy_loss'], color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(steps, metrics['value_loss'], color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(steps, metrics['entropy'], color='magenta', linewidth=2)
    axes[1, 2].set_xlabel('Steps')
    axes[1, 2].set_ylabel('Entropy')
    axes[1, 2].set_title('Policy Entropy (Exploration)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Additional metrics
    axes[2, 0].plot(steps, metrics['avg_rescued'], color='gold', linewidth=2)
    axes[2, 0].set_xlabel('Steps')
    axes[2, 0].set_ylabel('People Rescued')
    axes[2, 0].set_title('Rescue Actions Per Episode')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(steps, metrics['lr'], color='brown', linewidth=2)
    axes[2, 1].set_xlabel('Steps')
    axes[2, 1].set_ylabel('Learning Rate')
    axes[2, 1].set_title('Learning Rate Schedule')
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[2, 2].axis('off')
    final_metrics = f"""
    FINAL METRICS
    {'='*30}
    Avg Return: {metrics['avg_return'][-1]:.2f}
    Survival Rate: {metrics['avg_survival_rate'][-1]*100:.1f}%
    Avg Alive: {metrics['avg_alive'][-1]:.1f}
    Avg Deaths: {metrics['avg_deaths'][-1]:.1f}
    Avg Rescued: {metrics['avg_rescued'][-1]:.1f}
    
    BEST METRICS
    {'='*30}
    Best Return: {max(metrics['avg_return']):.2f}
    Best Survival: {max(metrics['avg_survival_rate'])*100:.1f}%
    """
    axes[2, 2].text(0.1, 0.5, final_metrics, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=200, bbox_inches='tight')
    print(f"Training plot saved to: {save_dir}/training_results.png")

def main():
    parser = argparse.ArgumentParser(description="PPO+GAT for Fire Rescue")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, 
                       help="Total training timesteps (default: 1M)")
    parser.add_argument("--num-steps", type=int, default=1024,
                       help="Rollout buffer size")
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="rescue_results")
    args = parser.parse_args()
    
    ppo_train(
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        lam=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        eval_every_updates=args.eval_every_updates,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()