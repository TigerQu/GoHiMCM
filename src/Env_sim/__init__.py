"""
Building Fire Environment Package
==================================

A reinforcement learning environment for simulating firefighter building sweeps
during fire emergencies. This package models:

1. Building topology (rooms, hallways, exits)
2. Fire and smoke dynamics
3. Civilian behavior and evacuation
4. Firefighter agents with realistic movement and search

The environment follows OpenAI Gym/Gymnasium interface conventions for use
with RL algorithms like PPO, A2C, etc.

Usage:
    from environment import build_standard_office_layout
    
    env = build_standard_office_layout()
    obs = env.reset(fire_node="RT0")
    
    done = False
    while not done:
        actions = {0: "search", 1: "move_H1"}
        obs, reward, done, info = env.do_action(actions)
"""

# Main environment class
from .env import BuildingFireEnvironment

# Pre-built layouts for testing and benchmarking
from .layouts import build_standard_office_layout, build_two_floor_warehouse

# Configuration constants
from .config import DEFAULT_CONFIG, FEATURE_DIM, NODE_TYPES

# Entity data classes (for type hints and inspection)
from .entities import Person, NodeMeta, EdgeMeta, Agent

# Version info
__version__ = "1.0.0"
__author__ = "Fire Evacuation Research Team"

# Public API
__all__ = [
    "BuildingFireEnvironment",
    "build_standard_office_layout",
    "build_two_floor_warehouse",
    "DEFAULT_CONFIG",
    "FEATURE_DIM",
    "NODE_TYPES",
    "Person",
    "NodeMeta",
    "EdgeMeta",
    "Agent",
]