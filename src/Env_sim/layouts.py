from .env import BuildingFireEnvironment

# Build the layout

def build_standard_office_layout() -> BuildingFireEnvironment:
    """
    Build the standard one-floor office layout as specified:
    - Central hallway divided into 3 segments (for granularity)
    - 3 rooms on top side of hallway
    - 3 rooms on bottom side of hallway
    - 2 exits (left and right ends of hallway)
    
    Returns:
        Initialized BuildingFireEnvironment
    """
    env = BuildingFireEnvironment()
    
    # Central Hallway (3 segments)
  
    for i in range(3):
        env.add_node(
            nid=f"H{i}",
            ntype="hall",
            length=6.0,        # 6 meters per segment
            area=12.0,         # 2m wide × 6m long
            floor=0
        )
    
    # Connect hallway segments
    env.add_edge("H0", "H1", length=6.0, width=2.0, door=False)  # Open hallway
    env.add_edge("H1", "H2", length=6.0, width=2.0, door=False)
    
    # ROOMS - TOP ROW (3 rooms along top of hallway)

    for i in range(3):
        env.add_node(
            nid=f"RT{i}",
            ntype="room",
            length=4.0,        # 4m × 4m room
            area=16.0,
            floor=0,
        )

        # Connect to corresponding hallway segment

        env.add_edge(
            f"RT{i}", f"H{i}",
            length=1.0,        # 1m doorway depth
            width=0.9,         # Standard door width
            door=True
        )
    

    #  3 rooms along bottom of hallway

    for i in range(3):
        env.add_node(
            nid=f"RB{i}",
            ntype="room",
            length=4.0,
            area=16.0,
            floor=0,   
        )
        # Connect to corresponding hallway segment
        env.add_edge(
            f"RB{i}", f"H{i}",
            length=1.0,
            width=0.9,
            door=True
        )
    

    # exists - left and right ends of hallway
    env.add_node(
        nid="EXIT_LEFT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0
    )
    env.add_node(
        nid="EXIT_RIGHT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0
    )
    
    # Connect exits to hallway ends
    env.add_edge("EXIT_LEFT", "H0", length=1.0, width=1.2, door=True)
    env.add_edge("EXIT_RIGHT", "H2", length=1.0, width=1.2, door=True)
    

    # populated with people (Hidden until agents search)
    
    env.spawn_person("RT0", age=34, mobility="adult", hp=95.0)
    env.spawn_person("RT2", age=55, mobility="limited", hp=90.0)
    env.spawn_person("RB0", age=28, mobility="adult", hp=100.0)
    env.spawn_person("RB2", age=6, mobility="child", hp=100.0) 
    env.spawn_person("RB1", age=32, mobility="adult", hp=100.0)
    env.spawn_person("RB1", age=41, mobility="adult", hp=97.0)
    

    # place agents at exits

    env.place_agent(agent_id=0, node_id="EXIT_LEFT")
    env.place_agent(agent_id=1, node_id="EXIT_RIGHT")
    
    return env


def build_two_floor_warehouse() -> BuildingFireEnvironment:
    """
    Additional layout: compact two-floor warehouse with stairs and two exits.
    - Floor 0: corridor C0 - C2 with two storage rooms S0,S1; EXIT_F0
    - Floor 1: corridor C3 - C5 with two storage rooms S2,S3; EXIT_F1
    - Stairs connect C1 <-> C4 (bidirectional)
    """
    env = BuildingFireEnvironment()

    # Floor 0 corridors
    for i in range(3):
        env.add_node(nid=f"C{i}", ntype="hall", length=7.0, area=14.0, floor=0)
    env.add_edge("C0", "C1", length=7.0, width=2.2, door=False)
    env.add_edge("C1", "C2", length=7.0, width=2.2, door=False)

    # Floor 0 rooms
    env.add_node("S0", "room", length=5.0, area=20.0, floor=0)
    env.add_node("S1", "room", length=5.0, area=20.0, floor=0)
    env.add_edge("S0", "C0", length=1.0, width=1.0, door=True)
    env.add_edge("S1", "C2", length=1.0, width=1.0, door=True)

    # Floor 0 exit
    env.add_node("EXIT_F0", "exit", length=2.0, area=4.0, floor=0)
    env.add_edge("EXIT_F0", "C0", length=1.0, width=1.4, door=True)

    # Floor 1 corridors
    for i in range(3, 6):
        env.add_node(nid=f"C{i}", ntype="hall", length=7.0, area=14.0, floor=1)
    env.add_edge("C3", "C4", length=7.0, width=2.2, door=False)
    env.add_edge("C4", "C5", length=7.0, width=2.2, door=False)

    # Floor 1 rooms
    env.add_node("S2", "room", length=5.0, area=20.0, floor=1)
    env.add_node("S3", "room", length=5.0, area=20.0, floor=1)
    env.add_edge("S2", "C3", length=1.0, width=1.0, door=True)
    env.add_edge("S3", "C5", length=1.0, width=1.0, door=True)

    # Floor 1 exit
    env.add_node("EXIT_F1", "exit", length=2.0, area=4.0, floor=1)
    env.add_edge("EXIT_F1", "C5", length=1.0, width=1.4, door=True)

    # Stairs (bidirectional, with slope)
    env.add_edge("C1", "C4", length=4.0, width=1.5, slope=30.0, door=False)

    # People
    env.spawn_person("S0", age=30, mobility="adult", hp=100.0)
    env.spawn_person("S1", age=62, mobility="limited", hp=95.0)
    env.spawn_person("S2", age=25, mobility="adult", hp=100.0)
    env.spawn_person("S3", age=12, mobility="child", hp=100.0)

    # Agents
    env.place_agent(0, "EXIT_F0")
    env.place_agent(1, "EXIT_F1")

    return env
