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


def build_babycare_layout() -> BuildingFireEnvironment:
    """
    Build a multi-floor (5 floors) babycare center layout.

    Features per floor:
    - A central corridor with multiple segments
    - Several nurseries (rooms for infants)
    - Nurse stations/rooms
    - Play area and sleeping room
    - Small kitchen / service room
    - Elevator and stairs connecting floors
    - Ground-floor emergency exits

    Returns:
        Initialized BuildingFireEnvironment for babycare center
    """
    env = BuildingFireEnvironment()

    FLOORS = 3  # reduced to 3 floors for clearer visualization
    # For each floor create 3 corridor segments (C0, C1, C2), 6 nurseries, 1 nurse room, 2 play areas, 1 kitchen, 1 storage room
    for f in range(FLOORS):
        # Corridors
        for i in range(3):
            env.add_node(nid=f"F{f}_C{i}", ntype="hall", length=8.0, area=16.0, floor=f)

        # Connect corridor segments on same floor
        env.add_edge(f"F{f}_C0", f"F{f}_C1", length=8.0, width=2.2, door=False)
        env.add_edge(f"F{f}_C1", f"F{f}_C2", length=8.0, width=2.2, door=False)

        # Nurseries (6 per floor - 2 per corridor)
        for r in range(6):
            nid = f"F{f}_NUR{r}"
            corridor = f"F{f}_C{r // 2}"  # 0-1 → C0, 2-3 → C1, 4-5 → C2
            env.add_node(nid=nid, ntype="room", length=5.0, area=20.0, floor=f)
            env.add_edge(nid, corridor, length=1.0, width=1.0, door=True)

        # Nurse room (one per floor, NOT connected to corridor)
        nurse_nid = f"F{f}_NURSE"
        env.add_node(nid=nurse_nid, ntype="room", length=4.0, area=12.0, floor=f)
        # Nurse station connects directly to all rooms

        # Play areas (2 per floor)
        for p in range(2):
            play_nid = f"F{f}_PLAY{p}"
            corridor = f"F{f}_C{2 if p == 0 else 0}"  # PLAY0 → C2, PLAY1 → C0
            env.add_node(nid=play_nid, ntype="room", length=6.0, area=24.0, floor=f)
            env.add_edge(play_nid, corridor, length=1.0, width=1.2, door=True)

        # Kitchen / service
        kit_nid = f"F{f}_KITCHEN"
        env.add_node(nid=kit_nid, ntype="room", length=4.0, area=10.0, floor=f)
        env.add_edge(kit_nid, f"F{f}_C0", length=1.0, width=1.0, door=True)

        # Storage room
        storage_nid = f"F{f}_STORAGE"
        env.add_node(nid=storage_nid, ntype="room", length=4.0, area=10.0, floor=f)
        env.add_edge(storage_nid, f"F{f}_C2", length=1.0, width=1.0, door=True)

        # Connect nurse station directly to all rooms (but not to corridors)
        for r in range(6):
            env.add_edge(nurse_nid, f"F{f}_NUR{r}", length=2.0, width=1.0, door=True)
        for p in range(2):
            env.add_edge(nurse_nid, f"F{f}_PLAY{p}", length=2.0, width=1.0, door=True)
        env.add_edge(nurse_nid, kit_nid, length=2.0, width=1.0, door=True)
        env.add_edge(nurse_nid, storage_nid, length=2.0, width=1.0, door=True)

    # Connect vertical circulation: stairs between floors
    for f in range(FLOORS - 1):
        # Stairs between central corridors (C1 of each floor)
        env.add_edge(f"F{f}_C1", f"F{f+1}_C1", length=4.0, width=1.6, slope=35.0, door=False)

    # Ground floor exits (connected to corridor segments)
    env.add_node(nid="EXIT_G_LEFT", ntype="exit", length=3.0, area=6.0, floor=0)
    env.add_node(nid="EXIT_G_RIGHT", ntype="exit", length=3.0, area=6.0, floor=0)
    env.add_edge("EXIT_G_LEFT", "F0_C0", length=1.0, width=1.6, door=True)
    env.add_edge("EXIT_G_RIGHT", "F0_C2", length=1.0, width=1.6, door=True)

    # Rooftop emergency access on top floor
    env.add_node(nid=f"F{FLOORS-1}_ROOF_EXIT", ntype="exit", length=2.0, area=4.0, floor=FLOORS-1)
    env.add_edge(f"F{FLOORS-1}_ROOF_EXIT", f"F{FLOORS-1}_C2", length=1.0, width=1.0, door=True)

    # Populate with staff and infants for realism
    # Place nurses near nurse rooms and some infants in nurseries
    for f in range(FLOORS):
        # One nurse per floor
        env.spawn_person(f"F{f}_NURSE", age=36, mobility="staff", hp=100.0)

        # Few infants and caregivers in nurseries
        for r in range(6):
            # caregiver
            env.spawn_person(f"F{f}_NUR{r}", age=28 + r, mobility="adult", hp=100.0)
            # infant (mobility set to 'infant' so environment logic can treat specially)
            env.spawn_person(f"F{f}_NUR{r}", age=1, mobility="infant", hp=100.0)

    # Place a few agents (evacuation managers) on ground-floor exits
    env.place_agent(agent_id=0, node_id="EXIT_G_LEFT")
    env.place_agent(agent_id=1, node_id="EXIT_G_RIGHT")

    return env

def build_two_floor_warehouse() -> BuildingFireEnvironment:
    """
    Single-floor warehouse layout with a grid structure:
    - Grid of hallways (H_{r}_{c})
    - Cargo rooms between hall intersections, each connected to 4 surrounding halls
    - Two exits on opposite sides
    """
    env = BuildingFireEnvironment()

    # Single floor warehouse grid: 4x6 (larger grid for more cargo rooms)
    ROWS = 4   # number of hallway rows
    COLS = 6   # number of hallway columns

    # Create corridor grid nodes (single floor, f=0)
    for r in range(ROWS):
        for c in range(COLS):
            env.add_node(nid=f"H_{r}_{c}", ntype="hall", length=10.0, area=30.0, floor=0)

    # Connect corridor grid horizontally and vertically
    for r in range(ROWS):
        for c in range(COLS):
            if c + 1 < COLS:
                env.add_edge(f"H_{r}_{c}", f"H_{r}_{c+1}", length=8.0, width=3.0, door=False)
            if r + 1 < ROWS:
                env.add_edge(f"H_{r}_{c}", f"H_{r+1}_{c}", length=8.0, width=3.0, door=False)

    # Create cargo shelf rooms placed between corridor intersections so each room connects to four surrounding halls
    # rooms indices go from 0..ROWS-2, 0..COLS-2
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            rid = f"R_{r}_{c}"
            env.add_node(nid=rid, ntype="room", length=6.0, area=40.0, floor=0)
            # connect to four surrounding halls (top-left, top-right, bottom-left, bottom-right)
            env.add_edge(rid, f"H_{r}_{c}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r}_{c+1}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r+1}_{c}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r+1}_{c+1}", length=1.0, width=1.2, door=True)

    # Two exits on opposite sides (left and right)
    env.add_node(nid="EXIT_WH_LEFT", ntype="exit", length=4.0, area=8.0, floor=0)
    env.add_node(nid="EXIT_WH_RIGHT", ntype="exit", length=4.0, area=8.0, floor=0)
    # connect exits to edge halls (bottom left and bottom right)
    env.add_edge("EXIT_WH_LEFT", "H_2_0", length=2.0, width=2.0, door=True)
    env.add_edge("EXIT_WH_RIGHT", f"H_2_{COLS-1}", length=2.0, width=2.0, door=True)

    # Populate with workers (forklift operators) near cargo rooms
    for r in range(0, ROWS - 1, 2):
        for c in range(0, COLS - 1, 2):
            rid = f"R_{r}_{c}"
            env.spawn_person(rid, age=35, mobility="adult", hp=100.0)

    # Agents placed at the two warehouse exits
    env.place_agent(0, "EXIT_WH_LEFT")
    env.place_agent(1, "EXIT_WH_RIGHT")

    return env

