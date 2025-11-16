from .env import BuildingFireEnvironment

# Build the layout

def build_standard_office_layout() -> BuildingFireEnvironment:
    """
    Build the standard one-floor office layout as specified:
    - Central hallway divided into 3 segments (for granularity)
    - 3 rooms on top side of hallway
    - 3 rooms on bottom side of hallway
    - 2 exits (left and right ends of hallway)
    
    NEW: Rooms are tagged with semantic properties:
    - role: "office" (standard workplace)
    - risk_level: "normal" (no special hazards)
    - required_sweeps: 1 (standard thoroughness)
    
    Returns:
        Initialized BuildingFireEnvironment
    """
    env = BuildingFireEnvironment()
    
    # Central Hallway (3 segments) - corridors require minimal sweeping
    for i in range(3):
        env.add_node(
            nid=f"H{i}",
            ntype="hall",
            length=6.0,
            area=12.0,
            floor=0,
            role="corridor",
            risk_level="normal",
            required_sweeps=1
        )
    
    # Connect hallway segments
    env.add_edge("H0", "H1", length=6.0, width=2.0, door=False)
    env.add_edge("H1", "H2", length=6.0, width=2.0, door=False)
    
    # ROOMS - TOP ROW (3 rooms along top of hallway)
    for i in range(3):
        env.add_node(
            nid=f"RT{i}",
            ntype="room",
            length=4.0,
            area=16.0,
            floor=0,
            # NEW: Semantic properties for standard office
            role="office",
            risk_level="normal",
            required_sweeps=1
        )
        
        # Connect to corresponding hallway segment
        env.add_edge(
            f"RT{i}", f"H{i}",
            length=1.0,
            width=0.9,
            door=True
        )
    
    # ROOMS - BOTTOM ROW (3 rooms along bottom of hallway)
    for i in range(3):
        env.add_node(
            nid=f"RB{i}",
            ntype="room",
            length=4.0,
            area=16.0,
            floor=0,
            role="office",
            risk_level="normal",
            required_sweeps=1
        )
        
        # Connect to corresponding hallway segment
        env.add_edge(
            f"RB{i}", f"H{i}",
            length=1.0,
            width=0.9,
            door=True
        )
    
    # Exits - left and right ends of hallway
    env.add_node(
        nid="EXIT_LEFT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_node(
        nid="EXIT_RIGHT",
        ntype="exit",
        length=2.0,
        area=4.0,
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    
    # Connect exits to hallway ends
    env.add_edge("EXIT_LEFT", "H0", length=1.0, width=1.2, door=True)
    env.add_edge("EXIT_RIGHT", "H2", length=1.0, width=1.2, door=True)
    
    # Populate with people (Hidden until agents search)
    env.spawn_person("RT0", age=34, mobility="adult", hp=95.0)
    env.spawn_person("RT2", age=55, mobility="limited", hp=90.0)
    env.spawn_person("RB0", age=28, mobility="adult", hp=100.0)
    env.spawn_person("RB2", age=6, mobility="child", hp=100.0) 
    env.spawn_person("RB1", age=32, mobility="adult", hp=100.0)
    env.spawn_person("RB1", age=41, mobility="adult", hp=97.0)
    
    # Place agents at exits
    env.place_agent(agent_id=0, node_id="EXIT_LEFT")
    env.place_agent(agent_id=1, node_id="EXIT_RIGHT")
    
    return env


def build_babycare_layout() -> BuildingFireEnvironment:
    """
    Build a multi-floor (3 floors) babycare center layout.

    Features per floor:
    - A central corridor with multiple segments
    - Several nurseries (rooms for infants) - HIGH-RISK, require 2 sweeps
    - Nurse stations/rooms
    - Play area and sleeping room
    - Small kitchen / service room
    - Stairs connecting floors
    - Ground-floor emergency exits

    NEW: Nurseries and childcare rooms are marked as high-risk:
    - role: "classroom" or "childcare"
    - risk_level: "high"
    - required_sweeps: 2 (must be verified by 2 different agents)

    Returns:
        Initialized BuildingFireEnvironment for babycare center
    """
    env = BuildingFireEnvironment()

    FLOORS = 3
    
    for f in range(FLOORS):
        # Corridors - standard sweep requirement
        for i in range(3):
            env.add_node(
                nid=f"F{f}_C{i}", 
                ntype="hall", 
                length=8.0, 
                area=16.0, 
                floor=f,
                role="corridor",
                risk_level="normal",
                required_sweeps=1
            )

        # Connect corridor segments on same floor
        env.add_edge(f"F{f}_C0", f"F{f}_C1", length=8.0, width=2.2, door=False)
        env.add_edge(f"F{f}_C1", f"F{f}_C2", length=8.0, width=2.2, door=False)

        # NEW: Nurseries (6 per floor) - HIGH-RISK rooms requiring redundant searches
        for r in range(6):
            nid = f"F{f}_NUR{r}"
            corridor = f"F{f}_C{r // 2}"
            env.add_node(
                nid=nid, 
                ntype="room", 
                length=5.0, 
                area=20.0, 
                floor=f,
                # HIGH-RISK: Infants and young children need redundant verification
                role="classroom",
                risk_level="high",
                required_sweeps=2  # Must be swept by 2 different agents
            )
            env.add_edge(nid, corridor, length=1.0, width=1.0, door=True)

        # Nurse room - staff area, normal risk
        nurse_nid = f"F{f}_NURSE"
        env.add_node(
            nid=nurse_nid, 
            ntype="room", 
            length=4.0, 
            area=12.0, 
            floor=f,
            role="staff_room",
            risk_level="normal",
            required_sweeps=1
        )

        # Play areas (2 per floor) - also high-risk due to children
        for p in range(2):
            play_nid = f"F{f}_PLAY{p}"
            corridor = f"F{f}_C{2 if p == 0 else 0}"
            env.add_node(
                nid=play_nid, 
                ntype="room", 
                length=6.0, 
                area=24.0, 
                floor=f,
                role="playroom",
                risk_level="high",
                required_sweeps=2
            )
            env.add_edge(play_nid, corridor, length=1.0, width=1.2, door=True)

        # Kitchen / service - normal risk
        kit_nid = f"F{f}_KITCHEN"
        env.add_node(
            nid=kit_nid, 
            ntype="room", 
            length=4.0, 
            area=10.0, 
            floor=f,
            role="kitchen",
            risk_level="normal",
            required_sweeps=1
        )
        env.add_edge(kit_nid, f"F{f}_C0", length=1.0, width=1.0, door=True)

        # Storage room - normal risk
        storage_nid = f"F{f}_STORAGE"
        env.add_node(
            nid=storage_nid, 
            ntype="room", 
            length=4.0, 
            area=10.0, 
            floor=f,
            role="storage",
            risk_level="normal",
            required_sweeps=1
        )
        env.add_edge(storage_nid, f"F{f}_C2", length=1.0, width=1.0, door=True)

        # Connect nurse station directly to all rooms for quick access
        for r in range(6):
            env.add_edge(nurse_nid, f"F{f}_NUR{r}", length=2.0, width=1.0, door=True)
        for p in range(2):
            env.add_edge(nurse_nid, f"F{f}_PLAY{p}", length=2.0, width=1.0, door=True)
        env.add_edge(nurse_nid, kit_nid, length=2.0, width=1.0, door=True)
        env.add_edge(nurse_nid, storage_nid, length=2.0, width=1.0, door=True)

    # Connect vertical circulation: stairs between floors
    for f in range(FLOORS - 1):
        env.add_edge(f"F{f}_C1", f"F{f+1}_C1", length=4.0, width=1.6, slope=35.0, door=False)

    # Ground floor exits
    env.add_node(
        nid="EXIT_G_LEFT", 
        ntype="exit", 
        length=3.0, 
        area=6.0, 
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_node(
        nid="EXIT_G_RIGHT", 
        ntype="exit", 
        length=3.0, 
        area=6.0, 
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_edge("EXIT_G_LEFT", "F0_C0", length=1.0, width=1.6, door=True)
    env.add_edge("EXIT_G_RIGHT", "F0_C2", length=1.0, width=1.6, door=True)

    # Rooftop emergency access
    env.add_node(
        nid=f"F{FLOORS-1}_ROOF_EXIT", 
        ntype="exit", 
        length=2.0, 
        area=4.0, 
        floor=FLOORS-1,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_edge(f"F{FLOORS-1}_ROOF_EXIT", f"F{FLOORS-1}_C2", length=1.0, width=1.0, door=True)

    # Populate with staff and infants
    for f in range(FLOORS):
        # One nurse per floor (adults with normal mobility)
        env.spawn_person(f"F{f}_NURSE", age=36, mobility="adult", hp=100.0)

        # Infants and caregivers in nurseries
        for r in range(6):
            # Caregiver (adult)
            env.spawn_person(f"F{f}_NUR{r}", age=28 + r, mobility="adult", hp=100.0)
            # Infant (child mobility - more vulnerable)
            env.spawn_person(f"F{f}_NUR{r}", age=1, mobility="child", hp=100.0)

    # Place agents at ground-floor exits
    env.place_agent(agent_id=0, node_id="EXIT_G_LEFT")
    env.place_agent(agent_id=1, node_id="EXIT_G_RIGHT")

    return env

def build_two_floor_warehouse() -> BuildingFireEnvironment:
    """
    Single-floor warehouse layout with a grid structure:
    - Grid of hallways (H_{r}_{c})
    - Cargo rooms between hall intersections
    - Two exits on opposite sides
    
    NEW: Some rooms are designated as labs (hazardous materials):
    - Corner rooms: role="lab", risk_level="high", required_sweeps=2
    - Other rooms: role="storage", risk_level="normal", required_sweeps=1
    """
    env = BuildingFireEnvironment()

    ROWS = 4
    COLS = 6

    # Create corridor grid nodes
    for r in range(ROWS):
        for c in range(COLS):
            env.add_node(
                nid=f"H_{r}_{c}", 
                ntype="hall", 
                length=10.0, 
                area=30.0, 
                floor=0,
                role="corridor",
                risk_level="normal",
                required_sweeps=1
            )

    # Connect corridor grid
    for r in range(ROWS):
        for c in range(COLS):
            if c + 1 < COLS:
                env.add_edge(f"H_{r}_{c}", f"H_{r}_{c+1}", length=8.0, width=3.0, door=False)
            if r + 1 < ROWS:
                env.add_edge(f"H_{r}_{c}", f"H_{r+1}_{c}", length=8.0, width=3.0, door=False)

    # NEW: Create cargo/lab rooms with risk classification
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            rid = f"R_{r}_{c}"
            
            # Corner rooms and center rooms are labs (hazardous materials)
            is_lab = ((r == 0 and c == 0) or              # Top-left corner
                      (r == 0 and c == COLS-2) or         # Top-right corner
                      (r == ROWS-2 and c == 0) or         # Bottom-left corner
                      (r == ROWS-2 and c == COLS-2) or    # Bottom-right corner
                      (r == 1 and c == 2))                # Center lab
            
            env.add_node(
                nid=rid, 
                ntype="room", 
                length=6.0, 
                area=40.0, 
                floor=0,
                # NEW: Labs require redundant searches due to hazardous materials
                role="lab" if is_lab else "storage",
                risk_level="high" if is_lab else "normal",
                required_sweeps=2 if is_lab else 1
            )
            
            # Connect to four surrounding halls
            env.add_edge(rid, f"H_{r}_{c}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r}_{c+1}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r+1}_{c}", length=1.0, width=1.2, door=True)
            env.add_edge(rid, f"H_{r+1}_{c+1}", length=1.0, width=1.2, door=True)

    # Exits on opposite sides
    env.add_node(
        nid="EXIT_WH_LEFT", 
        ntype="exit", 
        length=4.0, 
        area=8.0, 
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_node(
        nid="EXIT_WH_RIGHT", 
        ntype="exit", 
        length=4.0, 
        area=8.0, 
        floor=0,
        role="exit",
        risk_level="normal",
        required_sweeps=1
    )
    env.add_edge("EXIT_WH_LEFT", "H_2_0", length=2.0, width=2.0, door=True)
    env.add_edge("EXIT_WH_RIGHT", f"H_2_{COLS-1}", length=2.0, width=2.0, door=True)

    # Populate with workers (fewer than office, spread out)
    for r in range(0, ROWS - 1, 2):
        for c in range(0, COLS - 1, 2):
            rid = f"R_{r}_{c}"
            env.spawn_person(rid, age=35, mobility="adult", hp=100.0)

    # Agents at warehouse exits
    env.place_agent(0, "EXIT_WH_LEFT")
    env.place_agent(1, "EXIT_WH_RIGHT")

    return env