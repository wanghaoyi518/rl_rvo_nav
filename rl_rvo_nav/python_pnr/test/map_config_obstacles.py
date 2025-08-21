# Map configuration based on train_world_with_obstacles.yaml
# 10x10 world with 2 obstacles
MAP_CONFIG = {
    "grid": [
        # 0: free space, 1: obstacle
        # Row 0-9 (y=0 to y=9)
        [1,1,1,1,1,1,1,1,1,1,1],  # y=0 (bottom)
        [1,0,0,0,0,0,0,0,0,0,1],  # y=1
        [1,0,0,0,0,0,0,0,0,0,1],  # y=2
        [1,0,0,1,1,0,0,0,0,0,1],  # y=3 (obstacle 1: [2,2] to [4,4])
        [1,0,0,1,1,0,0,0,0,0,1],  # y=4 (obstacle 1: [2,2] to [4,4])
        [1,0,0,0,0,0,0,0,0,0,1],  # y=5
        [1,0,0,0,0,0,1,1,0,0,1],  # y=6 (obstacle 2: [6,6] to [8,8])
        [1,0,0,0,0,0,1,1,0,0,1],  # y=7 (obstacle 2: [6,6] to [8,8])
        [1,0,0,0,0,0,0,0,0,0,1],  # y=8
        [1,0,0,0,0,0,0,0,0,0,1],  # y=9
        [1,1,1,1,1,1,1,1,1,1,1],  # y=10 (top)
    ],
    "agents": [
        # Agent 0: start at (0,0), goal at (10,10)
        {"id": 0, "start": (1, 1), "goal": (9, 9)},
        # Agent 1: start at (10,0), goal at (0,10) 
        {"id": 1, "start": (9, 1), "goal": (1, 9)},
    ]
}

# Alternative configuration with different start/goal positions
MAP_CONFIG_ALT = {
    "grid": [
        # Same grid as above
        [1,1,1,1,1,1,1,1,1,1,1],  # y=0 (bottom)
        [1,0,0,0,0,0,0,0,0,0,1],  # y=1
        [1,0,0,0,0,0,0,0,0,0,1],  # y=2
        [1,0,0,1,1,0,0,0,0,0,1],  # y=3 (obstacle 1: [2,2] to [4,4])
        [1,0,0,1,1,0,0,0,0,0,1],  # y=4 (obstacle 1: [2,2] to [4,4])
        [1,0,0,0,0,0,0,0,0,0,1],  # y=5
        [1,0,0,0,0,0,1,1,0,0,1],  # y=6 (obstacle 2: [6,6] to [8,8])
        [1,0,0,0,0,0,1,1,0,0,1],  # y=7 (obstacle 2: [6,6] to [8,8])
        [1,0,0,0,0,0,0,0,0,0,1],  # y=8
        [1,0,0,0,0,0,0,0,0,0,1],  # y=9
        [1,1,1,1,1,1,1,1,1,1,1],  # y=10 (top)
    ],
    "agents": [
        # Agent 0: start at bottom-left, goal at top-right
        {"id": 0, "start": (1, 1), "goal": (9, 9)},
        # Agent 1: start at bottom-right, goal at top-left
        {"id": 1, "start": (9, 1), "goal": (1, 9)},
    ]
}
