#!/usr/bin/env python3
"""
Debug script to check goal position extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_goal_extraction():
    """Debug goal position extraction process."""
    print("🔍 Debugging Goal Position Extraction")
    print("=" * 50)
    
    # Simulate the agent state structure from gym environment
    print("1. Simulating agent state structure...")
    
    # Mock agent states (similar to what we see in logs)
    agent_states = {
        0: {
            'position': [2.41077034, 0.80708436],
            'velocity': [0.0, 0.0],
            'goal': [4.41077034, 0.80708436]  # 2 units in x direction
        },
        1: {
            'position': [4.27014158, 8.1656818],
            'velocity': [0.0, 0.0],
            'goal': [6.27014158, 8.1656818]  # 2 units in x direction
        }
    }
    
    print(f"   Agent 0: position={agent_states[0]['position']}, goal={agent_states[0]['goal']}")
    print(f"   Agent 1: position={agent_states[1]['position']}, goal={agent_states[1]['goal']}")
    
    # Test coordinate conversion
    print("\n2. Testing coordinate conversion...")
    
    # Workspace bounds (from logs)
    min_x = 0.46988213567423925
    max_x = 9.328114604068688
    min_y = 0.45991120289214216
    max_y = 8.663386429354984
    grid_resolution = 1.0
    
    print(f"   Workspace bounds: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
    print(f"   Grid resolution: {grid_resolution}")
    
    for agent_id, agent_state in agent_states.items():
        position = agent_state['position']
        goal = agent_state['goal']
        
        # Convert to grid coordinates
        pos_grid_x = int((position[0] - min_x) / grid_resolution)
        pos_grid_y = int((position[1] - min_y) / grid_resolution)
        
        goal_grid_x = int((goal[0] - min_x) / grid_resolution)
        goal_grid_y = int((goal[1] - min_y) / grid_resolution)
        
        print(f"   Agent {agent_id}:")
        print(f"     Position: {position} -> grid ({pos_grid_x}, {pos_grid_y})")
        print(f"     Goal: {goal} -> grid ({goal_grid_x}, {goal_grid_y})")
        print(f"     Grid distance: ({goal_grid_x - pos_grid_x}, {goal_grid_y - pos_grid_y})")
    
    # Test PAR environment goal extraction
    print("\n3. Testing PAR environment goal extraction...")
    
    try:
        from deadlock_resolution.par_environment import PAREnvironment
        
        # Create mock workspace
        workspace = {
            'bounds': {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y},
            'obstacles': [],
            'grid_resolution': grid_resolution,
            'par_offset': 2
        }
        
        # Create PAR environment
        participants = [0, 1]
        config = {'GRID_RESOLUTION': grid_resolution, 'PAR_OFFSET': 2}
        par_env = PAREnvironment(workspace, participants, config)
        
        # Test goal extraction
        start_positions = par_env.compute_start_positions(agent_states)
        goal_positions = par_env.compute_goal_positions(agent_states)
        
        print(f"   Start positions: {start_positions}")
        print(f"   Goal positions: {goal_positions}")
        
        # Check if goals are different from starts
        for agent_id in participants:
            if agent_id in start_positions and agent_id in goal_positions:
                start = start_positions[agent_id]
                goal = goal_positions[agent_id]
                if start == goal:
                    print(f"   ⚠️ Agent {agent_id}: start and goal are the same!")
                else:
                    print(f"   ✅ Agent {agent_id}: start and goal are different")
        
    except Exception as e:
        print(f"   ❌ Error testing PAR environment: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 Debug completed!")

if __name__ == "__main__":
    debug_goal_extraction()
