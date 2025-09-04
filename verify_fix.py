#!/usr/bin/env python3
"""
Verify that the coordinate conversion fix is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_fix():
    """Verify the coordinate conversion fix."""
    print("🔍 Verifying Coordinate Conversion Fix")
    print("=" * 50)
    
    try:
        # Import the fixed modules
        from deadlock_resolution.par_environment import PAREnvironment
        from deadlock_resolution.par_coordinator import PARCoordinator
        
        print("✅ Successfully imported fixed modules")
        
        # Test PAREnvironment
        print("\n1. Testing PAREnvironment...")
        workspace = {
            'bounds': {'min_x': 0, 'max_x': 10, 'min_y': 0, 'max_y': 10},
            'obstacles': [],
            'grid_resolution': 1.0,
            'par_offset': 2
        }
        participants = [0, 1]
        config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
        
        par_env = PAREnvironment(workspace, participants, config)
        print("   ✅ PAREnvironment created")
        
        # Test agent states
        agent_states = {
            0: {'position': [5.0, 5.0], 'goal': [2.0, 5.0]},
            1: {'position': [8.0, 8.0], 'goal': [2.0, 3.0]}
        }
        
        # Build environment
        sub_map, actor_set = par_env.build_par_environment(agent_states)
        print("   ✅ Environment built")
        
        # Check positions
        start_positions = par_env.compute_start_positions(agent_states)
        goal_positions = par_env.compute_goal_positions(agent_states)
        
        print(f"   Start positions: {start_positions}")
        print(f"   Goal positions: {goal_positions}")
        
        # Verify fix
        for agent_id in [0, 1]:
            start = start_positions.get(agent_id)
            goal = goal_positions.get(agent_id)
            if start == goal:
                print(f"   ❌ Agent {agent_id}: start = goal = {start}")
            else:
                print(f"   ✅ Agent {agent_id}: start = {start}, goal = {goal}")
        
        # Test PARCoordinator
        print("\n2. Testing PARCoordinator...")
        from python_pnr.push_and_rotate import PushAndRotate
        pnr_solver = PushAndRotate()
        config_dict = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
        
        coordinator = PARCoordinator(pnr_solver, config_dict, None)
        print("   ✅ PARCoordinator created")
        
        # Test workspace bounds computation
        workspace_info = coordinator.get_workspace_info(agent_states)
        print(f"   Workspace bounds: {workspace_info.get('bounds', 'N/A')}")
        
        print("\n" + "=" * 50)
        print("🏁 Fix verification completed!")
        
    except Exception as e:
        print(f"❌ Error in verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
