#!/usr/bin/env python3
"""
Debug import paths in test environment
"""

import sys
import os

def debug_import_paths():
    """Debug import paths in test environment."""
    print("🔍 Debugging Import Paths in Test Environment")
    print("=" * 60)
    
    # Current working directory
    print(f"1. Current working directory: {os.getcwd()}")
    
    # Python path
    print(f"\n2. Python path:")
    for i, path in enumerate(sys.path):
        print(f"   [{i}] {path}")
    
    # Try to simulate the gym environment import logic
    print(f"\n3. Simulating gym environment import logic...")
    
    # Simulate the path calculation from ir_gym.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"   Current file directory: {current_dir}")
    
    rl_rvo_nav_dir = os.path.join(current_dir, '..', '..', '..')
    print(f"   Calculated rl_rvo_nav_dir: {rl_rvo_nav_dir}")
    print(f"   rl_rvo_nav_dir exists: {os.path.exists(rl_rvo_nav_dir)}")
    
    # Check if our fixed modules exist in the calculated path
    deadlock_resolution_path = os.path.join(rl_rvo_nav_dir, 'deadlock_resolution')
    print(f"   deadlock_resolution path: {deadlock_resolution_path}")
    print(f"   deadlock_resolution exists: {os.path.exists(deadlock_resolution_path)}")
    
    if os.path.exists(deadlock_resolution_path):
        print(f"   Contents of deadlock_resolution:")
        for item in os.listdir(deadlock_resolution_path):
            print(f"     {item}")
    
    # Try to import the modules
    print(f"\n4. Testing imports...")
    
    # Add the calculated path to sys.path
    if rl_rvo_nav_dir not in sys.path:
        sys.path.insert(0, rl_rvo_nav_dir)
        print(f"   Added {rl_rvo_nav_dir} to sys.path")
    
    try:
        from deadlock_resolution.par_environment import PAREnvironment
        print(f"   ✅ Successfully imported PAREnvironment from deadlock_resolution")
        
        # Test the fixed functionality
        workspace = {
            'bounds': {'min_x': 0, 'max_x': 10, 'min_y': 0, 'max_y': 10},
            'obstacles': [],
            'grid_resolution': 1.0,
            'par_offset': 2
        }
        participants = [0, 1]
        config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
        
        par_env = PAREnvironment(workspace, participants, config)
        print(f"   ✅ PAREnvironment created successfully")
        
        # Test agent states
        agent_states = {
            0: {'position': [5.0, 5.0], 'goal': [2.0, 5.0]},
            1: {'position': [8.0, 8.0], 'goal': [2.0, 3.0]}
        }
        
        # Build environment
        sub_map, actor_set = par_env.build_par_environment(agent_states)
        print(f"   ✅ Environment built successfully")
        
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
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Import path debug completed!")

if __name__ == "__main__":
    debug_import_paths()
