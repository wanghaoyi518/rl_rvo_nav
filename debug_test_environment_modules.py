#!/usr/bin/env python3
"""
Debug actual modules used in test environment
"""

import sys
import os

def debug_test_environment_modules():
    """Debug actual modules used in test environment."""
    print("🔍 Debugging Actual Modules in Test Environment")
    print("=" * 60)
    
    try:
        # Try to import the modules that the test environment would use
        print("1. Testing direct imports...")
        
        # Import from current directory (what our debug scripts use)
        print(f"\n   Importing from current directory...")
        from deadlock_resolution.par_environment import PAREnvironment as PAREnv1
        from deadlock_resolution.par_coordinator import PARCoordinator as PARCoord1
        
        print(f"   ✅ Successfully imported from current directory")
        print(f"   PAREnvironment module: {PAREnv1.__module__}")
        print(f"   PARCoordinator module: {PARCoord1.__module__}")
        print(f"   PAREnvironment file: {getattr(PAREnv1, '__file__', 'N/A')}")
        print(f"   PARCoordinator file: {getattr(PARCoord1, '__file__', 'N/A')}")
        
        # Test functionality
        workspace = {
            'bounds': {'min_x': 0, 'max_x': 10, 'min_y': 0, 'max_y': 10},
            'obstacles': [],
            'grid_resolution': 1.0,
            'par_offset': 2
        }
        participants = [0, 1]
        config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
        
        par_env1 = PAREnv1(workspace, participants, config)
        print(f"   ✅ PAREnvironment created from current directory")
        
        # Test agent states
        agent_states = {
            0: {'position': [5.0, 5.0], 'goal': [2.0, 5.0]},
            1: {'position': [8.0, 8.0], 'goal': [2.0, 3.0]}
        }
        
        # Build environment
        sub_map, actor_set = par_env1.build_par_environment(agent_states)
        print(f"   ✅ Environment built from current directory")
        
        # Check positions
        start_positions = par_env1.compute_start_positions(agent_states)
        goal_positions = par_env1.compute_goal_positions(agent_states)
        
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
        
        # Now try to simulate the gym environment import
        print(f"\n2. Testing gym environment import simulation...")
        
        # Simulate the path calculation from ir_gym.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rl_rvo_nav_dir = os.path.join(current_dir, '..', '..', '..')
        
        print(f"   Current file directory: {current_dir}")
        print(f"   Calculated rl_rvo_nav_dir: {rl_rvo_nav_dir}")
        print(f"   rl_rvo_nav_dir exists: {os.path.exists(rl_rvo_nav_dir)}")
        
        # Add the calculated path to sys.path
        if rl_rvo_nav_dir not in sys.path:
            sys.path.insert(0, rl_rvo_nav_dir)
            print(f"   Added {rl_rvo_nav_dir} to sys.path")
        
        # Try to import again
        try:
            from deadlock_resolution.par_environment import PAREnvironment as PAREnv2
            from deadlock_resolution.par_coordinator import PARCoordinator as PARCoord2
            
            print(f"   ✅ Successfully imported after path manipulation")
            print(f"   PAREnvironment module: {PAREnv2.__module__}")
            print(f"   PARCoordinator module: {PARCoord2.__module__}")
            print(f"   PAREnvironment file: {getattr(PAREnv2, '__file__', 'N/A')}")
            print(f"   PARCoordinator file: {getattr(PARCoord2, '__file__', 'N/A')}")
            
            # Check if they're the same objects
            if PAREnv1 is PAREnv2:
                print(f"   ✅ Same PAREnvironment object")
            else:
                print(f"   ❌ Different PAREnvironment objects!")
            
            if PARCoord1 is PARCoord2:
                print(f"   ✅ Same PARCoordinator object")
            else:
                print(f"   ❌ Different PARCoordinator objects!")
            
            # Test functionality again
            par_env2 = PAREnv2(workspace, participants, config)
            print(f"   ✅ PAREnvironment created after path manipulation")
            
            # Build environment
            sub_map2, actor_set2 = par_env2.build_par_environment(agent_states)
            print(f"   ✅ Environment built after path manipulation")
            
            # Check positions
            start_positions2 = par_env2.compute_start_positions(agent_states)
            goal_positions2 = par_env2.compute_goal_positions(agent_states)
            
            print(f"   Start positions: {start_positions2}")
            print(f"   Goal positions: {goal_positions2}")
            
            # Verify fix
            for agent_id in [0, 1]:
                start = start_positions2.get(agent_id)
                goal = goal_positions2.get(agent_id)
                if start == goal:
                    print(f"   ❌ Agent {agent_id}: start = goal = {start}")
                else:
                    print(f"   ✅ Agent {agent_id}: start = {start}, goal = {goal}")
            
        except ImportError as e:
            print(f"   ❌ Import failed after path manipulation: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n3. Summary...")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Current file directory: {current_dir}")
        print(f"   Python path length: {len(sys.path)}")
        
        # Check if there are multiple deadlock_resolution directories
        deadlock_dirs = []
        for path in sys.path:
            potential_dir = os.path.join(path, 'deadlock_resolution')
            if os.path.exists(potential_dir):
                deadlock_dirs.append(potential_dir)
        
        print(f"   Found deadlock_resolution directories: {deadlock_dirs}")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Test environment modules debug completed!")

if __name__ == "__main__":
    debug_test_environment_modules()
