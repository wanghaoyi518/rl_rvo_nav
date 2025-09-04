#!/usr/bin/env python3
"""
Debug script to check coordinate conversion in PAR environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_coordinate_conversion():
    """Debug coordinate conversion process."""
    print("🔍 Debugging Coordinate Conversion in PAR Environment")
    print("=" * 70)
    
    try:
        # Import gym environment
        import sys
        sys.path.append('gym_env')
        import gym
        import gym_env
        import numpy as np
        
        # Create a simple gym environment
        print("1. Creating gym environment...")
        env = gym.make('mrnav-v1', world_name='gym_env/gym_test_world.yaml', robot_number=10, robot_init_mode=3, random_bear=True)
        print(f"   ✅ Gym environment created: {type(env)}")
        
        # Access the ir_gym component
        ir_gym_env = env.ir_gym
        print(f"   ✅ ir_gym component: {type(ir_gym_env)}")
        
        # Get robot states
        print("\n2. Getting robot states...")
        ts = ir_gym_env.components['robots'].total_states()
        robot_state_list = ts[0]
        print(f"   ✅ Robot state list: {len(robot_state_list)} robots")
        
        # Convert to agent states
        print("\n3. Converting to agent states...")
        agent_states = ir_gym_env._get_agent_states_dict(robot_state_list)
        print(f"   ✅ Agent states conversion successful: {len(agent_states)} agents")
        
        # Check first few agents
        for i in range(min(3, len(agent_states))):
            agent_state = agent_states[i]
            print(f"\n   Agent {i}:")
            print(f"     Position: {agent_state.get('position', 'N/A')}")
            print(f"     Goal: {agent_state.get('goal', 'N/A')}")
            
            # Check if goal is different from position
            position = agent_state.get('position', [])
            goal = agent_state.get('goal', [])
            
            if len(position) >= 2 and len(goal) >= 2:
                # Convert NumPy arrays to lists for comparison
                pos_list = position.tolist() if hasattr(position, 'tolist') else list(position)
                goal_list = goal.tolist() if hasattr(goal, 'tolist') else list(goal)
                
                # Extract numeric values for comparison
                pos_x = float(pos_list[0][0]) if isinstance(pos_list[0], (list, np.ndarray)) else float(pos_list[0])
                pos_y = float(pos_list[1][0]) if isinstance(pos_list[1], (list, np.ndarray)) else float(pos_list[1])
                goal_x = float(goal_list[0][0]) if isinstance(goal_list[0], (list, np.ndarray)) else float(goal_list[0])
                goal_y = float(goal_list[1][0]) if isinstance(goal_list[1], (list, np.ndarray)) else float(goal_list[1])
                
                print(f"     Extracted - Position: ({pos_x:.3f}, {pos_y:.3f})")
                print(f"     Extracted - Goal: ({goal_x:.3f}, {goal_y:.3f})")
                
                if pos_x == goal_x and pos_y == goal_y:
                    print(f"     ⚠️ Goal is the same as position!")
                else:
                    print(f"     ✅ Goal is different from position")
                    print(f"     Distance: ({goal_x - pos_x:.3f}, {goal_y - pos_y:.3f})")
        
        # Now test PAR environment coordinate conversion
        print("\n4. Testing PAR environment coordinate conversion...")
        try:
            from deadlock_resolution.par_environment import PAREnvironment
            
            # Create mock workspace with bounds from logs
            workspace = {
                'bounds': {
                    'min_x': 0.46988213567423925,
                    'max_x': 9.328114604068688,
                    'min_y': 0.45991120289214216,
                    'max_y': 8.663386429354984
                },
                'obstacles': [],
                'grid_resolution': 1.0,
                'par_offset': 2
            }
            
            # Create PAR environment
            participants = list(range(min(3, len(agent_states))))
            config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
            par_env = PAREnvironment(workspace, participants, config)
            
            print(f"   ✅ PAR environment created with {len(participants)} participants")
            print(f"   Workspace bounds: {workspace['bounds']}")
            print(f"   Grid resolution: {config['GRID_RESOLUTION']}")
            
            # Test coordinate conversion step by step
            print("\n5. Step-by-step coordinate conversion...")
            
            for i in participants:
                agent_state = agent_states[i]
                position = agent_state.get('position', [])
                goal = agent_state.get('goal', [])
                
                if len(position) >= 2 and len(goal) >= 2:
                    # Extract numeric values
                    pos_list = position.tolist() if hasattr(position, 'tolist') else list(position)
                    goal_list = goal.tolist() if hasattr(goal, 'tolist') else list(goal)
                    
                    pos_x = float(pos_list[0][0]) if isinstance(pos_list[0], (list, np.ndarray)) else float(pos_list[0])
                    pos_y = float(pos_list[1][0]) if isinstance(pos_list[1], (list, np.ndarray)) else float(pos_list[1])
                    goal_x = float(goal_list[0][0]) if isinstance(goal_list[0], (list, np.ndarray)) else float(goal_list[0])
                    goal_y = float(goal_list[1][0]) if isinstance(goal_list[1], (list, np.ndarray)) else float(goal_list[1])
                    
                    print(f"\n   Agent {i} coordinate conversion:")
                    print(f"     Continuous - Position: ({pos_x:.3f}, {pos_y:.3f})")
                    print(f"     Continuous - Goal: ({goal_x:.3f}, {goal_y:.3f})")
                    
                    # Manual grid conversion
                    manual_pos_x = int((pos_x - workspace['bounds']['min_x']) / config['GRID_RESOLUTION'])
                    manual_pos_y = int((pos_y - workspace['bounds']['min_y']) / config['GRID_RESOLUTION'])
                    manual_goal_x = int((goal_x - workspace['bounds']['min_x']) / config['GRID_RESOLUTION'])
                    manual_goal_y = int((goal_y - workspace['bounds']['min_y']) / config['GRID_RESOLUTION'])
                    
                    print(f"     Manual Grid - Position: ({manual_pos_x}, {manual_pos_y})")
                    print(f"     Manual Grid - Goal: ({manual_goal_x}, {manual_goal_y})")
                    
                    # Check if manual conversion shows different values
                    if manual_pos_x == manual_goal_x and manual_pos_y == manual_goal_y:
                        print(f"     ⚠️ Manual conversion also shows same values!")
                    else:
                        print(f"     ✅ Manual conversion shows different values")
                        print(f"     Grid Distance: ({manual_goal_x - manual_pos_x}, {manual_goal_y - manual_pos_y})")
            
            # Now test PAR environment methods
            print("\n6. Testing PAR environment methods...")
            
            # Test start positions
            start_positions = par_env.compute_start_positions(agent_states)
            print(f"   PAR Start positions: {start_positions}")
            
            # Test goal positions
            goal_positions = par_env.compute_goal_positions(agent_states)
            print(f"   PAR Goal positions: {goal_positions}")
            
            # Compare with manual conversion
            print("\n7. Comparing PAR vs Manual conversion...")
            for i in participants:
                if i in start_positions and i in goal_positions:
                    par_start = start_positions[i]
                    par_goal = goal_positions[i]
                    
                    # Get manual conversion for this agent
                    agent_state = agent_states[i]
                    position = agent_state.get('position', [])
                    goal = agent_state.get('goal', [])
                    
                    if len(position) >= 2 and len(goal) >= 2:
                        pos_list = position.tolist() if hasattr(position, 'tolist') else list(position)
                        goal_list = goal.tolist() if hasattr(goal, 'tolist') else list(goal)
                        
                        pos_x = float(pos_list[0][0]) if isinstance(pos_list[0], (list, np.ndarray)) else float(pos_list[0])
                        pos_y = float(pos_list[1][0]) if isinstance(pos_list[1], (list, np.ndarray)) else float(pos_list[1])
                        goal_x = float(goal_list[0][0]) if isinstance(goal_list[0], (list, np.ndarray)) else float(goal_list[0])
                        goal_y = float(goal_list[1][0]) if isinstance(goal_list[1], (list, np.ndarray)) else float(goal_list[1])
                        
                        manual_start = (int((pos_x - workspace['bounds']['min_x']) / config['GRID_RESOLUTION']),
                                      int((pos_y - workspace['bounds']['min_y']) / config['GRID_RESOLUTION']))
                        manual_goal = (int((goal_x - workspace['bounds']['min_x']) / config['GRID_RESOLUTION']),
                                     int((goal_y - workspace['bounds']['min_y']) / config['GRID_RESOLUTION']))
                        
                        print(f"   Agent {i}:")
                        print(f"     PAR: start={par_start}, goal={par_goal}")
                        print(f"     Manual: start={manual_start}, goal={manual_goal}")
                        
                        if par_start == par_goal:
                            print(f"     ⚠️ PAR shows same start/goal!")
                        else:
                            print(f"     ✅ PAR shows different start/goal")
                        
                        if manual_start == manual_goal:
                            print(f"     ⚠️ Manual shows same start/goal!")
                        else:
                            print(f"     ✅ Manual shows different start/goal")
        
        except Exception as e:
            print(f"   ❌ Error testing PAR environment: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("🏁 Coordinate conversion debug completed!")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_coordinate_conversion()
