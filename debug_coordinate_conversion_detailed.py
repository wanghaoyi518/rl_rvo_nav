#!/usr/bin/env python3
"""
Detailed coordinate conversion debugging script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_coordinate_conversion_detailed():
    """Debug coordinate conversion step by step."""
    print("🔍 Detailed Coordinate Conversion Debugging")
    print("=" * 70)
    
    try:
        # Import required modules
        import sys
        sys.path.append('gym_env')
        import gym
        import gym_env
        import numpy as np
        
        # Create gym environment
        print("1. Creating gym environment...")
        env = gym.make('mrnav-v1', world_name='gym_env/gym_test_world.yaml', robot_number=10, robot_init_mode=3, random_bear=True)
        ir_gym_env = env.ir_gym
        print(f"   ✅ Gym environment created")
        
        # Get agent states
        print("\n2. Getting agent states...")
        ts = ir_gym_env.components['robots'].total_states()
        robot_state_list = ts[0]
        agent_states = ir_gym_env._get_agent_states_dict(robot_state_list)
        print(f"   ✅ Agent states: {len(agent_states)} agents")
        
        # Test PAR environment step by step
        print("\n3. Testing PAR environment step by step...")
        try:
            from deadlock_resolution.par_environment import PAREnvironment
            
            # Create workspace
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
            
            # Test with specific participants (like in the test)
            participants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
            
            print(f"   Creating PAR environment with {len(participants)} participants...")
            par_env = PAREnvironment(workspace, participants, config)
            print(f"   ✅ PAR environment created")
            
            # Check initial boundaries
            print(f"\n4. Initial boundaries: min_x={par_env.min_x}, max_x={par_env.max_x}, min_y={par_env.min_y}, max_y={par_env.max_y}")
            
            # Build environment
            print("\n5. Building PAR environment...")
            sub_map, actor_set = par_env.build_par_environment(agent_states)
            print(f"   ✅ Environment built")
            print(f"   Final boundaries: min_x={par_env.min_x}, max_x={par_env.max_x}, min_y={par_env.min_y}, max_y={par_env.max_y}")
            print(f"   Sub-map size: {sub_map.width} x {sub_map.height}")
            
            # Check start and goal positions
            print("\n6. Checking start and goal positions...")
            start_positions = par_env.compute_start_positions(agent_states)
            goal_positions = par_env.compute_goal_positions(agent_states)
            
            print(f"   Start positions: {start_positions}")
            print(f"   Goal positions: {goal_positions}")
            
            # Debug specific agents with problems
            print("\n7. Debugging problematic agents...")
            for agent_id in [0, 1]:  # Focus on agents with start=goal
                if agent_id in agent_states:
                    agent_state = agent_states[agent_id]
                    print(f"\n   Agent {agent_id}:")
                    
                    # Check raw position and goal
                    raw_position = par_env.get_agent_position(agent_state)
                    raw_goal = par_env.get_agent_goal(agent_state)
                    print(f"     Raw position: {raw_position}")
                    print(f"     Raw goal: {raw_goal}")
                    
                    if raw_position and raw_goal:
                        # Manual coordinate conversion
                        pos_x, pos_y = raw_position
                        goal_x, goal_y = raw_goal
                        
                        print(f"     Position (x,y): ({pos_x}, {pos_y})")
                        print(f"     Goal (x,y): ({goal_x}, {goal_y})")
                        
                        # Manual grid conversion
                        manual_start_x = int((pos_x - par_env.min_x) / par_env.grid_resolution)
                        manual_start_y = int((pos_y - par_env.min_y) / par_env.grid_resolution)
                        manual_goal_x = int((goal_x - par_env.min_x) / par_env.grid_resolution)
                        manual_goal_y = int((goal_y - par_env.min_y) / par_env.grid_resolution)
                        
                        print(f"     Manual grid start: ({manual_start_x}, {manual_start_y})")
                        print(f"     Manual grid goal: ({manual_goal_x}, {manual_goal_y})")
                        
                        # Compare with computed values
                        computed_start = start_positions.get(agent_id)
                        computed_goal = goal_positions.get(agent_id)
                        
                        print(f"     Computed start: {computed_start}")
                        print(f"     Computed goal: {computed_goal}")
                        
                        # Check if they match
                        if computed_start == (manual_start_x, manual_start_y):
                            print(f"     ✅ Start position conversion correct")
                        else:
                            print(f"     ❌ Start position conversion mismatch!")
                        
                        if computed_goal == (manual_goal_x, manual_goal_y):
                            print(f"     ✅ Goal position conversion correct")
                        else:
                            print(f"     ❌ Goal position conversion mismatch!")
                        
                        # Check if start != goal
                        if (manual_start_x, manual_start_y) != (manual_goal_x, manual_goal_y):
                            print(f"     ✅ Manual conversion: start ≠ goal")
                        else:
                            print(f"     ❌ Manual conversion: start = goal")
                        
                        if computed_start != computed_goal:
                            print(f"     ✅ Computed conversion: start ≠ goal")
                        else:
                            print(f"     ❌ Computed conversion: start = goal")
            
            # Check actor set creation
            print("\n8. Checking actor set creation...")
            print(f"   Building actor set...")
            actor_set = par_env.build_actor_set(agent_states)
            print(f"   ✅ Actor set built with {len(actor_set)} actors")
            
            # Check each actor's start and goal
            for i, actor in enumerate(actor_set):
                if i < len(participants):
                    agent_id = participants[i]
                    print(f"   Actor {i} (Agent {agent_id}):")
                    print(f"     Start: ({actor.current.x}, {actor.current.y})")
                    print(f"     Goal: ({actor.goal.x}, {actor.goal.y})")
                    
                    if actor.current.x == actor.goal.x and actor.current.y == actor.goal.y:
                        print(f"     ⚠️ Start = Goal!")
                    else:
                        print(f"     ✅ Start ≠ Goal")
                        dx = actor.goal.x - actor.current.x
                        dy = actor.goal.y - actor.current.y
                        print(f"     Distance: ({dx}, {dy})")
        
        except Exception as e:
            print(f"   ❌ Error testing PAR environment: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("🏁 Detailed coordinate conversion debug completed!")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_coordinate_conversion_detailed()
