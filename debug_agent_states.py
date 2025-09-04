#!/usr/bin/env python3
"""
Debug script to check agent states in gym environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_agent_states():
    """Debug agent states in gym environment."""
    print("🔍 Debugging Agent States in Gym Environment")
    print("=" * 60)
    
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
        print(f"   ✅ Gym environment created: {type(env)}")
        
        # Check robot list
        print("\n2. Checking robot list...")
        if hasattr(ir_gym_env, 'robot_list') and ir_gym_env.robot_list:
            print(f"   Found {len(ir_gym_env.robot_list)} robots")
            
            for i, robot in enumerate(ir_gym_env.robot_list):
                print(f"   Robot {i}:")
                print(f"     Type: {type(robot)}")
                print(f"     Has goal: {hasattr(robot, 'goal')}")
                if hasattr(robot, 'goal'):
                    print(f"     Goal: {robot.goal}")
                print(f"     Has target: {hasattr(robot, 'target')}")
                if hasattr(robot, 'target'):
                    print(f"     Target: {robot.target}")
                print(f"     Has destination: {hasattr(robot, 'destination')}")
                if hasattr(robot, 'destination'):
                    print(f"     Destination: {robot.destination}")
        else:
            print("   ❌ No robot list found")
        
        # Check components
        print("\n3. Checking environment components...")
        if hasattr(ir_gym_env, 'components') and ir_gym_env.components:
            print(f"   Components: {list(ir_gym_env.components.keys())}")
            
            if 'robots' in ir_gym_env.components:
                robots_component = ir_gym_env.components['robots']
                print(f"   Robots component: {type(robots_component)}")
                
                # Try to get total states
                if hasattr(robots_component, 'total_states'):
                    print("   ✅ Robots component has total_states method")
                    
                    # Get robot states
                    try:
                        ts = robots_component.total_states()
                        robot_state_list = ts[0]
                        print(f"   Robot state list: {len(robot_state_list)} robots")
                        
                        # Check first robot state
                        if robot_state_list:
                            first_robot_state = robot_state_list[0]
                            print(f"   First robot state: {first_robot_state}")
                            print(f"   State length: {len(first_robot_state)}")
                            
                            # Extract position and velocity
                            if len(first_robot_state) >= 4:
                                position = first_robot_state[0:2]
                                velocity = first_robot_state[2:4]
                                print(f"   Position: {position}")
                                print(f"   Velocity: {velocity}")
                        
                    except Exception as e:
                        print(f"   ❌ Error getting total states: {e}")
                else:
                    print("   ❌ Robots component has no total_states method")
        else:
            print("   ❌ No components found")
        
        # Test agent states conversion
        print("\n4. Testing agent states conversion...")
        try:
            if hasattr(ir_gym_env, '_get_agent_states_dict'):
                # Get robot states
                ts = ir_gym_env.components['robots'].total_states()
                robot_state_list = ts[0]
                
                # Convert to agent states
                agent_states = ir_gym_env._get_agent_states_dict(robot_state_list)
                print(f"   ✅ Agent states conversion successful: {len(agent_states)} agents")
                
                # Check first agent state
                if 0 in agent_states:
                    agent_0 = agent_states[0]
                    print(f"   Agent 0 state:")
                    print(f"     Position: {agent_0.get('position', 'N/A')}")
                    print(f"     Velocity: {agent_0.get('velocity', 'N/A')}")
                    print(f"     Goal: {agent_0.get('goal', 'N/A')}")
                    
                    # Check if goal is different from position
                    position = agent_0.get('position', [])
                    goal = agent_0.get('goal', [])
                    
                    if len(position) >= 2 and len(goal) >= 2:
                        # Convert NumPy arrays to lists for comparison
                        pos_list = position.tolist() if hasattr(position, 'tolist') else list(position)
                        goal_list = goal.tolist() if hasattr(goal, 'tolist') else list(goal)
                        
                        # Extract numeric values for comparison
                        pos_x = float(pos_list[0][0]) if isinstance(pos_list[0], (list, np.ndarray)) else float(pos_list[0])
                        pos_y = float(pos_list[1][0]) if isinstance(pos_list[1], (list, np.ndarray)) else float(pos_list[1])
                        goal_x = float(goal_list[0][0]) if isinstance(goal_list[0], (list, np.ndarray)) else float(goal_list[0])
                        goal_y = float(goal_list[1][0]) if isinstance(goal_list[1], (list, np.ndarray)) else float(goal_list[1])
                        
                        if pos_x == goal_x and pos_y == goal_y:
                            print(f"     ⚠️ Goal is the same as position!")
                        else:
                            print(f"     ✅ Goal is different from position")
                            print(f"     Distance: ({goal_x - pos_x:.2f}, {goal_y - pos_y:.2f})")
                    else:
                        print(f"     ❌ Invalid position or goal data")
                
            else:
                print("   ❌ No _get_agent_states_dict method found")
                
        except Exception as e:
            print(f"   ❌ Error testing agent states conversion: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("🏁 Debug completed!")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_agent_states()
