#!/usr/bin/env python3
"""
Debug script to verify participants determination and boundary calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_participants_and_boundaries():
    """Debug participants determination and boundary calculation."""
    print("🔍 Debugging Participants and Boundaries")
    print("=" * 60)
    
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
        
        # Test deadlock detector
        print("\n3. Testing deadlock detector...")
        try:
            from deadlock_resolution.deadlock_detector import DeadlockDetector
            from config.deadlock_config import DeadlockConfig
            
            config = DeadlockConfig().config
            detector = DeadlockDetector(config)
            print(f"   ✅ Deadlock detector created")
            
            # Simulate neighbor states for agent 0
            print("\n4. Simulating neighbor detection...")
            agent_0_neighbors = {1: agent_states[1], 2: agent_states[2], 3: agent_states[3]}
            print(f"   Agent 0 neighbors: {list(agent_0_neighbors.keys())}")
            
            # Get deadlock participants
            participants = detector.get_deadlock_participants(0, agent_0_neighbors)
            print(f"   Deadlock participants: {participants}")
            print(f"   Expected: [0, 1, 2, 3]")
            print(f"   ✅ Participants correct: {participants == [0, 1, 2, 3]}")
            
            # Test PAR environment with these participants
            print("\n5. Testing PAR environment with participants...")
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
            
            # Create PAR environment with participants
            config = {'GRID_RESOLUTION': 1.0, 'PAR_OFFSET': 2}
            par_env = PAREnvironment(workspace, participants, config)
            print(f"   ✅ PAR environment created with {len(participants)} participants")
            
            # Check initial boundaries
            print(f"   Initial boundaries: min_x={par_env.min_x}, max_x={par_env.max_x}, min_y={par_env.min_y}, max_y={par_env.max_y}")
            
            # Build environment and check boundaries
            print("\n6. Building PAR environment...")
            sub_map, actor_set = par_env.build_par_environment(agent_states)
            print(f"   ✅ Environment built")
            print(f"   Final boundaries: min_x={par_env.min_x}, max_x={par_env.max_x}, min_y={par_env.min_y}, max_y={par_env.max_y}")
            print(f"   Sub-map size: {sub_map.width} x {sub_map.height}")
            
            # Check start and goal positions
            print("\n7. Checking start and goal positions...")
            start_positions = par_env.compute_start_positions(agent_states)
            goal_positions = par_env.compute_goal_positions(agent_states)
            
            print(f"   Start positions: {start_positions}")
            print(f"   Goal positions: {goal_positions}")
            
            # Verify that goals are different from starts
            print("\n8. Verifying position differences...")
            for agent_id in participants:
                if agent_id in start_positions and agent_id in goal_positions:
                    start = start_positions[agent_id]
                    goal = goal_positions[agent_id]
                    
                    if start == goal:
                        print(f"   ⚠️ Agent {agent_id}: start and goal are the same!")
                    else:
                        print(f"   ✅ Agent {agent_id}: start and goal are different")
                        print(f"     Start: {start}, Goal: {goal}")
                        print(f"     Distance: ({goal[0] - start[0]}, {goal[1] - start[1]})")
            
            # Check if boundaries include all positions and goals
            print("\n9. Verifying boundary coverage...")
            all_positions = []
            all_goals = []
            
            for agent_id in participants:
                if agent_id in agent_states:
                    agent_state = agent_states[agent_id]
                    position = agent_state.get('position', [])
                    goal = agent_state.get('goal', [])
                    
                    if len(position) >= 2:
                        pos_x = float(position[0][0]) if isinstance(position[0], (list, np.ndarray)) else float(position[0])
                        pos_y = float(position[1][0]) if isinstance(position[1], (list, np.ndarray)) else float(position[1])
                        all_positions.append((pos_x, pos_y))
                    
                    if len(goal) >= 2:
                        goal_x = float(goal[0][0]) if isinstance(goal[0], (list, np.ndarray)) else float(goal[0])
                        goal_y = float(goal[1][0]) if isinstance(goal[1], (list, np.ndarray)) else float(goal[1])
                        all_goals.append((goal_x, goal_y))
            
            print(f"   All positions: {all_positions}")
            print(f"   All goals: {all_goals}")
            print(f"   Boundary: x∈[{par_env.min_x:.3f}, {par_env.max_x:.3f}], y∈[{par_env.min_y:.3f}, {par_env.max_y:.3f}]")
            
            # Check if all positions and goals are within boundaries
            all_within = True
            for pos in all_positions + all_goals:
                if not (par_env.min_x <= pos[0] <= par_env.max_x and par_env.min_y <= pos[1] <= par_env.max_y):
                    all_within = False
                    print(f"   ⚠️ Position {pos} outside boundaries!")
            
            if all_within:
                print(f"   ✅ All positions and goals are within boundaries")
            else:
                print(f"   ❌ Some positions or goals are outside boundaries")
        
        except Exception as e:
            print(f"   ❌ Error testing deadlock detector: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("🏁 Participants and boundaries debug completed!")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_participants_and_boundaries()
