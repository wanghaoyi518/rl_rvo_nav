#!/usr/bin/env python3
"""
Debug PAR coordinator complete flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_par_coordinator_flow():
    """Debug PAR coordinator complete flow."""
    print("🔍 Debugging PAR Coordinator Complete Flow")
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
        
        # Test PAR coordinator flow
        print("\n3. Testing PAR coordinator flow...")
        try:
            from deadlock_resolution.par_coordinator import PARCoordinator
            from config.deadlock_config import DeadlockConfig
            
            config = DeadlockConfig().config
            print(f"   Creating PAR coordinator...")
            # Create a dummy PushAndRotate instance for testing
            from python_pnr.push_and_rotate import PushAndRotate
            pnr_solver = PushAndRotate()
            coordinator = PARCoordinator(pnr_solver, config, ir_gym_env)
            print(f"   ✅ PAR coordinator created")
            
            # Simulate deadlock detection
            print("\n4. Simulating deadlock detection...")
            deadlock_participants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            print(f"   Deadlock participants: {deadlock_participants}")
            
            # Call prepare_par_execution (this is what the test calls)
            print("\n5. Calling prepare_par_execution...")
            par_solution = coordinator.prepare_par_execution(agent_states, deadlock_participants)
            print(f"   ✅ prepare_par_environment completed")
            
            # Check if we got a solution
            if par_solution:
                print(f"   PAR solution type: {type(par_solution)}")
                if hasattr(par_solution, 'success'):
                    print(f"   Success: {par_solution.success}")
                if hasattr(par_solution, 'agents_moves'):
                    print(f"   Agents moves: {len(par_solution.agents_moves)}")
            else:
                print(f"   ❌ No PAR solution returned")
            
            # Now let's check the coordinator's internal state
            print("\n6. Checking coordinator internal state...")
            if hasattr(coordinator, 'par_environment') and coordinator.par_environment:
                par_env = coordinator.par_environment
                print(f"   PAR environment boundaries: min_x={par_env.min_x}, max_x={par_env.max_x}, min_y={par_env.min_y}, max_y={par_env.max_y}")
                print(f"   Sub-map size: {par_env.sub_map.width if par_env.sub_map else 'N/A'} x {par_env.sub_map.height if par_env.sub_map else 'N/A'}")
                
                # Check start and goal positions from the environment
                print(f"\n7. Checking positions from PAR environment...")
                start_positions = par_env.compute_start_positions(agent_states)
                goal_positions = par_env.compute_goal_positions(agent_states)
                
                print(f"   Start positions: {start_positions}")
                print(f"   Goal positions: {goal_positions}")
                
                # Check specific problematic agents
                print(f"\n8. Checking problematic agents...")
                for agent_id in [0, 1]:
                    if agent_id in start_positions and agent_id in goal_positions:
                        start = start_positions[agent_id]
                        goal = goal_positions[agent_id]
                        if start == goal:
                            print(f"   ⚠️ Agent {agent_id}: start = goal = {start}")
                        else:
                            print(f"   ✅ Agent {agent_id}: start = {start}, goal = {goal}")
                
                # Check actor set
                print(f"\n9. Checking actor set...")
                if hasattr(par_env, 'actor_set') and par_env.actor_set:
                    actor_set = par_env.actor_set
                    print(f"   Actor set size: {len(actor_set)}")
                    
                    for i, actor in enumerate(actor_set):
                        if i < len(deadlock_participants):
                            agent_id = deadlock_participants[i]
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
                else:
                    print(f"   ❌ No actor set found")
            
        except Exception as e:
            print(f"   ❌ Error testing PAR coordinator: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("🏁 PAR coordinator flow debug completed!")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_par_coordinator_flow()
