#!/usr/bin/env python3
"""
Test PAR Coordinator interface with PNR solver
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deadlock_resolution.par_coordinator import PARCoordinator
from deadlock_resolution.par_environment import PAREnvironment
from python_pnr.push_and_rotate import PushAndRotate
from python_pnr.sub_map import SubMap
from python_pnr.actor import Actor
from python_pnr.actor_set import ActorSet
from python_pnr.mapf_config import MAPFConfig
from python_pnr.node import Point
from config.deadlock_config import DeadlockConfig

def test_par_interface():
    """Test PAR coordinator interface with PNR solver."""
    print("🔍 Testing PAR Coordinator Interface with PNR Solver")
    print("=" * 60)
    
    # Initialize configuration
    config = DeadlockConfig().config
    print(f"✅ Configuration loaded: {len(config)} parameters")
    
    # Initialize PNR solver
    pnr_solver = PushAndRotate()
    print(f"✅ PNR solver initialized: {type(pnr_solver)}")
    
    # Initialize PAR coordinator
    par_coordinator = PARCoordinator(pnr_solver, config)
    print(f"✅ PAR coordinator initialized")
    
    # Create test data similar to the working test case
    print("\n🔧 Creating test data...")
    
    # Create a simple 5x5 grid (similar to test case but smaller)
    grid = [
        [1, 1, 1, 1, 1],  # Row 0: walls
        [1, 0, 0, 0, 1],  # Row 1: free space
        [1, 0, 0, 0, 1],  # Row 2: free space  
        [1, 0, 0, 0, 1],  # Row 3: free space
        [1, 1, 1, 1, 1]   # Row 4: walls
    ]
    
    # Create sub-map
    sub_map = SubMap(grid)
    print(f"✅ Sub-map created: {sub_map.width}x{sub_map.height}")
    
    # Create actor set with 2 agents
    actor_set = ActorSet()
    
    # Agent 0: from (1,1) to (3,3)
    agent0 = Actor(0, Point(1, 1), Point(3, 3))
    actor_set.add_actor(agent0)
    
    # Agent 1: from (3,1) to (1,3)  
    agent1 = Actor(1, Point(3, 1), Point(1, 3))
    actor_set.add_actor(agent1)
    
    print(f"✅ Actor set created: {len(actor_set)} agents")
    
    # Test direct PNR solver call
    print("\n🧪 Testing direct PNR solver call...")
    try:
        mapf_config = MAPFConfig(max_steps=100, timeout=1000, heuristic_weight=1.0)
        print(f"   MAPF Config: {mapf_config.__dict__}")
        
        result = pnr_solver.start_search(sub_map, mapf_config, actor_set)
        print(f"   ✅ Direct PNR call successful!")
        print(f"   Result success: {getattr(result, 'success', 'N/A')}")
        print(f"   Agents moves: {len(getattr(pnr_solver, 'agents_moves', []))}")
        
    except Exception as e:
        print(f"   ❌ Direct PNR call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PAR coordinator interface
    print("\n🧪 Testing PAR coordinator interface...")
    try:
        # Create start and goal positions dictionaries
        start_positions = {0: (1, 1), 1: (3, 1)}
        goal_positions = {0: (3, 3), 1: (1, 3)}
        
        print(f"   Start positions: {start_positions}")
        print(f"   Goal positions: {goal_positions}")
        
        # Call solve_par_problem
        result = par_coordinator.solve_par_problem(sub_map, actor_set, start_positions, goal_positions)
        print(f"   ✅ PAR coordinator call successful!")
        print(f"   Result type: {type(result)}")
        print(f"   Result success: {getattr(result, 'success', 'N/A')}")
        print(f"   Agents moves: {len(getattr(result, 'agents_moves', []))}")
        
    except Exception as e:
        print(f"   ❌ PAR coordinator call failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_par_interface()
