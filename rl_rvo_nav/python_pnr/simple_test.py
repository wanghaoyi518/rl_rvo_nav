#!/usr/bin/env python3
"""
Simple test for Push and Rotate implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_pnr.push_and_rotate import PushAndRotate
from python_pnr.sub_map import SubMap
from python_pnr.actor import Actor
from python_pnr.actor_set import ActorSet
from python_pnr.mapf_config import MAPFConfig
from python_pnr.node import Point, Node

def create_simple_scenario():
    """Create a simple scenario with just 2 agents"""
    # Simple 3x3 grid
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    
    sub_map = SubMap(grid)
    
    # Create 2 actors that need to swap
    actors = []
    
    # Actor 0: start at (1,1), goal at (1,1) (no movement needed)
    actor0 = Actor(0, Point(1, 1), Point(1, 1))
    actors.append(actor0)
    
    # Actor 1: start at (1,1), goal at (1,1) (no movement needed)
    actor1 = Actor(1, Point(1, 1), Point(1, 1))
    actors.append(actor1)
    
    actor_set = ActorSet()
    for actor in actors:
        actor_set.add_actor(actor)
    
    return sub_map, actor_set

def test_basic_functionality():
    """Test basic functionality"""
    print("Testing basic Push and Rotate functionality...")
    
    # Create test scenario
    sub_map, actor_set = create_simple_scenario()
    
    # Create configuration
    config = MAPFConfig()
    config.parallelizePaths2 = False  # Disable parallel paths for simplicity
    
    # Create solver
    solver = PushAndRotate()
    
    print(f"Initial positions:")
    for actor in actor_set:
        print(f"  Actor {actor.id}: ({actor.current.x}, {actor.current.y}) -> ({actor.goal.x}, {actor.goal.y})")
    
    # Solve
    print("\nSolving...")
    result = solver.start_search(sub_map, config, actor_set)
    
    if result:
        print("✓ Solution found!")
        print(f"Total moves: {len(solver.agents_moves)}")
        
        # Check for collisions
        print("\nChecking for collisions...")
        collision_found = False
        
        # Check position collisions
        positions_at_time = {}
        for i, move in enumerate(solver.agents_moves):
            agent_id, node = move
            time_step = i
            pos = (node.i, node.j)
            
            if time_step not in positions_at_time:
                positions_at_time[time_step] = {}
            
            if pos in positions_at_time[time_step]:
                print(f"✗ Collision detected at time {time_step}: Agent {agent_id} and {positions_at_time[time_step][pos]} at {pos}")
                collision_found = True
            else:
                positions_at_time[time_step][pos] = agent_id
        
        if not collision_found:
            print("✓ No collisions detected!")
        
        # Show final paths
        print("\nFinal paths:")
        for i, path in enumerate(solver.agents_paths):
            if path:
                print(f"  Actor {i}: {[(node.i, node.j) for node in path]}")
            else:
                print(f"  Actor {i}: No path")
            
    else:
        print("✗ No solution found")

if __name__ == "__main__":
    test_basic_functionality() 