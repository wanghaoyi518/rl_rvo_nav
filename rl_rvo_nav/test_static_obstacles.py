#!/usr/bin/env python3
"""
Test script for random static obstacle generation
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.static_obstacle_generator import StaticObstacleGenerator

def test_obstacle_generator():
    """Test the random static obstacle generator"""
    print("=== Testing Random Static Obstacle Generator ===")
    
    # Create generator
    generator = StaticObstacleGenerator(
        world_width=10.0,
        world_height=10.0,
        obstacle_size=1.0,
        obstacle_count=4,
        min_obstacle_distance=2.0
    )
    
    # Test multiple generations
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        
        # Example robot positions and goals
        robot_positions = [(1, 1), (8, 8), (2, 8), (8, 2), (5, 1), (5, 9)]
        robot_goals = [(8, 8), (1, 1), (8, 2), (2, 8), (5, 9), (5, 1)]
        
        obs_polygons, obs_lines = generator.generate_random_obstacles(
            robot_positions, robot_goals
        )
        
        print(f"Generated {len(obs_polygons)} obstacles:")
        for j, polygon in enumerate(obs_polygons):
            center_x = polygon[0][0] + 0.5
            center_y = polygon[0][1] + 0.5
            print(f"  Obstacle {j+1}: Center at ({center_x:.1f}, {center_y:.1f})")
            
        print(f"Generated {len(obs_lines)} obstacle lines")
        
        # Verify minimum distances
        centers = []
        for polygon in obs_polygons:
            center_x = polygon[0][0] + 0.5
            center_y = polygon[0][1] + 0.5
            centers.append((center_x, center_y))
        
        min_dist = float('inf')
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                             (centers[i][1] - centers[j][1])**2)
                min_dist = min(min_dist, dist)
        
        print(f"Minimum distance between obstacles: {min_dist:.2f}")
        print(f"Required minimum distance: {generator.min_obstacle_distance + generator.obstacle_size}")
        
        if min_dist >= generator.min_obstacle_distance + generator.obstacle_size:
            print("✓ Distance constraint satisfied")
        else:
            print("✗ Distance constraint violated")

def test_world_config_creation():
    """Test world configuration creation"""
    print("\n=== Testing World Configuration Creation ===")
    
    generator = StaticObstacleGenerator()
    
    # Base configuration
    base_config = {
        'world': {
            'world_height': 10,
            'world_width': 10,
            'step_time': 0.1
        },
        'robots': {
            'robot_mode': 'diff',
            'radius_list': [0.2, 0.2],
            'vel_max': [1.5, 1.5],
            'radius_exp': 0.1,
            'interval': 1,
            'square': [0, 0, 10, 10],
            'circular': [5, 5, 4]
        }
    }
    
    # Create config with obstacles
    config = generator.create_world_config(base_config)
    
    print("Generated world configuration:")
    print(f"  - Base config preserved: {config['world']}")
    print(f"  - Robot config preserved: {config['robots']}")
    print(f"  - Obstacles added: {config['obs_polygons']['number']} polygons")
    print(f"  - Lines added: {config['obs_lines']['number']} lines")
    
    # Save to file
    temp_file = "/tmp/test_static_world.yaml"
    generator.save_config_to_file(config, temp_file)
    print(f"  - Configuration saved to: {temp_file}")
    
    # Verify file contents
    import yaml
    with open(temp_file, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    if loaded_config == config:
        print("✓ Configuration file save/load successful")
    else:
        print("✗ Configuration file save/load failed")

if __name__ == "__main__":
    print("Testing Random Static Obstacle Generation System")
    print("=" * 50)
    
    try:
        test_obstacle_generator()
        test_world_config_creation()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\nYou can now run the training with:")
        print("cd /home/haoyiwang/Desktop/RL_RVO/rl_rvo_nav/rl_rvo_nav/policy_train")
        print("python3 train_static_obstacles.py --robot_number 6 --train_epoch 1000")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
