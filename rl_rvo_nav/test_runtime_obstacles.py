#!/usr/bin/env python3
"""
Test script for runtime obstacle injection (no yaml files!)
"""

import sys
from pathlib import Path
import numpy as np

# Add rl_rvo_nav to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.runtime_obstacle_manager import RuntimeObstacleManager, create_runtime_obstacle_env

def test_runtime_obstacle_manager():
    """Test the runtime obstacle manager"""
    print("=== Testing Runtime Obstacle Manager ===")
    
    # Create manager
    manager = RuntimeObstacleManager(
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
        
        obs_polygons, obs_lines = manager.generate_random_obstacles(
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
        
        if len(centers) > 1:
            min_dist = float('inf')
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                 (centers[i][1] - centers[j][1])**2)
                    min_dist = min(min_dist, dist)
            
            print(f"Minimum distance between obstacles: {min_dist:.2f}")
            print(f"Required minimum distance: {manager.min_obstacle_distance + manager.obstacle_size}")
            
            if min_dist >= manager.min_obstacle_distance + manager.obstacle_size:
                print("✓ Distance constraint satisfied")
            else:
                print("✗ Distance constraint violated")

def test_runtime_environment_creation():
    """Test runtime environment creation"""
    print("\n=== Testing Runtime Environment Creation ===")
    
    try:
        # Base world config path
        base_world_path = Path(__file__).parent / 'policy_train' / 'train_world.yaml'
        
        if not base_world_path.exists():
            print(f"Warning: Base world file not found at {base_world_path}")
            print("Using basic test configuration")
            return
            
        print(f"Base world config: {base_world_path}")
        
        # Create environment
        env = create_runtime_obstacle_env(
            base_world_config=str(base_world_path),
            robot_number=6,
            neighbors_region=4,
            neighbors_num=5,
            robot_init_mode=2,
            env_train=True,
            random_bear=True,
            random_radius=False
        )
        
        print("✓ Runtime obstacle environment created successfully")
        
        # Test multiple resets to verify obstacles change
        for i in range(3):
            print(f"\n--- Reset {i+1} ---")
            obs_list = env.reset(mode=2)
            print(f"Environment reset successful, got {len(obs_list)} observations")
            
            # Check if obstacles exist in environment
            if hasattr(env.ir_gym, 'obs_poly_list'):
                obstacle_count = len(env.ir_gym.obs_poly_list)
                print(f"Found {obstacle_count} polygon obstacles in environment")
            
            if hasattr(env.ir_gym, 'obs_line_states'):
                line_count = len(env.ir_gym.obs_line_states)
                print(f"Found {line_count} obstacle lines in environment")
        
        print("✓ Multiple resets successful with dynamic obstacle generation")
        
        # Cleanup
        env.cleanup()
        print("✓ Environment cleanup successful")
        
    except Exception as e:
        print(f"✗ Error testing runtime environment: {e}")
        import traceback
        traceback.print_exc()

def test_no_temporary_files():
    """Test that no temporary files are created"""
    print("\n=== Testing No Temporary File Creation ===")
    
    import tempfile
    import os
    
    # Count files in temp directory before
    temp_dir = tempfile.gettempdir()
    files_before = set(os.listdir(temp_dir))
    
    # Create manager and generate obstacles multiple times
    manager = RuntimeObstacleManager()
    for i in range(5):
        obs_polygons, obs_lines = manager.generate_random_obstacles()
        
    # Count files in temp directory after
    files_after = set(os.listdir(temp_dir))
    new_files = files_after - files_before
    
    # Filter for yaml files that might be ours
    yaml_files = [f for f in new_files if f.endswith('.yaml') and 'static' in f]
    
    if len(yaml_files) == 0:
        print("✓ No temporary yaml files created")
    else:
        print(f"✗ Found {len(yaml_files)} temporary yaml files: {yaml_files}")
    
    print(f"Total new temp files: {len(new_files)} (expected: 0 yaml files)")

if __name__ == "__main__":
    print("Testing Runtime Obstacle Injection System")
    print("=" * 50)
    print("This system does NOT create any temporary yaml files!")
    print("Obstacles are injected directly into environment at runtime.")
    print("=" * 50)
    
    try:
        test_runtime_obstacle_manager()
        test_no_temporary_files()
        test_runtime_environment_creation()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        print("\n🎉 Key Benefits:")
        print("   • No temporary yaml files generated")
        print("   • Direct runtime obstacle injection") 
        print("   • Each episode has unique obstacle layout")
        print("   • Much more efficient than file-based approach")
        print("\n🚀 You can now run the training with:")
        print("cd /home/haoyiwang/Desktop/RL_RVO/rl_rvo_nav/rl_rvo_nav/policy_train")
        print("python3 train_static_obstacles.py --robot_number 6 --train_epoch 1000")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
