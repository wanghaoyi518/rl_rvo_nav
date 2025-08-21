#!/usr/bin/env python3
"""
Test script to verify obstacle training environment setup
"""

import os
import sys
import gym
import yaml
from pathlib import Path

def test_static_obstacles():
    """Test static obstacle configuration."""
    print("=== Testing Static Obstacles ===")
    
    # Path to training directory
    train_path = Path(__file__).parent.parent / 'policy_train'
    world_path = str(train_path / 'train_world_with_obstacles.yaml')
    
    try:
        # Load and verify YAML configuration
        with open(world_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"World config loaded successfully")
        print(f"Obstacles: {config.get('obs_polygons', {}).get('number', 0)}")
        print(f"Obstacle lines: {config.get('obs_lines', {}).get('number', 0)}")
        
        # Create environment
        env = gym.make('mrnav-v1', world_name=world_path, robot_number=2, 
                      neighbors_region=4, neighbors_num=5, robot_init_mode=2, 
                      env_train=True, random_bear=True, random_radius=False, 
                      reward_parameter=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), full=False)
        
        print("Environment created successfully with static obstacles")
        
        # Test reset
        obs_list = env.reset(mode=2)
        print(f"Environment reset successful, observation shape: {len(obs_list)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing static obstacles: {e}")
        return False

def test_dynamic_obstacles():
    """Test dynamic obstacle generation."""
    print("\n=== Testing Dynamic Obstacles ===")
    
    try:
        # Import the obstacle generation function from training directory
        train_path = Path(__file__).parent.parent / 'policy_train'
        sys.path.append(str(train_path))
        from train_with_obstacles import generate_dynamic_obstacles, create_obstacle_world_config
        
        # Test obstacle generation
        obs_polygons, obs_lines = generate_dynamic_obstacles(obstacle_count=2, min_distance=5.0)
        print(f"Generated {len(obs_polygons)} obstacles")
        print(f"Generated {len(obs_lines)} obstacle lines")
        
        # Test world config creation
        world_config = create_obstacle_world_config(obstacle_count=2, min_distance=5.0)
        print(f"World config created with {world_config['obs_polygons']['number']} obstacles")
        
        # Save to temporary file and test environment creation
        temp_world_path = str(Path(__file__).parent / 'temp_test_world.yaml')
        with open(temp_world_path, 'w') as f:
            yaml.dump(world_config, f, default_flow_style=False, sort_keys=False)
        
        # Create environment with dynamic obstacles
        env = gym.make('mrnav-v1', world_name=temp_world_path, robot_number=2, 
                      neighbors_region=4, neighbors_num=5, robot_init_mode=2, 
                      env_train=True, random_bear=True, random_radius=False, 
                      reward_parameter=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), full=False)
        
        print("Environment created successfully with dynamic obstacles")
        
        # Test reset
        obs_list = env.reset(mode=2)
        print(f"Environment reset successful, observation shape: {len(obs_list)}")
        
        # Clean up
        os.remove(temp_world_path)
        print("Temporary world file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"Error testing dynamic obstacles: {e}")
        return False

def test_obstacle_distance_constraint():
    """Test that obstacles respect distance constraint."""
    print("\n=== Testing Obstacle Distance Constraint ===")
    
    try:
        train_path = Path(__file__).parent.parent / 'policy_train'
        sys.path.append(str(train_path))
        from train_with_obstacles import generate_dynamic_obstacles
        import numpy as np
        
        # Generate obstacles multiple times to test constraint
        for i in range(5):
            obs_polygons, _ = generate_dynamic_obstacles(obstacle_count=2, min_distance=5.0)
            
            if len(obs_polygons) >= 2:
                # Calculate centers of obstacles
                center1 = np.mean(obs_polygons[0], axis=0)
                center2 = np.mean(obs_polygons[1], axis=0)
                
                distance = np.linalg.norm(center1 - center2)
                print(f"Test {i+1}: Obstacle distance = {distance:.2f} (min required: 5.0)")
                
                if distance < 5.0:
                    print(f"  WARNING: Distance constraint violated!")
                    return False
            else:
                print(f"Test {i+1}: Not enough obstacles generated")
        
        print("All distance constraint tests passed!")
        return True
        
    except Exception as e:
        print(f"Error testing distance constraint: {e}")
        return False

def test_training_script_compatibility():
    """Test that the training script can be imported and used."""
    print("\n=== Testing Training Script Compatibility ===")
    
    try:
        train_path = Path(__file__).parent.parent / 'policy_train'
        sys.path.append(str(train_path))
        
        # Test importing the training script
        import train_with_obstacles
        print("Training script imported successfully")
        
        # Test that it has the required functions
        assert hasattr(train_with_obstacles, 'generate_dynamic_obstacles')
        assert hasattr(train_with_obstacles, 'create_obstacle_world_config')
        print("Required functions found in training script")
        
        return True
        
    except Exception as e:
        print(f"Error testing training script compatibility: {e}")
        return False

if __name__ == "__main__":
    print("Testing Obstacle Training Environment Setup")
    print("=" * 50)
    
    # Test static obstacles
    static_ok = test_static_obstacles()
    
    # Test dynamic obstacles
    dynamic_ok = test_dynamic_obstacles()
    
    # Test distance constraint
    constraint_ok = test_obstacle_distance_constraint()
    
    # Test training script compatibility
    compatibility_ok = test_training_script_compatibility()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Static obstacles: {'✓' if static_ok else '✗'}")
    print(f"Dynamic obstacles: {'✓' if dynamic_ok else '✗'}")
    print(f"Distance constraint: {'✓' if constraint_ok else '✗'}")
    print(f"Training script compatibility: {'✓' if compatibility_ok else '✗'}")
    
    if static_ok and dynamic_ok and constraint_ok and compatibility_ok:
        print("\n🎉 All tests passed! Obstacle training environment is ready.")
        print("\nUsage examples:")
        print("1. Train with static obstacles:")
        print("   cd policy_train && python train_with_obstacles.py --use_obstacles --robot_number 2")
        print("\n2. Train with dynamic obstacles:")
        print("   cd policy_train && python train_with_obstacles.py --use_obstacles --dynamic_obstacles --robot_number 2")
        print("\n3. Train without obstacles (original):")
        print("   cd policy_train && python train_process.py --robot_number 2")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
