#!/usr/bin/env python3
"""
Random Environment Generator for RL-RVO Navigation Testing

This script generates random test environments for comparing RL agents with and without deadlock detection.
Each environment includes:
- 2, 4, or 6 agents
- 10x10 world
- 2 random 2x2 rectangular obstacles
- Random start and goal positions for each agent
"""

import yaml
import numpy as np
import random
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Any

class RandomEnvironmentGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the environment generator with a random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # World parameters
        self.world_height = 10
        self.world_width = 10
        self.step_time = 0.1
        
        # Robot parameters
        self.robot_radius = 0.2
        self.vel_max = 1.5
        self.radius_exp = 0.2
        self.interval = 1
        
        # Obstacle parameters
        self.obstacle_size = 2  # 2x2 obstacles
        
    def generate_random_point(self, exclude_obstacles: List[List[Tuple[float, float]]] = None) -> Tuple[float, float]:
        """Generate a random point in the world, avoiding obstacles."""
        while True:
            x = random.uniform(0, self.world_width)
            y = random.uniform(0, self.world_height)
            
            # Check if point is inside any obstacle
            if exclude_obstacles:
                inside_obstacle = False
                for obs in exclude_obstacles:
                    if (obs[0][0] <= x <= obs[1][0] and 
                        obs[0][1] <= y <= obs[1][1]):
                        inside_obstacle = True
                        break
                if inside_obstacle:
                    continue
            
            return (x, y)
    
    def generate_obstacles(self) -> Tuple[List[List[Tuple[float, float]]], List[List[float]]]:
        """Generate 2 random 2x2 rectangular obstacles with minimum distance constraint."""
        obstacles = []
        obs_polygons = []
        obs_lines = []
        
        min_obstacle_distance = 5.0  # 障碍物中心点之间的最小距离
        
        for i in range(2):
            max_attempts = 1000  # 最大尝试次数，避免无限循环
            attempts = 0
            
            while attempts < max_attempts:
                # Generate obstacle position (ensuring it fits within world bounds)
                x = random.uniform(0, self.world_width - self.obstacle_size)
                y = random.uniform(0, self.world_height - self.obstacle_size)
                
                # Calculate obstacle center
                center_x = x + self.obstacle_size / 2
                center_y = y + self.obstacle_size / 2
                
                # Check distance constraint with existing obstacles
                valid_position = True
                for existing_obs in obstacles:
                    existing_center_x = existing_obs[0][0] + self.obstacle_size / 2
                    existing_center_y = existing_obs[0][1] + self.obstacle_size / 2
                    
                    distance = np.sqrt((center_x - existing_center_x)**2 + (center_y - existing_center_y)**2)
                    if distance < min_obstacle_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    # Define obstacle corners
                    bottom_left = (x, y)
                    bottom_right = (x + self.obstacle_size, y)
                    top_right = (x + self.obstacle_size, y + self.obstacle_size)
                    top_left = (x, y + self.obstacle_size)
                    
                    obstacles.append([bottom_left, top_right])
                    
                    # Add to polygon list (clockwise order)
                    obs_polygons.append([
                        [bottom_left[0], bottom_left[1]],
                        [bottom_right[0], bottom_right[1]],
                        [top_right[0], top_right[1]],
                        [top_left[0], top_left[1]]
                    ])
                    
                    # Add line segments for obs_lines
                    obs_lines.extend([
                        [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]],
                        [bottom_right[0], bottom_right[1], top_right[0], top_right[1]],
                        [top_right[0], top_right[1], top_left[0], top_left[1]],
                        [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]
                    ])
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid position for obstacle {i+1} after {max_attempts} attempts")
                # Fallback: place obstacle at a random position without distance constraint
                x = random.uniform(0, self.world_width - self.obstacle_size)
                y = random.uniform(0, self.world_height - self.obstacle_size)
                
                bottom_left = (x, y)
                bottom_right = (x + self.obstacle_size, y)
                top_right = (x + self.obstacle_size, y + self.obstacle_size)
                top_left = (x, y + self.obstacle_size)
                
                obstacles.append([bottom_left, top_right])
                
                obs_polygons.append([
                    [bottom_left[0], bottom_left[1]],
                    [bottom_right[0], bottom_right[1]],
                    [top_right[0], top_right[1]],
                    [top_left[0], top_left[1]]
                ])
                
                obs_lines.extend([
                    [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]],
                    [bottom_right[0], bottom_right[1], top_right[0], top_right[1]],
                    [top_right[0], top_right[1], top_left[0], top_left[1]],
                    [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]
                ])
        
        return obstacles, obs_polygons, obs_lines
    
    def generate_agent_positions(self, robot_number: int, obstacles: List[List[Tuple[float, float]]]) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate random start and goal positions for agents."""
        init_states = []
        goal_list = []
        
        for _ in range(robot_number):
            # Generate start position
            start_pos = self.generate_random_point(obstacles)
            init_states.append([start_pos[0], start_pos[1], 0])  # [x, y, theta]
            
            # Generate goal position (different from start)
            while True:
                goal_pos = self.generate_random_point(obstacles)
                # Ensure goal is different from start
                if np.linalg.norm(np.array(goal_pos) - np.array(start_pos)) > 1.0:
                    break
            goal_list.append([goal_pos[0], goal_pos[1], 0])  # [x, y, theta]
        
        return init_states, goal_list
    
    def generate_environment(self, robot_number: int) -> Dict[str, Any]:
        """Generate a complete random environment configuration."""
        # Generate obstacles
        obstacles, obs_polygons, obs_lines = self.generate_obstacles()
        
        # Generate agent positions
        init_states, goal_list = self.generate_agent_positions(robot_number, obstacles)
        
        # Create environment configuration
        env_config = {
            'world': {
                'world_height': self.world_height,
                'world_width': self.world_width,
                'step_time': self.step_time
            },
            'robots': {
                'robot_number': robot_number,
                'robot_mode': 'diff',
                'robot_init_mode': 0,
                'init_state_list': init_states,
                'goal_list': goal_list,
                'radius_list': [self.robot_radius] * robot_number,
                'vel_max': [self.vel_max] * robot_number,
                'radius_exp': self.radius_exp,
                'interval': self.interval,
                'square': [0, 0, self.world_width, self.world_height],
                'circular': [self.world_width/2, self.world_height/2, 4]
            },
            'obs_polygons': {
                'number': len(obs_polygons),
                'vertexes_list': obs_polygons
            },
            'obs_lines': {
                'number': len(obs_lines),
                'obs_line_states': obs_lines
            }
        }
        
        return env_config
    
    def save_environment(self, env_config: Dict[str, Any], filename: str):
        """Save environment configuration to YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False, sort_keys=False)
    
    def generate_multiple_environments(self, robot_numbers: List[int], num_environments: int, output_dir: str):
        """Generate multiple random environments."""
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        for robot_num in robot_numbers:
            for i in range(num_environments):
                # Generate environment
                env_config = self.generate_environment(robot_num)
                
                # Create filename
                filename = f"random_env_{robot_num}agents_{i+1:03d}.yaml"
                filepath = os.path.join(output_dir, filename)
                
                # Save environment
                self.save_environment(env_config, filepath)
                generated_files.append(filepath)
                
                print(f"Generated: {filename}")
        
        return generated_files

def main():
    parser = argparse.ArgumentParser(description='Generate random test environments for RL-RVO navigation')
    parser.add_argument('--robot_numbers', nargs='+', type=int, default=[2, 4, 6], 
                       help='Number of agents per environment (default: 2 4 6)')
    parser.add_argument('--num_environments', type=int, default=10,
                       help='Number of environments to generate per agent count (default: 10)')
    parser.add_argument('--output_dir', type=str, default='random_environments',
                       help='Output directory for generated environments (default: random_environments)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate robot numbers
    valid_robot_numbers = [2, 4, 6]
    for num in args.robot_numbers:
        if num not in valid_robot_numbers:
            print(f"Warning: {num} is not in valid robot numbers {valid_robot_numbers}. Skipping.")
            args.robot_numbers.remove(num)
    
    if not args.robot_numbers:
        print("Error: No valid robot numbers provided.")
        return
    
    print(f"Generating {args.num_environments} environments for each of {args.robot_numbers} agents")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Create generator and generate environments
    generator = RandomEnvironmentGenerator(seed=args.seed)
    generated_files = generator.generate_multiple_environments(
        args.robot_numbers, 
        args.num_environments, 
        args.output_dir
    )
    
    print(f"\nGenerated {len(generated_files)} environment files:")
    for filepath in generated_files:
        print(f"  {filepath}")
    
    print(f"\nEnvironments saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
