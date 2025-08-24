#!/usr/bin/env python3
"""
Runtime obstacle manager for dynamically adding/removing obstacles during environment execution
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from ir_sim.world import obs_polygon
from ir_sim.env import env_obs_line, env_obs_poly

class RuntimeObstacleManager:
    """Manage obstacles dynamically during environment runtime without yaml files"""
    
    def __init__(self, 
                 world_width: float = 12.0,
                 world_height: float = 12.0,
                 obstacle_size: float = 0.8,
                 obstacle_count: int = 4,
                 min_obstacle_distance: float = 1.5,
                 margin: float = 1.0,
                 max_attempts: int = 100):
        """
        Initialize runtime obstacle manager
        
        Args:
            world_width: Width of the world
            world_height: Height of the world
            obstacle_size: Size of square obstacles (1x1)
            obstacle_count: Number of obstacles to generate (4)
            min_obstacle_distance: Minimum distance between obstacles (2.0)
            margin: Margin from world boundaries
            max_attempts: Maximum attempts to place obstacles
        """
        self.world_width = world_width
        self.world_height = world_height
        self.obstacle_size = obstacle_size
        self.obstacle_count = obstacle_count
        self.min_obstacle_distance = min_obstacle_distance
        self.margin = margin
        self.max_attempts = max_attempts
        
    def generate_random_obstacles(self, 
                                robot_positions: List[Tuple[float, float]] = None,
                                robot_goals: List[Tuple[float, float]] = None) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """
        Generate random static obstacles avoiding robot positions and goals
        
        Args:
            robot_positions: List of robot initial positions [(x, y), ...]
            robot_goals: List of robot goal positions [(x, y), ...]
            
        Returns:
            Tuple of (obs_polygons, obs_lines) for runtime injection
        """
        # Collect all forbidden positions
        forbidden_positions = []
        if robot_positions:
            forbidden_positions.extend(robot_positions)
        if robot_goals:
            forbidden_positions.extend(robot_goals)
            
        # Generate obstacle positions
        obstacle_positions = self._generate_obstacle_positions(forbidden_positions)
        
        # Convert to polygon format
        obs_polygons = []
        obs_lines = []
        
        for pos in obstacle_positions:
            x, y = pos
            # Create 1x1 square obstacle
            polygon = [
                [x, y],                     # Bottom-left
                [x + self.obstacle_size, y], # Bottom-right  
                [x + self.obstacle_size, y + self.obstacle_size], # Top-right
                [x, y + self.obstacle_size]  # Top-left
            ]
            obs_polygons.append(polygon)
            
            # Create lines for each edge
            lines = [
                [x, y, x + self.obstacle_size, y],                           # Bottom edge
                [x + self.obstacle_size, y, x + self.obstacle_size, y + self.obstacle_size], # Right edge
                [x + self.obstacle_size, y + self.obstacle_size, x, y + self.obstacle_size], # Top edge
                [x, y + self.obstacle_size, x, y]                            # Left edge
            ]
            obs_lines.extend(lines)
            
        return obs_polygons, obs_lines
    
    def _generate_obstacle_positions(self, forbidden_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Generate valid obstacle positions"""
        positions = []
        
        for i in range(self.obstacle_count):
            attempts = 0
            while attempts < self.max_attempts:
                # Generate random position within valid bounds
                x = random.uniform(self.margin, 
                                 self.world_width - self.obstacle_size - self.margin)
                y = random.uniform(self.margin, 
                                 self.world_height - self.obstacle_size - self.margin)
                
                new_pos = (x, y)
                
                # Check if position is valid
                if self._is_position_valid(new_pos, positions, forbidden_positions):
                    positions.append(new_pos)
                    break
                    
                attempts += 1
            
            if attempts >= self.max_attempts:
                print(f"Warning: Could not place obstacle {i+1} after {self.max_attempts} attempts")
                
        return positions
    
    def _is_position_valid(self, 
                          new_pos: Tuple[float, float], 
                          existing_positions: List[Tuple[float, float]],
                          forbidden_positions: List[Tuple[float, float]]) -> bool:
        """Check if obstacle position is valid"""
        x, y = new_pos
        
        # Check distance from existing obstacles
        for ex_x, ex_y in existing_positions:
            # Calculate distance between obstacle centers
            center_dist = np.sqrt((x + self.obstacle_size/2 - ex_x - self.obstacle_size/2)**2 + 
                                (y + self.obstacle_size/2 - ex_y - self.obstacle_size/2)**2)
            
            # Check minimum distance (center to center + obstacle sizes)
            min_required_dist = self.min_obstacle_distance + self.obstacle_size
            if center_dist < min_required_dist:
                return False
        
        # Check distance from forbidden positions (robot positions/goals)
        for fx, fy in forbidden_positions:
            # Distance from obstacle center to forbidden point
            obstacle_center_x = x + self.obstacle_size / 2
            obstacle_center_y = y + self.obstacle_size / 2
            
            dist = np.sqrt((obstacle_center_x - fx)**2 + (obstacle_center_y - fy)**2)
            
            # Ensure sufficient clearance (at least 1.5 for robot safety)
            min_clearance = self.obstacle_size / 2 + 1.5
            if dist < min_clearance:
                return False
                
        return True
    
    def inject_obstacles_to_environment(self, env, robot_positions=None, robot_goals=None):
        """
        Directly inject random obstacles into a running environment
        
        Args:
            env: The gym environment instance
            robot_positions: Optional robot positions to avoid
            robot_goals: Optional robot goals to avoid
        """
        # Generate new obstacle layout
        obs_polygons, obs_lines = self.generate_random_obstacles(robot_positions, robot_goals)
        
        # Access the base environment (ir_gym -> env_base)
        if hasattr(env, 'ir_gym'):
            base_env = env.ir_gym
        else:
            base_env = env
            
        # Create new polygon obstacles
        new_poly_list = []
        for polygon in obs_polygons:
            poly_obj = obs_polygon(vertex=polygon)
            new_poly_list.append(poly_obj)
        
        # Update the polygon component
        if hasattr(base_env, 'components') and 'obs_polygons' in base_env.components:
            base_env.components['obs_polygons'].obs_poly_list = new_poly_list
            base_env.obs_poly_list = new_poly_list
            # Update count
            base_env.obs_poly_num = len(obs_polygons)
        
        # Update line obstacles
        if hasattr(base_env, 'components') and 'obs_lines' in base_env.components:
            base_env.components['obs_lines'].obs_line_states = obs_lines
            base_env.obs_line_states = obs_lines
        
        print(f"Injected {len(obs_polygons)} obstacles and {len(obs_lines)} lines into environment")
        
    def clear_obstacles_from_environment(self, env):
        """
        Clear all obstacles from environment
        
        Args:
            env: The gym environment instance
        """
        # Access the base environment (ir_gym -> env_base)
        if hasattr(env, 'ir_gym'):
            base_env = env.ir_gym
        else:
            base_env = env
            
        # Clear polygon obstacles
        if hasattr(base_env, 'components') and 'obs_polygons' in base_env.components:
            base_env.components['obs_polygons'].obs_poly_list = []
            base_env.obs_poly_list = []
            base_env.obs_poly_num = 0
        
        # Clear line obstacles
        if hasattr(base_env, 'components') and 'obs_lines' in base_env.components:
            base_env.components['obs_lines'].obs_line_states = []
            base_env.obs_line_states = []
            
        print("Cleared all obstacles from environment")


class RuntimeObstacleEnvWrapper:
    """Gym environment wrapper that injects random obstacles at runtime"""
    
    def __init__(self, base_env, obstacle_manager: RuntimeObstacleManager):
        """
        Initialize wrapper
        
        Args:
            base_env: Base gym environment 
            obstacle_manager: Runtime obstacle manager
        """
        self.base_env = base_env
        self.obstacle_manager = obstacle_manager
        self.episode_count = 0
        
    def reset(self, mode=0, **kwargs):
        """Reset environment with new random obstacles"""
        self.episode_count += 1
        
        # Reset base environment first
        obs_list = self.base_env.reset(mode=mode, **kwargs)
        
        # Clear existing obstacles
        self.obstacle_manager.clear_obstacles_from_environment(self.base_env)
        
        # Get robot positions and goals (if available)
        robot_positions = None
        robot_goals = None
        
        try:
            if hasattr(self.base_env, 'ir_gym') and hasattr(self.base_env.ir_gym, 'robot_list'):
                robot_list = self.base_env.ir_gym.robot_list
                robot_positions = [(robot.state[0], robot.state[1]) for robot in robot_list]
                robot_goals = [(robot.goal[0], robot.goal[1]) for robot in robot_list]
        except:
            pass  # If we can't get robot positions, proceed without them
            
        # Inject new random obstacles
        self.obstacle_manager.inject_obstacles_to_environment(
            self.base_env, robot_positions, robot_goals
        )
        
        print(f"Episode {self.episode_count}: Generated new random obstacle layout")
        
        return obs_list
    
    def step_ir(self, action, **kwargs):
        """Step the environment"""
        return self.base_env.step_ir(action, **kwargs)
    
    def render(self, **kwargs):
        """Render the environment"""
        return self.base_env.render(**kwargs)
    
    def reset_one(self, id):
        """Reset one robot"""
        return self.base_env.reset_one(id)
    
    def show(self):
        """Show environment"""
        return self.base_env.show()
    
    @property
    def ir_gym(self):
        """Access to underlying ir_gym"""
        return self.base_env.ir_gym
    
    @property
    def observation_space(self):
        """Observation space"""
        return self.base_env.observation_space
    
    @property
    def action_space(self):
        """Action space"""
        return self.base_env.action_space
        
    def cleanup(self):
        """Cleanup resources"""
        # No temporary files to clean up!
        pass


def create_runtime_obstacle_env(base_world_config: str, **env_kwargs):
    """
    Create a gym environment with runtime obstacle injection
    
    Args:
        base_world_config: Path to base world configuration
        **env_kwargs: Additional environment arguments
        
    Returns:
        Runtime obstacle environment wrapper
    """
    import gym
    import gym_env
    
    # Create base environment (without obstacles)
    base_env = gym.make('mrnav-v1', world_name=base_world_config, **env_kwargs)
    
    # Create obstacle manager with optimized parameters
    obstacle_manager = RuntimeObstacleManager(
        world_width=12.0,
        world_height=12.0,
        obstacle_count=4,
        obstacle_size=0.8,
        min_obstacle_distance=1.5
    )
    
    # Create wrapper
    wrapped_env = RuntimeObstacleEnvWrapper(base_env, obstacle_manager)
    
    return wrapped_env
