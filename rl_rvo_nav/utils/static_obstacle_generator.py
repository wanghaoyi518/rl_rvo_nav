#!/usr/bin/env python3
"""
Random static obstacle generator for varying obstacle layouts in each episode
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any

class StaticObstacleGenerator:
    """Generate random static obstacles with varying layouts for each episode"""
    
    def __init__(self, 
                 world_width: float = 12.0,
                 world_height: float = 12.0,
                 obstacle_size: float = 0.8,
                 obstacle_count: int = 4,
                 min_obstacle_distance: float = 1.5,
                 margin: float = 1.0,
                 max_attempts: int = 100):
        """
        Initialize random static obstacle generator
        
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
            Tuple of (obs_polygons, obs_lines) for yaml configuration
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
    
    def create_world_config(self, 
                           base_config: Dict[str, Any],
                           robot_positions: List[Tuple[float, float]] = None,
                           robot_goals: List[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Create world configuration with random obstacles
        
        Args:
            base_config: Base world configuration dictionary
            robot_positions: Robot initial positions
            robot_goals: Robot goal positions
            
        Returns:
            Updated world configuration with random obstacles
        """
        obs_polygons, obs_lines = self.generate_random_obstacles(robot_positions, robot_goals)
        
        # Update configuration
        config = base_config.copy()
        config['obs_polygons'] = {
            'number': len(obs_polygons),
            'vertexes_list': obs_polygons
        }
        config['obs_lines'] = {
            'number': len(obs_lines),
            'obs_line_states': obs_lines
        }
        
        return config
    
    def save_config_to_file(self, 
                           config: Dict[str, Any], 
                           filepath: str) -> None:
        """Save configuration to YAML file"""
        import yaml
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
    def visualize_obstacles(self, obs_polygons: List[List[List[float]]]) -> None:
        """Visualize obstacle layout (for debugging)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            
            # Draw world boundaries
            ax.add_patch(patches.Rectangle((0, 0), self.world_width, self.world_height, 
                                         linewidth=2, edgecolor='black', facecolor='none'))
            
            # Draw obstacles
            for polygon in obs_polygons:
                x_coords = [point[0] for point in polygon] + [polygon[0][0]]
                y_coords = [point[1] for point in polygon] + [polygon[0][1]]
                
                ax.add_patch(patches.Polygon(polygon, closed=True, 
                                           facecolor='red', alpha=0.5, edgecolor='darkred'))
            
            ax.set_xlim(-0.5, self.world_width + 0.5)
            ax.set_ylim(-0.5, self.world_height + 0.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Random Obstacle Layout ({len(obs_polygons)} obstacles)')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Test the generator
    generator = StaticObstacleGenerator()
    
    # Example robot positions
    robot_positions = [(1, 1), (8, 8), (2, 8), (8, 2)]
    robot_goals = [(8, 8), (1, 1), (8, 2), (2, 8)]
    
    obs_polygons, obs_lines = generator.generate_random_obstacles(robot_positions, robot_goals)
    
    print(f"Generated {len(obs_polygons)} obstacles:")
    for i, polygon in enumerate(obs_polygons):
        print(f"Obstacle {i+1}: {polygon}")
        
    # Visualize if possible
    generator.visualize_obstacles(obs_polygons)
