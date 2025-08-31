"""
PAR Environment Builder Module

This module provides functionality to build the environment for Push and Rotate (PAR) algorithm.
It handles sub-map construction, start/goal position computation, and region expansion.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from python_pnr.sub_map import SubMap
from python_pnr.actor_set import ActorSet
from python_pnr.actor import Actor


class PAREnvironment:
    """
    Environment builder for Push and Rotate (PAR) algorithm.
    
    This class handles the construction of the local environment for PAR execution,
    including sub-map creation, start/goal position computation, and region expansion.
    """
    
    def __init__(self, workspace: Dict, participants: List[int], config: Dict):
        """
        Initialize the PAR environment builder.
        
        Args:
            workspace: Workspace configuration dictionary
            participants: List of agent IDs participating in PAR
            config: Configuration dictionary containing PAR parameters
        """
        self.workspace = workspace
        self.participants = participants
        self.config = config
        self.par_offset = config.get('PAR_OFFSET', 2)
        self.grid_resolution = config.get('GRID_RESOLUTION', 1.0)
        
        # Environment boundaries
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        
        # Sub-map and actor set
        self.sub_map = None
        self.actor_set = None
    
    def build_par_environment(self, agent_states: Dict) -> Tuple[SubMap, ActorSet]:
        """
        Build the complete PAR environment including sub-map and actor set.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Tuple[SubMap, ActorSet]: Built sub-map and actor set for PAR
        """
        # Compute region boundaries
        self.compute_region_boundaries(agent_states)
        
        # Expand region with offset
        self.expand_region()
        
        # Build sub-map
        self.sub_map = self.build_sub_map()
        
        # Build actor set
        self.actor_set = self.build_actor_set(agent_states)
        
        return self.sub_map, self.actor_set
    
    def compute_region_boundaries(self, agent_states: Dict):
        """
        Compute the boundaries of the region containing all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
        """
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                
                if position is not None:
                    x, y = position
                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)
    
    def expand_region(self):
        """Expand the region boundaries by the specified offset."""
        self.min_x -= self.par_offset
        self.max_x += self.par_offset
        self.min_y -= self.par_offset
        self.max_y += self.par_offset
    
    def build_sub_map(self) -> SubMap:
        """
        Build the sub-map for the PAR region.
        
        Returns:
            SubMap: Built sub-map
        """
        # Calculate grid dimensions
        width = int((self.max_x - self.min_x) / self.grid_resolution) + 1
        height = int((self.max_y - self.min_y) / self.grid_resolution) + 1
        
        # Create sub-map
        # Create a simple grid for SubMap (all free space)
        grid_size = 5  # 20x20 grid
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        sub_map = SubMap(grid)
        sub_map.width = width
        sub_map.height = height
        sub_map.origin_x = self.min_x
        sub_map.origin_y = self.min_y
        sub_map.resolution = self.grid_resolution
        
        # Initialize grid with free space (assuming no obstacles in the region)
        # In a real implementation, you would check for obstacles here
        sub_map.grid = [[0 for _ in range(width)] for _ in range(height)]
        
        return sub_map
    
    def build_actor_set(self, agent_states: Dict) -> ActorSet:
        """
        Build the actor set for participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            ActorSet: Built actor set
        """
        actor_set = ActorSet()
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                
                if position is not None:
                    x, y = position
                    # Convert to grid coordinates
                    grid_x = int((x - self.min_x) / self.grid_resolution)
                    grid_y = int((y - self.min_y) / self.grid_resolution)
                    
                    # Create actor
                    from python_pnr.node import Point
                    start_point = Point(grid_x, grid_y)
                    goal_point = Point(grid_x, grid_y)  # For now, use same position as goal
                    actor = Actor(agent_id, start_point, goal_point)
                    actor_set.add_actor(actor)
        
        return actor_set
    
    def compute_start_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute start positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to start positions
        """
        start_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                
                if position is not None:
                    x, y = position
                    # Convert to grid coordinates
                    grid_x = int((x - self.min_x) / self.grid_resolution)
                    grid_y = int((y - self.min_y) / self.grid_resolution)
                    
                    start_positions[agent_id] = (grid_x, grid_y)
        
        return start_positions
    
    def compute_goal_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute goal positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to goal positions
        """
        goal_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                goal = self.get_agent_goal(agent_state)
                
                if goal is not None:
                    x, y = goal
                    # Convert to grid coordinates
                    grid_x = int((x - self.min_x) / self.grid_resolution)
                    grid_y = int((y - self.min_y) / self.grid_resolution)
                    
                    # Ensure goal is within the sub-map bounds
                    grid_x = max(0, min(grid_x, self.sub_map.width - 1))
                    grid_y = max(0, min(grid_y, self.sub_map.height - 1))
                    
                    goal_positions[agent_id] = (grid_x, grid_y)
        
        return goal_positions
    
    def get_agent_position(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        """
        Extract agent position from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Optional[Tuple[float, float]]: Agent position (x, y) or None if not available
        """
        if 'position' in agent_state:
            position = agent_state['position']
            if isinstance(position, (list, np.ndarray)) and len(position) >= 2:
                return (float(position[0]), float(position[1]))
        
        # Try alternative position fields
        for field in ['pos', 'location', 'pose']:
            if field in agent_state:
                pos_data = agent_state[field]
                if isinstance(pos_data, (list, np.ndarray)) and len(pos_data) >= 2:
                    return (float(pos_data[0]), float(pos_data[1]))
        
        return None
    
    def get_agent_goal(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        """
        Extract agent goal from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Optional[Tuple[float, float]]: Agent goal (x, y) or None if not available
        """
        if 'goal' in agent_state:
            goal = agent_state['goal']
            if isinstance(goal, (list, np.ndarray)) and len(goal) >= 2:
                return (float(goal[0]), float(goal[1]))
        
        # Try alternative goal fields
        for field in ['target', 'destination', 'end_pos']:
            if field in agent_state:
                goal_data = agent_state[field]
                if isinstance(goal_data, (list, np.ndarray)) and len(goal_data) >= 2:
                    return (float(goal_data[0]), float(goal_data[1]))
        
        return None
    
    def grid_to_continuous(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert grid coordinates to continuous coordinates.
        
        Args:
            grid_pos: Grid position (x, y)
            
        Returns:
            Tuple[float, float]: Continuous position (x, y)
        """
        grid_x, grid_y = grid_pos
        x = self.min_x + grid_x * self.grid_resolution
        y = self.min_y + grid_y * self.grid_resolution
        return (x, y)
    
    def continuous_to_grid(self, continuous_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert continuous coordinates to grid coordinates.
        
        Args:
            continuous_pos: Continuous position (x, y)
            
        Returns:
            Tuple[int, int]: Grid position (x, y)
        """
        x, y = continuous_pos
        grid_x = int((x - self.min_x) / self.grid_resolution)
        grid_y = int((y - self.min_y) / self.grid_resolution)
        return (grid_x, grid_y)
    
    def grid_to_continuous(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert grid coordinates to continuous coordinates.
        
        Args:
            grid_pos: Position in grid coordinates (x, y)
            
        Returns:
            Tuple[float, float]: Position in continuous coordinates (x, y)
        """
        grid_x, grid_y = grid_pos
        x = self.min_x + grid_x * self.grid_resolution
        y = self.min_y + grid_y * self.grid_resolution
        return (x, y)
    
    def is_position_in_bounds(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is within the PAR region bounds.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            bool: True if position is within bounds
        """
        x, y = position
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)
    
    def get_region_info(self) -> Dict:
        """
        Get information about the PAR region.
        
        Returns:
            Dict: Region information including boundaries and dimensions
        """
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y,
            'width': self.max_x - self.min_x,
            'height': self.max_y - self.min_y,
            'participants': self.participants,
            'grid_resolution': self.grid_resolution
        }
