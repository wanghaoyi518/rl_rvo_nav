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
        # print(f"🔍 DEBUG: PAREnvironment.build_par_environment called with {len(agent_states)} agents")
        # print(f"🔍 DEBUG: Participants: {self.participants}")
        # print(f"🔍 DEBUG: Agent states keys: {list(agent_states.keys())}")
        
        # Compute region boundaries
        self.compute_region_boundaries(agent_states)
        
        # Expand region with offset
        self.expand_region()
        
        # Build sub-map
        self.sub_map = self.build_sub_map()
        print(f"🔍 DEBUG: Sub-map built: {self.sub_map.width} x {self.sub_map.height}")
        
        # Build actor set
        self.actor_set = self.build_actor_set(agent_states)
        # print(f"🔍 DEBUG: Actor set built with {len(self.actor_set.actors)} actors")
        
        # print(f"🔍 DEBUG: build_par_environment completed")
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
                
                # Consider both position and goal
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                
                if position is not None:
                    x, y = position
                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)
                
                if goal is not None:
                    x, y = goal
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
        
        # Create sub-map with proper dimensions
        sub_map = SubMap([[0 for _ in range(width)] for _ in range(height)])
        sub_map.width = width
        sub_map.height = height
        sub_map.origin_x = self.min_x
        sub_map.origin_y = self.min_y
        sub_map.resolution = self.grid_resolution
        
        # Initialize grid with obstacles from the environment
        self._populate_obstacles(sub_map)
        
        return sub_map
    
    def _populate_obstacles(self, sub_map: SubMap):
        """Populate the sub-map with obstacles from the environment."""
        try:
            # Get obstacles from the workspace
            if hasattr(self.workspace, 'get') and callable(self.workspace.get):
                obstacles = self.workspace.get('obstacles', [])
            else:
                obstacles = getattr(self.workspace, 'obstacles', [])
            
            # print(f"🔍 PAR ENVIRONMENT: Found {len(obstacles)} obstacles in workspace")
            
            # Process each obstacle
            for obstacle in obstacles:
                self._add_obstacle_to_grid(sub_map, obstacle)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not populate obstacles: {e}")
            # Continue with free space if obstacle processing fails
    
    def _add_obstacle_to_grid(self, sub_map: SubMap, obstacle):
        """Add a single obstacle to the grid."""
        try:
            if hasattr(obstacle, 'pos') and hasattr(obstacle, 'radius'):
                # Circular obstacle
                center_x, center_y = obstacle.pos[0], obstacle.pos[1]
                radius = obstacle.radius
                self._add_circular_obstacle(sub_map, center_x, center_y, radius)
                
            elif hasattr(obstacle, 'vertices'):
                # Polygon obstacle
                vertices = obstacle.vertices
                self._add_polygon_obstacle(sub_map, vertices)
                
            elif isinstance(obstacle, (list, tuple)) and len(obstacle) >= 2:
                # Point obstacle
                x, y = obstacle[0], obstacle[1]
                self._add_point_obstacle(sub_map, x, y)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not process obstacle {obstacle}: {e}")
    
    def _add_circular_obstacle(self, sub_map: SubMap, center_x: float, center_y: float, radius: float):
        """Add a circular obstacle to the grid."""
        # Convert to grid coordinates
        grid_center_x = int((center_x - self.min_x) / self.grid_resolution)
        grid_center_y = int((center_y - self.min_y) / self.grid_resolution)
        grid_radius = int(radius / self.grid_resolution) + 1
        
        # Mark grid cells within the circle as obstacles
        for i in range(max(0, grid_center_y - grid_radius), min(sub_map.height, grid_center_y + grid_radius + 1)):
            for j in range(max(0, grid_center_x - grid_radius), min(sub_map.width, grid_center_x + grid_radius + 1)):
                # Check if cell is within circle
                if (i - grid_center_y) ** 2 + (j - grid_center_x) ** 2 <= grid_radius ** 2:
                    if 0 <= i < sub_map.height and 0 <= j < sub_map.width:
                        sub_map.grid[i][j] = 1  # Mark as obstacle
    
    def _add_polygon_obstacle(self, sub_map: SubMap, vertices):
        """Add a polygon obstacle to the grid."""
        # Convert vertices to grid coordinates
        grid_vertices = []
        for vertex in vertices:
            if len(vertex) >= 2:
                grid_x = int((vertex[0] - self.min_x) / self.grid_resolution)
                grid_y = int((vertex[1] - self.min_y) / self.grid_resolution)
                grid_vertices.append((grid_x, grid_y))
        
        if len(grid_vertices) < 3:
            return
        
        # Simple polygon filling (for convex polygons)
        self._fill_polygon_grid(sub_map, grid_vertices)
    
    def _add_point_obstacle(self, sub_map: SubMap, x: float, y: float):
        """Add a point obstacle to the grid."""
        grid_x = int((x - self.min_x) / self.grid_resolution)
        grid_y = int((y - self.min_y) / self.grid_resolution)
        
        if 0 <= grid_y < sub_map.height and 0 <= grid_x < sub_map.width:
            sub_map.grid[grid_y][grid_x] = 1  # Mark as obstacle
    
    def _fill_polygon_grid(self, sub_map: SubMap, grid_vertices):
        """Fill polygon area in the grid."""
        # Simple bounding box approach for convex polygons
        min_x = min(v[0] for v in grid_vertices)
        max_x = max(v[0] for v in grid_vertices)
        min_y = min(v[1] for v in grid_vertices)
        max_y = max(v[1] for v in grid_vertices)
        
        for i in range(max(0, min_y), min(sub_map.height, max_y + 1)):
            for j in range(max(0, min_x), min(sub_map.width, max_x + 1)):
                if self._point_in_polygon(j, i, grid_vertices):
                    if 0 <= i < sub_map.height and 0 <= j < sub_map.width:
                        sub_map.grid[i][j] = 1  # Mark as obstacle
    
    def _point_in_polygon(self, x: int, y: int, vertices) -> bool:
        """Check if a point is inside a polygon using ray casting."""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def build_actor_set(self, agent_states: Dict) -> ActorSet:
        """
        Build the actor set for participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            ActorSet: Built actor set
        """
        # print(f"🔍 DEBUG: build_actor_set called with {len(agent_states)} agents")
        # print(f"🔍 DEBUG: Participants: {self.participants}")
        
        # Ensure boundaries are computed and expanded before building actors
        if self.min_x == float('inf'):
            # print(f"🔍 DEBUG: Bounds not computed yet in build_actor_set, calling compute_region_boundaries and expand_region")
            self.compute_region_boundaries(agent_states)
            self.expand_region()
            # print(f"🔍 DEBUG: Updated bounds in build_actor_set - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        actor_set = ActorSet()
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                
                print(f"🔍 DEBUG: Agent {agent_id} - Raw position: {position}, Raw goal: {goal}")
                
                if position is not None:
                    x, y = position
                    # Convert to grid coordinates with proper rounding
                    grid_x = round((x - self.min_x) / self.grid_resolution)
                    grid_y = round((y - self.min_y) / self.grid_resolution)
                    
                    print(f"🔍 DEBUG: Agent {agent_id} - Grid start: ({grid_x}, {grid_y})")
                    print(f"🔍 DEBUG: Agent {agent_id} - Start calculation: ({x} - {self.min_x}) / {self.grid_resolution} = {grid_x}")
                    print(f"🔍 DEBUG: Agent {agent_id} - Start calculation: ({y} - {self.min_y}) / {self.grid_resolution} = {grid_y}")
                    
                    # Create actor with proper start and goal positions
                    from python_pnr.node import Point
                    start_point = Point(grid_x, grid_y)
                    
                    # Use actual goal position, not the same as start
                    if goal is not None:
                        goal_x, goal_y = goal
                        goal_grid_x = round((goal_x - self.min_x) / self.grid_resolution)
                        goal_grid_y = round((goal_y - self.min_y) / self.grid_resolution)
                        
                        print(f"🔍 DEBUG: Agent {agent_id} - Grid goal: ({goal_grid_x}, {goal_grid_y})")
                        print(f"🔍 DEBUG: Agent {agent_id} - Goal calculation: ({goal_x} - {self.min_x}) / {self.grid_resolution} = {goal_grid_x}")
                        print(f"🔍 DEBUG: Agent {agent_id} - Goal calculation: ({goal_y} - {self.min_y}) / {self.grid_resolution} = {goal_grid_y}")
                        
                        # Ensure goal is within the sub-map bounds
                        goal_grid_x = max(0, min(goal_grid_x, self.sub_map.width - 1)) if self.sub_map else goal_grid_x
                        goal_grid_y = max(0, min(goal_grid_y, self.sub_map.height - 1)) if self.sub_map else goal_grid_y
                        
                        print(f"🔍 DEBUG: Agent {agent_id} - Bounded grid goal: ({goal_grid_x}, {goal_grid_y})")
                        
                        goal_point = Point(goal_grid_x, goal_grid_y)
                    else:
                        # Fallback: use start position as goal if no goal available
                        goal_point = Point(grid_x, grid_y)
                        print(f"🔍 DEBUG: Agent {agent_id} - Using start as goal: ({grid_x}, {grid_y})")
                    
                    # Calculate and log the distance between start and goal
                    continuous_start = (x, y)
                    continuous_goal = (goal_x, goal_y) if goal is not None else (x, y)
                    continuous_distance = np.sqrt((continuous_goal[0] - continuous_start[0])**2 + (continuous_goal[1] - continuous_start[1])**2)
                    
                    grid_distance = np.sqrt((goal_point.x - start_point.x)**2 + (goal_point.y - start_point.y)**2)
                    
                    print(f"🔍 DEBUG: Agent {agent_id} - Final: start=({start_point.x}, {start_point.y}), goal=({goal_point.x}, {goal_point.y})")
                    print(f"🔍 DEBUG: Agent {agent_id} - Continuous distance: {continuous_distance:.4f}m, Grid distance: {grid_distance} cells")
                    print(f"🔍 DEBUG: Agent {agent_id} - Grid resolution: {self.grid_resolution}m, Min distance threshold: {self.grid_resolution/2:.4f}m")
                    
                    if continuous_distance < self.grid_resolution/2:
                        print(f"⚠️  WARNING: Agent {agent_id} continuous distance ({continuous_distance:.4f}m) < grid_resolution/2 ({self.grid_resolution/2:.4f}m)")
                        print(f"⚠️  This will cause start=goal in grid coordinates!")
                    
                    actor = Actor(agent_id, start_point, goal_point)
                    actor_set.add_actor(actor)
                else:
                    # print(f"🔍 DEBUG: Agent {agent_id} - No position found")
                    pass
        
        # print(f"🔍 DEBUG: build_actor_set completed with {len(actor_set.actors)} actors")
        return actor_set
    
    def compute_start_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute start positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to start positions
        """
        print(f"🔍 DEBUG: compute_start_positions called with {len(agent_states)} agents")
        
        # Ensure boundaries are computed first
        if self.min_x == float('inf'):
            print(f"🔍 DEBUG: Bounds not computed yet, calling compute_region_boundaries and expand_region")
            self.compute_region_boundaries(agent_states)
            self.expand_region()
        
        print(f"🔍 DEBUG: Current bounds - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        start_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                
                if position is not None:
                    x, y = position
                    print(f"🔍 DEBUG: Agent {agent_id} - Raw position: ({x}, {y})")
                    
                    # Convert to grid coordinates with proper rounding
                    grid_x = round((x - self.min_x) / self.grid_resolution)
                    grid_y = round((y - self.min_y) / self.grid_resolution)
                    
                    print(f"🔍 DEBUG: Agent {agent_id} - Grid position: ({grid_x}, {grid_y})")
                    print(f"🔍 DEBUG: Agent {agent_id} - Calculation: ({x} - {self.min_x}) / {self.grid_resolution} = {grid_x}")
                    print(f"🔍 DEBUG: Agent {agent_id} - Calculation: ({y} - {self.min_y}) / {self.grid_resolution} = {grid_y}")
                    
                    start_positions[agent_id] = (grid_x, grid_y)
                else:
                    # print(f"🔍 DEBUG: Agent {agent_id} - No position found in agent_state: {agent_state}")
                    pass
        
        # print(f"🔍 DEBUG: Final start_positions: {start_positions}")
        return start_positions
    
    def compute_goal_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute goal positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to goal positions
        """
        print(f"🔍 DEBUG: compute_goal_positions called with {len(agent_states)} agents")
        
        # Ensure boundaries are computed first
        if self.min_x == float('inf'):
            print(f"🔍 DEBUG: Bounds not computed yet, calling compute_region_boundaries and expand_region")
            self.compute_region_boundaries(agent_states)
            self.expand_region()
        
        print(f"🔍 DEBUG: Current bounds - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        goal_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                
                # Check if agent has already reached its goal
                if position is not None and goal is not None:
                    pos_x, pos_y = position
                    goal_x, goal_y = goal
                    continuous_distance = np.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
                    
                    if continuous_distance < self.grid_resolution/2:
                        print(f"🔍 DEBUG: Agent {agent_id} already at goal (distance: {continuous_distance:.4f}m), skipping from goal positions")
                        continue
                
                if goal is not None:
                    x, y = goal
                    print(f"🔍 DEBUG: Agent {agent_id} - Raw goal: ({x}, {y})")
                    
                    # Convert to grid coordinates with proper rounding
                    grid_x = round((x - self.min_x) / self.grid_resolution)
                    grid_y = round((y - self.min_y) / self.grid_resolution)
                    
                    print(f"🔍 DEBUG: Agent {agent_id} - Grid goal: ({grid_x}, {grid_y})")
                    print(f"🔍 DEBUG: Agent {agent_id} - Calculation: ({x} - {self.min_x}) / {self.grid_resolution} = {grid_x}")
                    print(f"🔍 DEBUG: Agent {agent_id} - Calculation: ({y} - {self.min_y}) / {self.grid_resolution} = {grid_y}")
                    
                    # Ensure goal is within the sub-map bounds
                    grid_x = max(0, min(grid_x, self.sub_map.width - 1)) if self.sub_map else grid_x
                    grid_y = max(0, min(grid_y, self.sub_map.height - 1)) if self.sub_map else grid_y
                    
                    print(f"🔍 DEBUG: Agent {agent_id} - Bounded grid goal: ({grid_x}, {grid_y})")
                    
                    goal_positions[agent_id] = (grid_x, grid_y)
                else:
                    print(f"🔍 DEBUG: Agent {agent_id} - No goal found in agent_state: {agent_state}")
        
        print(f"🔍 DEBUG: Final goal_positions: {goal_positions}")
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
        grid_x = round((x - self.min_x) / self.grid_resolution)
        grid_y = round((y - self.min_y) / self.grid_resolution)
        return (grid_x, grid_y)
    
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
