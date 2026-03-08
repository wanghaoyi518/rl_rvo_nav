from typing import List, Tuple, Optional
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from python_pnr.sub_map import SubMap
from python_pnr.isearch import ISearch


class GlobalPathPlanner:
    """A* based global path planner with waypoint sparsification.

    This planner reuses the existing python_pnr A* (ISearch) and SubMap grid.
    Occupancy grid building and inflation should follow the same policy as the
    PAR coordinator/environment to ensure consistency.
    """

    def __init__(self, grid: List[List[int]], resolution: float, waypoint_spacing: float):
        self._grid = grid
        self._resolution = float(resolution)
        self._waypoint_spacing = float(waypoint_spacing)
        self._sub_map = SubMap(self._grid)
        self._search = ISearch(self._sub_map)

    def plan_path(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Plan path from start to goal and return sparse waypoints in continuous coordinates.

        The returned waypoints are expressed in world coordinates (x, y).
        """
        # Map continuous world coords (x, y) to grid (j, i) indices
        start_i, start_j = self._world_to_grid(start_xy)
        goal_i, goal_j = self._world_to_grid(goal_xy)

        # Check if start and goal are within grid bounds
        if not self._is_grid_position_valid(start_i, start_j):
            return [goal_xy]
        if not self._is_grid_position_valid(goal_i, goal_j):
            return [goal_xy]

        # Define goal predicate in grid coordinates
        def is_goal(start, cur, sub_map, actor_set):
            return (cur.i == goal_i) and (cur.j == goal_j)

        # Run A* with correct goal indices so heuristic targets the actual goal
        result = self._search.startSearch(self._sub_map, None, start_i, start_j, goal_i, goal_j, is_goal, True, True, 0, -1, -1, set())

        if not getattr(result, 'pathfound', False):
            return [goal_xy]

        dense_grid_path: List[Tuple[int, int]] = [(p.i, p.j) for p in getattr(result, 'lppath', [])]
        dense_world_path: List[Tuple[float, float]] = [self._grid_to_world((i, j)) for (i, j) in dense_grid_path]

        # Sparsify by distance threshold with simple LOS skip disabled (keep minimalism)
        sparse: List[Tuple[float, float]] = []
        last_kept: Optional[Tuple[float, float]] = None
        for pt in dense_world_path:
            if last_kept is None:
                sparse.append(pt)
                last_kept = pt
                continue
            if self._dist2(pt, last_kept) >= self._waypoint_spacing * self._waypoint_spacing:
                sparse.append(pt)
                last_kept = pt
        if len(sparse) == 0 or sparse[-1] != dense_world_path[-1]:
            sparse.append(dense_world_path[-1])


        return sparse

    def separate_waypoints(self, waypoint_lists: List[List[Tuple[float, float]]], min_distance: float = 1.0) -> List[List[Tuple[float, float]]]:
        """
        Separate waypoints from multiple agents to avoid conflicts.
        Ensures minimum Manhattan distance between any two waypoints from different agents.
        Also ensures waypoints do not fall on obstacles.
        
        Args:
            waypoint_lists: List of waypoint lists for each agent
            min_distance: Minimum Manhattan distance between waypoints
            
        Returns:
            List of separated waypoint lists
        """
        if len(waypoint_lists) <= 1:
            return waypoint_lists
            
        separated_lists = []
        used_positions = set()
        
        for agent_id, waypoints in enumerate(waypoint_lists):
            separated_waypoints = []
            
            for waypoint in waypoints:
                # Try to find a non-conflicting position that avoids obstacles
                new_waypoint = self._find_separated_position(waypoint, used_positions, min_distance)
                separated_waypoints.append(new_waypoint)
                used_positions.add(new_waypoint)
            
            separated_lists.append(separated_waypoints)
            
        return separated_lists
    
    def _find_separated_position(self, original_pos: Tuple[float, float], used_positions: set, min_distance: float) -> Tuple[float, float]:
        """
        Find a position that maintains minimum distance from used positions and avoids obstacles.
        Uses Manhattan distance for simplicity.
        """
        # Check if original position is valid (not on obstacle and separated from used positions)
        if self._is_position_valid(original_pos, used_positions, min_distance):
            return original_pos
            
        # Try to find a nearby valid position
        max_attempts = 20
        for attempt in range(max_attempts):
            # Generate candidate positions in expanding rings
            radius = min_distance + attempt * 0.5
            candidates = self._generate_candidate_positions(original_pos, radius)
            
            for candidate in candidates:
                if self._is_position_valid(candidate, used_positions, min_distance):
                    return candidate
                    
        # If no valid position found, return original (fallback)
        print(f"WARNING: Could not find valid separated position for {original_pos}, using original")
        return original_pos
    
    def _is_position_valid(self, pos: Tuple[float, float], used_positions: set, min_distance: float) -> bool:
        """Check if position is valid: not on obstacle and maintains minimum distance from used positions."""
        # First check if position is on an obstacle
        if self._is_position_on_obstacle(pos):
            return False
            
        # Then check if position maintains minimum distance from used positions
        return self._is_position_separated(pos, used_positions, min_distance)
    
    def _is_position_on_obstacle(self, pos: Tuple[float, float]) -> bool:
        """Check if the given position falls on an obstacle in the grid."""
        try:
            # Convert continuous coordinates to grid coordinates
            grid_i, grid_j = self._world_to_grid(pos)
            
            # Use the new boundary check method
            return not self._is_grid_position_valid(grid_i, grid_j)
            
        except Exception as e:
            print(f"ERROR checking obstacle at {pos}: {e}")
            return True  # If error, consider it obstacle for safety
    
    def _is_position_separated(self, pos: Tuple[float, float], used_positions: set, min_distance: float) -> bool:
        """Check if position maintains minimum distance from all used positions."""
        for used_pos in used_positions:
            manhattan_dist = abs(pos[0] - used_pos[0]) + abs(pos[1] - used_pos[1])
            if manhattan_dist < min_distance:
                return False
        return True
    
    def _generate_candidate_positions(self, center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        """Generate candidate positions around a center point."""
        candidates = []
        center_x, center_y = center
        
        # Generate positions in a square pattern around the center
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                if dx == 0 and dy == 0:
                    continue
                candidate = (center_x + dx, center_y + dy)
                candidates.append(candidate)
                
        return candidates

    def _world_to_grid(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        # Use floor-style indexing consistent with PAR (no +0.5 here)
        j = int(x / self._resolution)
        i = int(y / self._resolution)
        return i, j

    def _is_grid_position_valid(self, grid_i: int, grid_j: int) -> bool:
        """Check if grid position is within bounds and not on obstacle."""
        # Check bounds (grid indices should be in range [0, len-1])
        if (grid_i < 0 or grid_i >= len(self._grid) or 
            grid_j < 0 or grid_j >= len(self._grid[0])):
            print(f"DEBUG BOUNDS: Position ({grid_i}, {grid_j}) out of bounds. Grid size: {len(self._grid)} x {len(self._grid[0])}")
            return False
        
        # Check if position is on obstacle
        is_free = self._grid[grid_i][grid_j] == 0
        if not is_free:
            print(f"DEBUG OBSTACLE: Position ({grid_i}, {grid_j}) is on obstacle")
        return is_free

    def _grid_to_world(self, ij: Tuple[int, int]) -> Tuple[float, float]:
        i, j = int(ij[0]), int(ij[1])
        # Return cell centers to match PAR/logging convention
        x = (float(j) + 0.5) * self._resolution
        y = (float(i) + 0.5) * self._resolution
        return (x, y)

    def _dist2(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return dx * dx + dy * dy


