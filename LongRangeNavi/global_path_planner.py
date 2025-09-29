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

        # Define goal predicate in grid coordinates
        def is_goal(start, cur, sub_map, actor_set):
            return (cur.i == goal_i) and (cur.j == goal_j)

        # Run A*
        result = self._search.startSearch(self._sub_map, None, start_i, start_j, 0, 0, is_goal, True, True, 0, -1, -1, set())

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

    def _world_to_grid(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        j = int(round(x / self._resolution))
        i = int(round(y / self._resolution))
        return i, j

    def _grid_to_world(self, ij: Tuple[int, int]) -> Tuple[float, float]:
        i, j = int(ij[0]), int(ij[1])
        x = float(j) * self._resolution
        y = float(i) * self._resolution
        return (x, y)

    def _dist2(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return dx * dx + dy * dy


