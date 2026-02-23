"""
CBS Coordinator Module

Provides Conflict-Based Search (CBS) as an alternative to PAR for deadlock resolution.
Interface aligned with PARCoordinator: prepare_cbs_execution(agent_states, participants)
and get_agent_path(agent_id) returning continuous waypoints for waypoint injection.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class CBSCoordinator:
    """
    Coordinator for CBS (Conflict-Based Search) path planning.
    Uses the same workspace/grid as long-range and PAR; returns continuous paths
    so that ir_gym can inject waypoints without grid_to_continuous conversion.
    """

    def __init__(self, config: Dict, gym_env=None):
        """
        Args:
            config: Config dict with CBS_ROBOT_RADIUS, CBS_GRID_RESOLUTION (optional),
                    CBS_MAX_ITER, CBS_LOW_LEVEL_MAX_ITER (optional). Uses GRID_RESOLUTION
                    from config if CBS_GRID_RESOLUTION not set.
            gym_env: Reference to gym env for grid building and bounds.
        """
        self.config = config if isinstance(config, dict) else {}
        self.gym_env = gym_env
        self.current_cbs_solution = {}
        self.current_participants: List[int] = []
        self._resolution: Optional[float] = None
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._grid_height: int = 0
        self._grid_width: int = 0

    def prepare_cbs_execution(
        self, agent_states: Dict, deadlock_participants: List[int]
    ) -> Optional["CBSCoordinator"]:
        """
        Plan collision-free paths for participants using CBS.
        Aligned with PARCoordinator.prepare_par_execution: same inputs. Returns
        self on success so callers use get_agent_path(agent_id) for waypoint injection.

        Args:
            agent_states: Dict of agent_id -> {position, goal, ...} (same as PAR).
            deadlock_participants: List of agent IDs to plan for.

        Returns:
            self (for compatibility with PAR flow that checks solution validity),
            or None if CBS failed. Paths are stored; use get_agent_path(agent_id)
            to get continuous waypoints per agent.
        """
        self.current_cbs_solution = {}
        self.current_participants = list(deadlock_participants)

        if not deadlock_participants or not agent_states:
            return None

        try:
            grid, resolution, world_w, world_h = self.gym_env._build_occupancy_grid_for_long_range()
        except Exception as e:
            print(f"CBS: Failed to build occupancy grid: {e}")
            return None

        if grid is None or resolution is None or resolution <= 0:
            print("CBS: Invalid grid or resolution")
            return None

        self._resolution = float(resolution)
        self._offset_x = float(getattr(self.gym_env, "offset_x", 0.0))
        self._offset_y = float(getattr(self.gym_env, "offset_y", 0.0))
        self._grid_height = len(grid)
        self._grid_width = len(grid[0]) if self._grid_height else 0

        static_obstacles = []
        for row in range(self._grid_height):
            for col in range(self._grid_width):
                if grid[row][col] != 0:
                    static_obstacles.append((col, row))

        robot_radius_world = float(
            self.config.get("CBS_ROBOT_RADIUS", self.config.get("GRID_RESOLUTION", 0.5) * 0.5)
        )
        robot_radius_cells = max(1, int(round(robot_radius_world / self._resolution)))

        starts_list = []
        goals_list = []
        order_ids = []

        for pid in deadlock_participants:
            if pid not in agent_states:
                continue
            state = agent_states[pid]
            pos = self._get_position(state)
            goal = self._get_goal(state)
            if pos is None:
                continue
            if goal is None:
                goal = (pos[0] + 0.5, pos[1])

            c_start = self._continuous_to_grid(pos[0], pos[1])
            c_goal = self._continuous_to_grid(goal[0], goal[1])
            if c_start is None or c_goal is None:
                continue
            starts_list.append(c_start)
            goals_list.append(c_goal)
            order_ids.append(pid)

        if len(starts_list) != len(deadlock_participants) or len(starts_list) == 0:
            print("CBS: Missing start/goal for some participants")
            return None

        try:
            from cbs_mapf.planner import Planner
        except ImportError as e:
            print(f"CBS: cbs_mapf not available: {e}")
            return None

        max_iter = int(self.config.get("CBS_MAX_ITER", 200))
        low_level_max_iter = int(self.config.get("CBS_LOW_LEVEL_MAX_ITER", 100))

        planner = Planner(
            grid_size=1,
            robot_radius=robot_radius_cells,
            static_obstacles=static_obstacles,
        )
        result = planner.plan(
            starts_list,
            goals_list,
            max_iter=max_iter,
            low_level_max_iter=low_level_max_iter,
        )

        if result is None or (isinstance(result, np.ndarray) and result.size == 0):
            print("CBS: No solution found")
            return None

        result = np.asarray(result)
        if result.ndim != 3 or result.shape[0] != len(order_ids):
            print("CBS: Unexpected result shape")
            return None

        for i, pid in enumerate(order_ids):
            path = result[i]
            cont_path = []
            for t in range(path.shape[0]):
                col, row = int(path[t, 0]), int(path[t, 1])
                x, y = self._grid_to_continuous(col, row)
                cont_path.append((x, y))
            self.current_cbs_solution[pid] = cont_path

        return self

    def get_agent_path(self, agent_id: int) -> List[Tuple[float, float]]:
        """
        Return continuous waypoints for the agent (same role as PARCoordinator.get_agent_path
        but already in continuous space so ir_gym can inject without grid_to_continuous).

        Args:
            agent_id: Agent ID.

        Returns:
            List of (x, y) continuous waypoints, or empty list if no path.
        """
        return self.current_cbs_solution.get(agent_id, [])

    def _get_position(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        if "position" in agent_state:
            p = agent_state["position"]
            if isinstance(p, (list, np.ndarray)) and len(p) >= 2:
                return (float(p[0]), float(p[1]))
        for key in ("pos", "location", "pose"):
            if key in agent_state:
                p = agent_state[key]
                if isinstance(p, (list, np.ndarray)) and len(p) >= 2:
                    return (float(p[0]), float(p[1]))
        return None

    def _get_goal(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        goal = agent_state.get("goal")
        if goal is None:
            for key in ("target", "destination", "end_pos"):
                if key in agent_state:
                    goal = agent_state[key]
                    break
        if goal is None:
            return None
        if hasattr(goal, "shape") and getattr(goal, "shape", ()) == (2, 1):
            return (float(goal[0, 0]), float(goal[1, 0]))
        if isinstance(goal, (list, np.ndarray)) and len(goal) >= 2:
            return (float(goal[0]), float(goal[1]))
        return None

    def _continuous_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self._resolution is None or self._resolution <= 0:
            return None
        col = int((float(x) - self._offset_x) / self._resolution)
        row = int((float(y) - self._offset_y) / self._resolution)
        col = max(0, min(col, self._grid_width - 1))
        row = max(0, min(row, self._grid_height - 1))
        return (col, row)

    def _grid_to_continuous(self, col: int, row: int) -> Tuple[float, float]:
        x = self._offset_x + (int(col) + 0.5) * self._resolution
        y = self._offset_y + (int(row) + 0.5) * self._resolution
        return (x, y)

    def reset(self) -> None:
        self.current_cbs_solution = {}
        self.current_participants = []
