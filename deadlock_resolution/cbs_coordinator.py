"""
CBS Coordinator Module

Provides Conflict-Based Search (CBS) as an alternative to PAR for deadlock resolution.
Interface aligned with PARCoordinator: prepare_cbs_execution(agent_states, participants)
and get_agent_path(agent_id) returning continuous waypoints for waypoint injection.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import multiprocessing as mp
import time


def _cbs_plan_worker(
    starts_list: List,
    goals_list: List,
    static_obstacles: List,
    plan_config: Dict,
    result_queue: mp.Queue,
) -> None:
    """Run in subprocess: call cbs_mapf Planner.plan() and put result in queue."""
    try:
        from cbs_mapf.planner import Planner
        max_iter = int(plan_config.get("CBS_MAX_ITER", 200))
        low_level_max_iter = int(plan_config.get("CBS_LOW_LEVEL_MAX_ITER", 100))
        robot_radius_cells = max(0, int(plan_config.get("CBS_ROBOT_RADIUS_CELLS", 0)))
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
            max_process=1,
            debug=False,
        )
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


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
        self._import_checked: bool = False

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
        self._last_debug_info = None

        if not deadlock_participants or not agent_states:
            return None

        cbs_debug = bool(self.config.get("CBS_DEBUG", False))
        try:
            grid, resolution, world_w, world_h = self.gym_env._build_occupancy_grid_for_long_range()
        except Exception as e:
            if cbs_debug:
                print(f"CBS: Failed to build occupancy grid: {e}")
            return None

        if grid is None or resolution is None or resolution <= 0:
            if cbs_debug:
                print("CBS: Invalid grid or resolution")
            return None

        self._resolution = float(resolution)
        self._offset_x = float(getattr(self.gym_env, "offset_x", 0.0))
        self._offset_y = float(getattr(self.gym_env, "offset_y", 0.0))
        self._grid_height = len(grid)
        self._grid_width = len(grid[0]) if self._grid_height else 0

        # cbs_mapf/space-time-astar use (x,y) = (col, row); grid extent is from obstacle bounds,
        # so add boundary points to cover full world so start/goal are inside the grid
        static_obstacles = []
        for row in range(self._grid_height):
            for col in range(self._grid_width):
                if grid[row][col] != 0:
                    static_obstacles.append((col, row))
        static_obstacles.append((-1, -1))
        static_obstacles.append((self._grid_width, -1))
        static_obstacles.append((-1, self._grid_height))
        static_obstacles.append((self._grid_width, self._grid_height))
        obstacle_set = set(static_obstacles)

        def is_free(col: int, row: int) -> bool:
            if not (0 <= col < self._grid_width and 0 <= row < self._grid_height):
                return False
            return (col, row) not in obstacle_set and grid[row][col] == 0

        def snap_to_free(col: int, row: int, max_radius: int = 4) -> Optional[Tuple[int, int]]:
            if is_free(col, row):
                return (col, row)
            for r in range(1, max_radius + 1):
                for dc in range(-r, r + 1):
                    for dr in range(-r, r + 1):
                        if abs(dc) != r and abs(dr) != r:
                            continue
                        c2, r2 = col + dc, row + dr
                        if is_free(c2, r2):
                            return (c2, r2)
            return None

        robot_radius_world = float(
            self.config.get("CBS_ROBOT_RADIUS", self.config.get("GRID_RESOLUTION", 0.5) * 0.5)
        )
        # Grid is already dilated by DEADLOCK_OBSTACLE_MARGIN_CELLS when built (shared by all
        # deadlock solvers). Use CBS_ROBOT_RADIUS_CELLS only for extra CBS-only margin (default 0).

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
            c_start = snap_to_free(c_start[0], c_start[1])
            c_goal = snap_to_free(c_goal[0], c_goal[1])
            if c_start is None or c_goal is None:
                if cbs_debug:
                    print(f"CBS: Agent {pid} start or goal in obstacle and no free cell nearby")
                continue
            # cbs_mapf/space-time-astar expect (x,y) = (col, row)
            starts_list.append(list(c_start))
            goals_list.append(c_goal)
            order_ids.append(pid)

        # Ensure distinct goal cells: CBS cannot have two agents with the same goal cell
        used_goal_cells = set()
        for i in range(len(goals_list)):
            g = tuple(goals_list[i])
            if g in used_goal_cells:
                # find a free cell adjacent to g not in used_goal_cells and not in starts
                start_set = set(tuple(s) for s in starts_list)
                best = None
                for radius in range(1, 6):
                    for dc in range(-radius, radius + 1):
                        for dr in range(-radius, radius + 1):
                            if abs(dc) != radius and abs(dr) != radius:
                                continue
                            c2, r2 = g[0] + dc, g[1] + dr
                            cand = (c2, r2)
                            if is_free(c2, r2) and cand not in used_goal_cells and cand not in start_set:
                                best = cand
                                break
                        if best is not None:
                            break
                    if best is not None:
                        break
                if best is None:
                    if cbs_debug:
                        print("CBS: Could not assign distinct goal for agent at duplicate goal")
                    return None
                goals_list[i] = list(best)
                used_goal_cells.add(best)
            else:
                used_goal_cells.add(g)

        if len(starts_list) != len(deadlock_participants) or len(starts_list) == 0:
            if cbs_debug:
                print("CBS: Missing start/goal for some participants")
            return None

        # Store debug info for verification (grid + inputs); saved by deadlock_logger when present
        self._last_debug_info = {
            "grid": [row[:] for row in grid],
            "resolution": self._resolution,
            "offset_x": self._offset_x,
            "offset_y": self._offset_y,
            "grid_height": self._grid_height,
            "grid_width": self._grid_width,
            "obstacle_count": len(static_obstacles),
            "starts_list": [list(s) for s in starts_list],
            "goals_list": [list(g) for g in goals_list],
            "order_ids": list(order_ids),
        }

        try:
            from cbs_mapf.planner import Planner
            if not self._import_checked:
                if cbs_debug:
                    print("CBS: cbs_mapf solver available")
                self._import_checked = True
        except ImportError as e:
            if not self._import_checked:
                if cbs_debug:
                    print(f"CBS: cbs_mapf not available (install with: pip install cbs-mapf): {e}")
                self._import_checked = True
            return None

        plan_config = {
            "CBS_MAX_ITER": int(self.config.get("CBS_MAX_ITER", 200)),
            "CBS_LOW_LEVEL_MAX_ITER": int(self.config.get("CBS_LOW_LEVEL_MAX_ITER", 100)),
            "CBS_ROBOT_RADIUS_CELLS": max(0, int(self.config.get("CBS_ROBOT_RADIUS_CELLS", 0))),
        }
        timeout_sec = float(self.config.get("CBS_TIMEOUT_SEC", 15.0))
        result_queue = mp.Queue()
        proc = mp.Process(
            target=_cbs_plan_worker,
            args=(starts_list, goals_list, static_obstacles, plan_config, result_queue),
        )
        proc.start()
        try:
            msg = result_queue.get(timeout=timeout_sec)
        except Exception:
            proc.terminate()
            proc.join(timeout=2.0)
            try:
                proc.kill()
            except Exception:
                pass
            if cbs_debug:
                print("CBS: Timeout exceeded (CBS_TIMEOUT_SEC={})".format(timeout_sec))
            return None
        if msg[0] == "error":
            if cbs_debug:
                print("CBS: Plan failed:", msg[1])
            return None
        result = msg[1]

        if result is None or (isinstance(result, np.ndarray) and result.size == 0):
            if cbs_debug:
                print("CBS: No solution found")
            return None

        result = np.asarray(result)
        if result.ndim != 3 or result.shape[0] != len(order_ids):
            if cbs_debug:
                print("CBS: Unexpected result shape")
            return None

        # cbs_mapf returns path as (x,y) = (col, row) per timestep
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
