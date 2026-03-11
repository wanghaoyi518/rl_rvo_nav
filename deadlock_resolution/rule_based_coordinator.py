"""
Rule-based Sequential Coordinator Module

Provides a rule-based deadlock solver: participants ordered left-to-right, bottom-to-top;
only one agent moves at a time; when the current-priority agent reaches the next agent's
position, the next agent starts. Interface aligned with PAR/CBS: prepare_rule_based_execution
and get_agent_path(agent_id) returning continuous waypoints.
"""

from typing import Dict, List, Tuple, Optional


def _get_position_from_state(agent_state: Dict) -> Optional[Tuple[float, float]]:
    """Extract (x, y) position from agent state dict. Same convention as CBSCoordinator."""
    if "position" in agent_state:
        p = agent_state["position"]
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return (float(p[0]), float(p[1]))
        try:
            import numpy as np
            if isinstance(p, np.ndarray) and p.size >= 2:
                return (float(p.flat[0]), float(p.flat[1]))
        except Exception:
            pass
    for key in ("pos", "location", "pose"):
        if key in agent_state:
            p = agent_state[key]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return (float(p[0]), float(p[1]))
    return None


def _get_goal_from_state(agent_state: Dict) -> Optional[Tuple[float, float]]:
    """Extract (x, y) goal from agent state dict. Same convention as CBSCoordinator."""
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
    if isinstance(goal, (list, tuple)) and len(goal) >= 2:
        return (float(goal[0]), float(goal[1]))
    try:
        import numpy as np
        if isinstance(goal, np.ndarray) and goal.size >= 2:
            return (float(goal.flat[0]), float(goal.flat[1]))
    except Exception:
        pass
    return None


def _interpolate_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    step: float,
) -> List[Tuple[float, float]]:
    """Linear interpolation from start to end with given step size (world units)."""
    if step <= 0:
        return [start, end] if start != end else [start]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= 1e-9:
        return [start]
    n = max(1, int(round(dist / step)))
    out = []
    for i in range(n + 1):
        t = i / n
        x = start[0] + t * dx
        y = start[1] + t * dy
        out.append((x, y))
    return out


class RuleBasedSequentialCoordinator:
    """
    Rule-based sequential deadlock solver. Participants ordered by (x, y) ascending
    (left-to-right, bottom-to-top). Only one agent moves at a time; when current
    reaches next agent's position, next agent starts. Paths are continuous; wait
    is encoded as repeated waypoints so env can inject all at once.
    """

    def __init__(self, config: Dict, gym_env=None):
        """
        Args:
            config: Dict with GRID_RESOLUTION, RULE_BASED_WAIT_WAYPOINTS (optional),
                    RULE_BASED_INTERP_STEP (optional), DEBUG_MODE (optional).
            gym_env: Optional reference to gym env (for future obstacle-aware extensions).
        """
        self.config = config if isinstance(config, dict) else {}
        self.gym_env = gym_env
        self.current_solution: Dict[int, List[Tuple[float, float]]] = {}
        self.current_participants: List[int] = []
        self._last_order: List[int] = []
        self._last_debug_info: Optional[Dict] = None

    def prepare_rule_based_execution(
        self, agent_states: Dict, deadlock_participants: List[int]
    ) -> Optional["RuleBasedSequentialCoordinator"]:
        """
        Build sequential paths for participants. Same inputs as PAR/CBS.
        Returns self on success; paths stored for get_agent_path(agent_id).
        """
        self.current_solution = {}
        self.current_participants = list(deadlock_participants)

        if not deadlock_participants or not agent_states:
            return None

        positions = {}
        goals = {}
        for pid in deadlock_participants:
            if pid not in agent_states:
                return None
            state = agent_states[pid]
            pos = _get_position_from_state(state)
            if pos is None:
                return None
            positions[pid] = pos
            g = _get_goal_from_state(state)
            if g is None and len(deadlock_participants) > 1:
                g = (pos[0] + 0.5, pos[1])
            goals[pid] = g

        order = sorted(
            deadlock_participants,
            key=lambda aid: (positions[aid][0], positions[aid][1]),
        )

        step = float(self.config.get("RULE_BASED_INTERP_STEP", 0))
        if step <= 0:
            step = float(self.config.get("GRID_RESOLUTION", 0.5))
        wait_override = int(self.config.get("RULE_BASED_WAIT_WAYPOINTS", 0))
        debug = bool(self.config.get("DEBUG_MODE", False))

        n = len(order)
        if n == 0:
            return None

        if n == 1:
            p0 = order[0]
            goal = goals.get(p0)
            if goal is None:
                goal = (positions[p0][0] + 0.5, positions[p0][1])
            seg = _interpolate_segment(positions[p0], goal, step)
            self.current_solution[p0] = seg
            self._last_order = [p0]
            self._last_debug_info = {
                "order": [p0],
                "starts": dict(positions),
                "goals": dict(goals),
                "path_lengths": {p0: len(seg)},
            }
            if debug:
                print(f"RuleBased: single agent {p0} path length {len(seg)}")
            return self

        path_lengths = []
        for i in range(n):
            start = positions[order[i]]
            if i < n - 1:
                end = positions[order[i + 1]]
            else:
                goal = goals.get(order[i])
                if goal is None:
                    goal = (start[0] + 0.5, start[1])
                end = goal
            seg = _interpolate_segment(start, end, step)
            path_lengths.append((order[i], len(seg), seg))

        prev_len = path_lengths[0][1]
        self.current_solution[path_lengths[0][0]] = path_lengths[0][2]

        for idx in range(1, n):
            pid, seg_len, seg = path_lengths[idx]
            N_wait = wait_override if wait_override > 0 else prev_len
            start_pos = positions[pid]
            wait_list = [start_pos] * N_wait
            path = wait_list + seg
            self.current_solution[pid] = path
            prev_len = len(path)
            if debug:
                print(f"RuleBased: agent {pid} wait={N_wait} segment={seg_len} total={len(path)}")

        self._last_order = list(order)
        self._last_debug_info = {
            "order": list(order),
            "starts": dict(positions),
            "goals": dict(goals),
            "path_lengths": {pid: len(self.current_solution[pid]) for pid in order},
        }
        return self

    def get_waypoint_tuples(
        self,
    ) -> Tuple[List[int], List[Tuple[Tuple[float, float], ...]]]:
        """
        Build per-timestep waypoint tuples from current_solution; pad shorter paths
        with last waypoint so all agents have the same number of steps.
        Returns (participant_ids, waypoint_tuples). Empty if no solution.
        """
        if not self.current_solution:
            return ([], [])
        participant_ids = getattr(self, "_last_order", None) or list(self.current_solution.keys())
        if not participant_ids:
            return ([], [])
        max_len = max(len(self.current_solution[pid]) for pid in participant_ids)
        if max_len == 0:
            return (participant_ids, [])
        waypoint_tuples = []
        for t in range(max_len):
            row = []
            for pid in participant_ids:
                path = self.current_solution[pid]
                if t < len(path):
                    pt = path[t]
                else:
                    pt = path[-1]
                row.append((float(pt[0]), float(pt[1])))
            waypoint_tuples.append(tuple(row))
        return (participant_ids, waypoint_tuples)

    def get_agent_path(self, agent_id: int) -> List[Tuple[float, float]]:
        """Return continuous waypoints for the agent, or empty list."""
        return self.current_solution.get(agent_id, [])

    def reset(self) -> None:
        """Clear current solution and participants."""
        self.current_solution = {}
        self.current_participants = []
        self._last_order = []
        self._last_debug_info = None
