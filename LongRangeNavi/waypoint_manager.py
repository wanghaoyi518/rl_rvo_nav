from typing import List, Tuple, Optional
import math


class WaypointManager:
    """Manage per-agent waypoint progression for long-range navigation."""

    def __init__(self, agent_id: int, waypoint_list: List[Tuple[float, float]], reach_threshold: float = 1.0,
                 force_switch_enabled: bool = False,
                 force_switch_steps: int = 0):
        self.agent_id = int(agent_id)
        self._waypoints: List[Tuple[float, float]] = list(waypoint_list or [])
        self._reach_threshold: float = float(reach_threshold)
        self._index: int = 0
        # Optional: force advance to the next waypoint after N steps without reaching (non-final only)
        self._force_switch_enabled: bool = bool(force_switch_enabled)
        self._force_switch_steps: int = int(force_switch_steps)
        self._stay_steps: int = 0

    def get_current_goal(self) -> Optional[Tuple[float, float]]:
        if 0 <= self._index < len(self._waypoints):
            return self._waypoints[self._index]
        return None

    def update(self, agent_position: Tuple[float, float]):
        """Update waypoint progress based on current agent position.

        Returns:
            (goal_reached, final_goal_reached)
        """
        goal = self.get_current_goal()
        if goal is None:
            return False, True

        dx = float(agent_position[0]) - float(goal[0])
        dy = float(agent_position[1]) - float(goal[1])
        dist = math.hypot(dx, dy)

        if dist <= self._reach_threshold:
            self._index += 1
            self._stay_steps = 0
            if self._index >= len(self._waypoints):
                return True, True
            return True, False

        # Force-switch: if enabled and stuck on the same waypoint beyond threshold, advance index (but not beyond last)
        if self._force_switch_enabled and self._force_switch_steps > 0:
            self._stay_steps += 1
            # Only force-advance if there exists a next waypoint (do not skip final goal)
            if self._stay_steps >= self._force_switch_steps and self._index < max(0, len(self._waypoints) - 1):
                self._index += 1
                self._stay_steps = 0
                # Not final completion because we never skip past the last waypoint here
                return True, False

        return False, False

    def get_progress_info(self):
        remaining = max(0, len(self._waypoints) - self._index)
        return {
            "agent_id": self.agent_id,
            "current_index": self._index,
            "remaining": remaining,
            "total": len(self._waypoints),
        }


