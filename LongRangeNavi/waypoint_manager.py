from typing import List, Tuple, Optional
import math


class WaypointManager:
    """Manage per-agent waypoint progression for long-range navigation."""

    def __init__(self, agent_id: int, waypoint_list: List[Tuple[float, float]], reach_threshold: float = 1.0):
        self.agent_id = int(agent_id)
        self._waypoints: List[Tuple[float, float]] = list(waypoint_list or [])
        self._reach_threshold: float = float(reach_threshold)
        self._index: int = 0

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
            if self._index >= len(self._waypoints):
                return True, True
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


