from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np

from .node import Node, Point


class StartGoalSelector:
    """
    起点/终点选择：
    - 起点：从连续位置选最近可用离散节点（避免与其他 agent 冲突）
    - 终点：从 waypoints 中选出离开 bbox 的第一个连续目标，再映射到可达离散节点
    """

    def __init__(self, builder) -> None:
        self.builder = builder

    def find_close_available_node(self, sub_map, pos_world: np.ndarray, occupied: set) -> Optional[Node]:
        i, j = self._round_to_grid(sub_map, pos_world)
        # 若占用或不可行，按曼哈顿圈层扩张寻找
        if sub_map.is_traversable(i, j) and (i, j) not in occupied:
            return Node(i, j)
        max_r = max(sub_map.height, sub_map.width)
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                dj1 = r - abs(di)
                for dj in (-dj1, dj1):
                    ni, nj = i + di, j + dj
                    if sub_map.is_traversable(ni, nj) and (ni, nj) not in occupied:
                        return Node(ni, nj)
        return None

    def get_goal_point_for_mapf(
        self,
        waypoints: List[np.ndarray],
        bbox_world: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        skipped: List[np.ndarray] = []
        for wp in waypoints:
            if not self._inside_bbox(wp, bbox_world):
                return wp, skipped
            skipped.append(wp)
        # 若全部在 bbox 内，使用最后一个（全局目标）
        if waypoints:
            return waypoints[-1], skipped
        # 无 waypoint，回退：使用 bbox 右下角附近一点
        (min_x, min_y), (max_x, max_y) = bbox_world
        return np.array([max_x, max_y], dtype=float), skipped

    def find_accessible_node_for_goal(self, sub_map, target_world: np.ndarray, occupied: set) -> Optional[Node]:
        i, j = self._round_to_grid(sub_map, target_world)
        if sub_map.is_traversable(i, j) and (i, j) not in occupied:
            return Node(i, j)
        # 逐圈扩张寻找可达节点
        max_r = max(sub_map.height, sub_map.width)
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                dj1 = r - abs(di)
                for dj in (-dj1, dj1):
                    ni, nj = i + di, j + dj
                    if sub_map.is_traversable(ni, nj) and (ni, nj) not in occupied:
                        return Node(ni, nj)
        return None

    # ===== 辅助 =====
    def _round_to_grid(self, sub_map, pos_world: np.ndarray) -> Tuple[int, int]:
        x, y = float(pos_world[0]), float(pos_world[1])
        gi, gj = self.builder.world_to_grid(x, y)
        return int(gi), int(gj)

    def _inside_bbox(self, wp: np.ndarray, bbox_world: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        (min_x, min_y), (max_x, max_y) = bbox_world
        x, y = float(wp[0]), float(wp[1])
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)


