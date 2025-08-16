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
        current_position: np.ndarray = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        改进的目标点选择策略：
        1. 优先选择距离当前位置最近的waypoint
        2. 确保目标点在bbox内
        3. 避免选择过远的目标点
        """
        if not waypoints:
            # 无 waypoint，回退：使用 bbox 右下角附近一点
            (min_x, min_y), (max_x, max_y) = bbox_world
            skipped: List[np.ndarray] = []
            return np.array([max_x, max_y], dtype=float), skipped
        
        # 如果有当前位置信息，优先选择最近的waypoint
        if current_position is not None:
            # 计算到每个waypoint的距离
            distances = []
            for wp in waypoints:
                dist = np.linalg.norm(current_position - wp)
                distances.append(dist)
            
            # 按距离排序，优先选择最近的
            sorted_indices = np.argsort(distances)
            
            # 从最近的开始，找到第一个在bbox内的waypoint
            for idx in sorted_indices:
                wp = waypoints[idx]
                if self._inside_bbox(wp, bbox_world):
                    # 检查距离是否合理（避免选择过远的目标）
                    max_reasonable_distance = 10.0  # 最大合理距离
                    if distances[idx] <= max_reasonable_distance:
                        skipped = waypoints[:idx]
                        return wp, skipped
            
            # 如果所有waypoint都不在bbox内或距离过远，选择最近的
            best_idx = sorted_indices[0]
            skipped = waypoints[:best_idx]
            return waypoints[best_idx], skipped
        
        # 原有的选择策略（作为fallback）
        inside_indices = [idx for idx, wp in enumerate(waypoints) if self._inside_bbox(wp, bbox_world)]
        if inside_indices:
            idx = max(inside_indices)  # 使用序列中靠后的一个
            skipped: List[np.ndarray] = waypoints[:idx]
            return waypoints[idx], skipped
        else:
            # 若全部在 bbox 外，回退使用全局终点（最后一个）
            skipped = waypoints[:-1]
            return waypoints[-1], skipped

    def find_accessible_node_for_goal(self, sub_map, target_world: np.ndarray, occupied: set, max_search_radius: int = 5) -> Optional[Node]:
        """
        改进的可达节点搜索：
        1. 限制搜索半径，避免选择过远的节点
        2. 优先选择距离目标点最近的节点
        3. 添加距离检查
        """
        i, j = self._round_to_grid(sub_map, target_world)
        
        # clamp to submap bounds if target falls outside current bbox
        i = max(0, min(int(i), sub_map.height - 1))
        j = max(0, min(int(j), sub_map.width - 1))
        
        # 首先检查目标点本身
        if sub_map.is_traversable(i, j) and (i, j) not in occupied:
            return Node(i, j)
        
        # 限制搜索半径，避免选择过远的节点
        search_radius = min(max_search_radius, max(sub_map.height, sub_map.width))
        
        # 按距离排序的候选节点
        candidates = []
        
        for r in range(1, search_radius + 1):
            for di in range(-r, r + 1):
                dj1 = r - abs(di)
                for dj in (-dj1, dj1):
                    ni, nj = i + di, j + dj
                    if (0 <= ni < sub_map.height and 0 <= nj < sub_map.width and
                        sub_map.is_traversable(ni, nj) and (ni, nj) not in occupied):
                        # 计算到目标点的距离
                        dist = np.sqrt((ni - i)**2 + (nj - j)**2)
                        candidates.append((dist, ni, nj))
        
        # 按距离排序，返回最近的
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, best_i, best_j = candidates[0]
            return Node(best_i, best_j)
        
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


