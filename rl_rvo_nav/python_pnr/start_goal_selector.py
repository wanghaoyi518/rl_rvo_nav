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
        使用A*算法寻找距离目标点最近的可达节点
        参考C++版本的FindAccessibleNodeForGoal方法实现
        
        Args:
            sub_map: 子地图
            target_world: 目标点的世界坐标
            occupied: 被占用的节点集合
            max_search_radius: 最大搜索半径
            
        Returns:
            Optional[Node]: 找到的最佳节点，如果没找到则返回None
        """
        # 将目标点转换为网格坐标
        target_i, target_j = self._round_to_grid(sub_map, target_world)
        
        # 限制在子地图范围内
        target_i = max(0, min(target_i, sub_map.height - 1))
        target_j = max(0, min(target_j, sub_map.width - 1))
        
        # 使用目标点作为起始搜索点
        start_node = Node(target_i, target_j)
        
        # 检查起始点是否可用
        if (sub_map.is_traversable(start_node.i, start_node.j) and 
            (start_node.i, start_node.j) not in occupied):
            return start_node
        
        # 如果起始点不可用，寻找替代起始点
        start_node = self._find_unoccupied_node(start_node, occupied, sub_map)
        if start_node.i < 0:
            return None
        
        # A*算法数据结构
        open_list = []  # 待探索节点列表
        closed_set = set()  # 已探索节点集合
        
        # 初始化起始节点
        start_node.g = 0
        start_node.H = self._euclidean_distance(start_node.i, start_node.j, target_i, target_j)
        start_node.F = start_node.g + start_node.H
        start_node.parent = None
        
        # 将起始节点加入开放列表
        open_list.append(start_node)
        best_node = start_node
        
        # A*搜索主循环
        while open_list:
            # 选择F值最小的节点
            curr_node = self._pop_min_f_node(open_list)
            closed_set.add((curr_node.i, curr_node.j))
            
            # 更新最佳候选节点
            if (curr_node.F < best_node.F or 
                (abs(curr_node.F - best_node.F) < 1e-6 and curr_node.H < best_node.H)):
                best_node = curr_node
            
            # 检查是否到达目标
            if curr_node.H < 1e-6:
                # 清理节点信息
                curr_node.parent = None
                curr_node.F = 0.0
                curr_node.H = 0.0
                curr_node.g = 0.0
                return curr_node
            
            # 扩展邻居节点（上下左右四个方向）
            neighbors = [
                Node(curr_node.i + 1, curr_node.j),
                Node(curr_node.i - 1, curr_node.j),
                Node(curr_node.i, curr_node.j + 1),
                Node(curr_node.i, curr_node.j - 1)
            ]
            
            for neighbor in neighbors:
                # 检查邻居节点是否有效
                if (self._is_valid_node(neighbor, sub_map) and
                    (neighbor.i, neighbor.j) not in occupied and
                    (neighbor.i, neighbor.j) not in closed_set):
                    
                    # 计算邻居节点的代价
                    neighbor.g = curr_node.g + 1  # 假设每步代价为1
                    neighbor.H = self._euclidean_distance(neighbor.i, neighbor.j, target_i, target_j)
                    neighbor.F = neighbor.g + neighbor.H
                    neighbor.parent = curr_node
                    
                    # 检查是否已在开放列表中
                    existing_node = self._find_node_in_list(open_list, neighbor)
                    if existing_node:
                        # 如果新路径更短，更新节点
                        if neighbor.g < existing_node.g:
                            existing_node.g = neighbor.g
                            existing_node.H = neighbor.H
                            existing_node.F = neighbor.F
                            existing_node.parent = neighbor.parent
                    else:
                        # 添加新节点到开放列表
                        open_list.append(neighbor)
        
        # 如果没有找到理想目标，返回最佳候选
        best_node.parent = None
        best_node.F = 0.0
        best_node.H = 0.0
        best_node.g = 0.0
        return best_node
    
    def _find_unoccupied_node(self, start_node: Node, occupied: set, sub_map) -> Node:
        """
        寻找未占用的替代起始节点
        """
        # 首先检查起始点本身
        if (sub_map.is_traversable(start_node.i, start_node.j) and 
            (start_node.i, start_node.j) not in occupied):
            return start_node
        
        # 按曼哈顿距离扩展搜索
        max_r = max(sub_map.height, sub_map.width)
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                dj1 = r - abs(di)
                for dj in (-dj1, dj1):
                    ni, nj = start_node.i + di, start_node.j + dj
                    if (0 <= ni < sub_map.height and 0 <= nj < sub_map.width and
                        sub_map.is_traversable(ni, nj) and (ni, nj) not in occupied):
                        return Node(ni, nj)
        
        # 如果没找到，返回无效节点
        return Node(-1, -1)
    
    def _euclidean_distance(self, i1: int, j1: int, i2: int, j2: int) -> float:
        """
        计算两点间的欧几里得距离
        """
        return np.sqrt((i1 - i2)**2 + (j1 - j2)**2)
    
    def _pop_min_f_node(self, open_list: list) -> Node:
        """
        从开放列表中取出F值最小的节点
        """
        min_idx = 0
        min_f = open_list[0].F
        
        for i, node in enumerate(open_list):
            if node.F < min_f:
                min_f = node.F
                min_idx = i
        
        return open_list.pop(min_idx)
    
    def _is_valid_node(self, node: Node, sub_map) -> bool:
        """
        检查节点是否有效（在地图范围内且可通行）
        """
        return (0 <= node.i < sub_map.height and 
                0 <= node.j < sub_map.width and 
                sub_map.is_traversable(node.i, node.j))
    
    def _find_node_in_list(self, node_list: list, target_node: Node) -> Optional[Node]:
        """
        在节点列表中查找指定节点
        """
        for node in node_list:
            if node.i == target_node.i and node.j == target_node.j:
                return node
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


