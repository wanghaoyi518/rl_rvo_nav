from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


class SubMapBuilder:
    """
    基于 env_adapter 提供的全局占据栅格与坐标转换，构建 bbox 与子图。
    env_adapter 需实现：
      - get_static_occupancy() -> List[List[int]]
      - get_resolution() -> float
      - get_origin() -> Tuple[float, float]
      - world_to_grid(x, y) -> (i, j)
      - grid_to_world(i, j) -> (x, y)
      - crop_subgrid(bbox_world, pad_cells=1) -> (grid, (origin_i, origin_j), resolution)
    """

    def __init__(self, env_adapter) -> None:
        self.env = env_adapter

    def compute_bbox(self, positions_world: List[np.ndarray], margin_m: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs = [float(p[0]) for p in positions_world]
        ys = [float(p[1]) for p in positions_world]
        min_x, max_x = min(xs) - margin_m, max(xs) + margin_m
        min_y, max_y = min(ys) - margin_m, max(ys) + margin_m
        return (min_x, min_y), (max_x, max_y)

    def compute_bbox_with_integer_dimensions(self, positions_world: List[np.ndarray], margin_m: float = 0.0, resolution: float = 1.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        计算整数尺寸的BBox，确保宽度和高度是分辨率的整数倍
        
        Args:
            positions_world: 智能体位置列表
            margin_m: 边距（米）
            resolution: 网格分辨率（米/格子）
            
        Returns:
            ((min_x, min_y), (max_x, max_y)): 整数尺寸的BBox
        """
        xs = [float(p[0]) for p in positions_world]
        ys = [float(p[1]) for p in positions_world]
        
        # 计算基础BBox
        min_x, max_x = min(xs) - margin_m, max(xs) + margin_m
        min_y, max_y = min(ys) - margin_m, max(ys) + margin_m
        
        # 计算当前尺寸
        current_width = max_x - min_x
        current_height = max_y - min_y
        
        # 计算目标整数格子数（向上取整，确保覆盖所有智能体）
        target_width_cells = int(np.ceil(current_width / resolution))
        target_height_cells = int(np.ceil(current_height / resolution))
        
        # 计算整数尺寸
        target_width = target_width_cells * resolution
        target_height = target_height_cells * resolution
        
        # 计算中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 从中心点扩展，确保整数尺寸
        half_width = target_width / 2
        half_height = target_height / 2
        
        # 确保最小尺寸（至少3x3格子）
        min_cells = 3
        if target_width_cells < min_cells:
            half_width = (min_cells * resolution) / 2
        if target_height_cells < min_cells:
            half_height = (min_cells * resolution) / 2
        
        # 计算最终的BBox
        final_min_x = center_x - half_width
        final_max_x = center_x + half_width
        final_min_y = center_y - half_height
        final_max_y = center_y + half_height
        
        return (final_min_x, final_min_y), (final_max_x, final_max_y)

    def build_submap(self, bbox_world: Tuple[Tuple[float, float], Tuple[float, float]], pad_cells: int = 1):
        grid, origin_ij, resolution = self.env.crop_subgrid(bbox_world, pad_cells)
        return grid, origin_ij, resolution

    # 坐标转换辅助（转调 env_adapter）
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return self.env.world_to_grid(x, y)

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return self.env.grid_to_world(i, j)


