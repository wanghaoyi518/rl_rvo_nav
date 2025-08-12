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

    def build_submap(self, bbox_world: Tuple[Tuple[float, float], Tuple[float, float]], pad_cells: int = 1):
        grid, origin_ij, resolution = self.env.crop_subgrid(bbox_world, pad_cells)
        return grid, origin_ij, resolution

    # 坐标转换辅助（转调 env_adapter）
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return self.env.world_to_grid(x, y)

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return self.env.grid_to_world(i, j)


