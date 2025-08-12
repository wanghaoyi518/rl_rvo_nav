from __future__ import annotations

from typing import List, Tuple
import numpy as np


class SimpleEnvAdapter:
    """
    最小环境适配器：基于 ir_sim 的 env_base，提供 MAPF 所需接口。
    - 如存在 components['map_matrix']，用其构建占据；否则使用全自由栅格（无障碍）。
    - 坐标转换按 env.xy_reso 与 offset 实现。
    """

    def __init__(self, ir_env, override_resolution: float = None) -> None:
        # 允许传入 gym 包装器或底层 ir_gym
        self.env_wrapper = ir_env
        self.core = getattr(ir_env, 'ir_gym', ir_env)  # ir_gym 实例（继承 env_base）
        self._override_resolution = float(override_resolution) if override_resolution is not None else None

    def get_static_occupancy(self) -> List[List[int]]:
        mat = self.core.components.get('map_matrix', None)
        if mat is None:
            # 全自由栅格：按世界尺寸和分辨率生成
            res = self.get_resolution()
            width = int(round(self.core._env_base__width / res)) if hasattr(self.core, '_env_base__width') else 100
            height = int(round(self.core._env_base__height / res)) if hasattr(self.core, '_env_base__height') else 100
            grid = [[0 for _ in range(width)] for _ in range(height)]
            return grid
        # mat 中 255 为障碍，0 为可行（见 env_base.init_environment 处理逻辑后翻转）；此处将非零视作障碍
        occ = (mat > 0).astype(np.uint8)
        return occ.tolist()

    def get_resolution(self) -> float:
        return float(self._override_resolution) if self._override_resolution is not None else float(self.core.xy_reso)

    def get_origin(self) -> Tuple[float, float]:
        # 对应网格 (i=0,j=0) 的世界坐标。ir_sim 使用 offset 作为世界坐标偏移。
        off = self.core.components.get('offset', np.array([0.0, 0.0], dtype=float))
        return float(off[0]), float(off[1])

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        x0, y0 = self.get_origin()
        res = self.get_resolution()
        # 假设 i 向下、j 向右；ir_sim 的 map_matrix 经过转置与左右翻转，这里统一用“向下/向右为正”，按需求可再校正
        i = int(np.floor((y - y0) / res))
        j = int(np.floor((x - x0) / res))
        return i, j

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x0, y0 = self.get_origin()
        res = self.get_resolution()
        x = x0 + (j + 0.5) * res
        y = y0 + (i + 0.5) * res
        return x, y

    def crop_subgrid(self, bbox_world, pad_cells: int = 1):
        (min_x, min_y), (max_x, max_y) = bbox_world
        # 转为网格范围
        imin, jmin = self.world_to_grid(min_x, min_y)
        imax, jmax = self.world_to_grid(max_x, max_y)
        if imin > imax:
            imin, imax = imax, imin
        if jmin > jmax:
            jmin, jmax = jmax, jmin
        imin = max(0, imin - pad_cells)
        jmin = max(0, jmin - pad_cells)

        occ = self.get_static_occupancy()
        H, W = len(occ), len(occ[0]) if occ else 0
        imax = min(H - 1, imax + pad_cells)
        jmax = min(W - 1, jmax + pad_cells)

        sub = [row[jmin:jmax + 1] for row in occ[imin:imax + 1]]
        origin_ij = (imin, jmin)
        res = self.get_resolution()
        return sub, origin_ij, res


