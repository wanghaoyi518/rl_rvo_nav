from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from .node import Point


class MAPFExecutor:
    """
    将离散 Point(i,j) 路径转为连续世界坐标的目标点序列，供外层控制器跟踪。
    简化实现：靠近当前目标格子一定阈值则推进到下一个。
    """

    def __init__(self, builder) -> None:
        self.builder = builder
        self.paths_grid: Dict[int, List[Point]] = {}
        self.path_index: Dict[int, int] = {}
        self.finished: Dict[int, bool] = {}
        self._next_targets_world: Dict[int, np.ndarray] = {}
        self.arrival_threshold: float = 0.2  # 米

    def prepare(self, paths_grid: Dict[int, List[Point]]) -> None:
        self.paths_grid = {aid: list(points) for aid, points in paths_grid.items()}
        self.path_index = {aid: 0 for aid in self.paths_grid.keys()}
        self.finished = {aid: False for aid in self.paths_grid.keys()}
        self._recompute_targets()

    def step(self, current_positions: Dict[int, np.ndarray], dt: float) -> None:
        # 推进所有未完成 agent 的路径索引
        for aid, cur_pos in current_positions.items():
            if aid not in self.paths_grid or self.finished.get(aid, False):
                continue
            target = self._next_targets_world.get(aid)
            if target is None:
                continue
            if np.linalg.norm(cur_pos[:2] - target[:2]) <= self.arrival_threshold:
                # 到达该离散格子，推进到下一个
                self.path_index[aid] = min(self.path_index[aid] + 1, len(self.paths_grid[aid]) - 1)
        self._recompute_targets()

    def next_target(self, agent_id: int) -> Optional[np.ndarray]:
        return self._next_targets_world.get(agent_id)

    def is_finished(self) -> bool:
        # 当所有 agent 都到达各自路径末尾时完成
        for aid, path in self.paths_grid.items():
            if not path:
                continue
            if self.path_index.get(aid, 0) < len(path) - 1:
                return False
        return True

    # ===== 内部 =====
    def _recompute_targets(self) -> None:
        self._next_targets_world.clear()
        for aid, path in self.paths_grid.items():
            if not path:
                continue
            idx = self.path_index.get(aid, 0)
            idx = max(0, min(idx, len(path) - 1))
            node = path[idx]
            x, y = self.builder.grid_to_world(int(node.x), int(node.y))
            self._next_targets_world[aid] = np.array([x, y], dtype=float)


