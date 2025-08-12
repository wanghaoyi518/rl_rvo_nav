from .node import Node


class SubMap:
    def __init__(self, grid, origin_i=0, origin_j=0):
        self.grid = grid  # 2D list of 0/1 (0: free, 1: obstacle)
        self.origin_i = origin_i  # 子图左上角在全局网格中的 i 偏移
        self.origin_j = origin_j  # 子图左上角在全局网格中的 j 偏移
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def _to_local(self, i_global, j_global):
        li = i_global - self.origin_i
        lj = j_global - self.origin_j
        return li, lj

    def in_bounds(self, i_global, j_global):
        li, lj = self._to_local(i_global, j_global)
        return 0 <= li < self.height and 0 <= lj < self.width

    def is_traversable(self, i_global, j_global):
        li, lj = self._to_local(i_global, j_global)
        return (0 <= li < self.height and 0 <= lj < self.width and self.grid[li][lj] == 0)

    def get_node(self, i_local, j_local):
        # 兼容旧接口：传入局部索引，返回全局 Node
        gi = i_local + self.origin_i
        gj = j_local + self.origin_j
        return Node(gi, gj, self.is_traversable(gi, gj))