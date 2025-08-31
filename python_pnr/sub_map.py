from .node import Node

class SubMap:
    def __init__(self, grid, origin_i=0, origin_j=0):
        self.grid = grid  # 2D list of 0/1 (0: free, 1: obstacle)
        self.origin_i = origin_i
        self.origin_j = origin_j
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def in_bounds(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

    def is_traversable(self, i, j):
        return self.in_bounds(i, j) and self.grid[i][j] == 0

    def get_node(self, i, j):
        return Node(i + self.origin_i, j + self.origin_j, self.is_traversable(i, j)) 