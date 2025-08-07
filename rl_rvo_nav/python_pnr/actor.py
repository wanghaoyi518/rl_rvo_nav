from .node import Point

class Actor:
    def __init__(self, id, start: Point, goal: Point):
        self.id = id
        self.start = Point(start.x, start.y)
        self.init_start = Point(start.x, start.y)  # 不可变原始起点
        self.goal = Point(goal.x, goal.y)
        self.current = Point(start.x, start.y)
        self.path = []
        self.finished = False
        self.blocked = False
        self.moves = []  # 记录每一步的节点

    def reset(self):
        self.current = self.start
        self.path = []
        self.finished = False
        self.blocked = False
        self.moves = []

    def at_goal(self):
        return self.current == self.goal 