class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def euclidean_norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

class Node:
    def __init__(self, i=0, j=0, traversable=True):
        self.i = i  # row
        self.j = j  # col
        self.traversable = traversable

    def __eq__(self, other):
        if other is None:
            return False
        return self.i == other.i and self.j == other.j

    def __lt__(self, other):
        return (self.i, self.j) < (other.i, other.j)

    def __hash__(self):
        return hash((self.i, self.j)) 

class ActorMove:
    """与C++ ActorMove结构体对应，存储增量移动"""
    def __init__(self, di, dj, id):
        self.di = di  # 行方向增量 (-1, 0, 1)
        self.dj = dj  # 列方向增量 (-1, 0, 1)
        self.id = id  # agent ID
    
    def __str__(self):
        return f"ActorMove(di={self.di}, dj={self.dj}, id={self.id})"
    
    def __repr__(self):
        return self.__str__() 