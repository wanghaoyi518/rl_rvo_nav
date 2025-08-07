import math

# 常量定义
CN_EPS = 0.00001
CN_INFINITY = 1000000000
CN_PI_CONSTANT = math.pi
CN_SQRT_TWO = math.sqrt(2)

COMMON_SPEED_BUFF_SIZE = 1000
SPEED_BUFF_SIZE = 250
SMALL_SPEED = 0.1
MISSION_SMALL_SPEED = 0.001
ECBS_SUBOUT_FACTOR = 10.0

# 基础几何工具

def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def sq_euclidean_distance(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2

def less_point(a, b):
    return (a.x, a.y) < (b.x, b.y) 