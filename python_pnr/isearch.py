from .node import Node, Point
import heapq
import time

class SearchResult:
    """对应C++版本的SearchResult结构"""
    def __init__(self):
        self.pathfound = False
        self.pathlength = 0.0
        self.lppath = []  # 路径作为相邻节点的序列
        self.hppath = []  # 路径作为非相邻节点的序列
        self.nodescreated = 0
        self.nodesexpanded = 0
        self.numberofsteps = 0
        self.time = 0.0
        self.lastNode = None
        self.minF = 0.0

class ISearch:
    def __init__(self, sub_map=None, with_time=False):
        self.sub_map = sub_map
        self.with_time = with_time
        self.hweight = 1.0
        self.breakingties = True
        self.sresult = SearchResult()
        self.lppath = []
        self.hppath = []

    def startSearch(self, map_obj, agent_set, start_i, start_j, goal_i=0, goal_j=0, 
                   is_goal=None, fresh_start=True, return_path=True, start_time=0, 
                   goal_time=-1, max_time=-1, occupied_nodes=None, constraints=None, 
                   with_cat=False, cat=None):
        """对应C++版本的startSearch方法"""
        if occupied_nodes is None:
            occupied_nodes = set()
        if constraints is None:
            constraints = set()
        
        # Store map_obj for boundary checking in path construction
        self.map_obj = map_obj
        
        self.sresult.pathfound = False
        begin_time = time.time()
        
        if goal_time != -1:
            max_time = goal_time
        
        # 检查起始位置是否被占用
        agent_id = -1
        if self.is_occupied(agent_set, start_i, start_j):
            agent_id = self.get_actor_id(agent_set, start_i, start_j)
        
        if fresh_start:
            self.clear_lists()
            self.sresult.numberofsteps = 0
        
        # 创建起始节点
        start_node = Node(start_i, start_j)
        start_node.g = 0
        start_node.h = self.compute_h_from_cell_to_cell(start_i, start_j, goal_i, goal_j)
        start_node.f = start_node.g + self.hweight * start_node.h
        
        # 添加到开放列表
        open_set = [(start_node.f, start_node)]
        
        # 关闭列表
        close_set = {}
        
        while open_set:
            self.sresult.numberofsteps += 1
            
            # 检查超时
            current_time = time.time()
            if current_time - begin_time > 1.0:  # 1秒超时
                print(f"⚠️ A* search timeout after {self.sresult.numberofsteps} steps")
                break
            
            # 检查最大步数
            if self.sresult.numberofsteps > 1000:  # 最大1000步
                # print(f"⚠️ A* search exceeded max steps ({self.sresult.numberofsteps})")
                pass
                break
            
            # 获取当前节点
            # Handle both old and new tuple formats for backward compatibility
            popped = heapq.heappop(open_set)
            if len(popped) == 3:  # New format with tie-breaker
                _, _, current = popped
            else:  # Old format without tie-breaker
                _, current = popped

            # 跳过已在关闭列表中的重复节点，避免重复扩张
            close_key = current.i * map_obj.width + current.j
            if close_key in close_set:
                continue

            # 检查是否到达目标
            if ((is_goal is not None and is_goal(start_node, current, map_obj, agent_set)) or
                (is_goal is None and current.i == goal_i and current.j == goal_j)):
                if self.check_goal(current, goal_time, agent_id, constraints):
                    self.sresult.pathfound = True
                    break

            # 将当前节点加入关闭列表
            close_set[close_key] = current
            
            # 生成后继节点
            if max_time == -1 or current.g < max_time:
                successors = self.find_successors(current, map_obj, goal_i, goal_j, agent_id, 
                                                occupied_nodes, constraints, with_cat, cat)
                for neighbor in successors:
                    neigh_key = neighbor.i * map_obj.width + neighbor.j
                    if neigh_key not in close_set:
                        neighbor.parent = current
                        neighbor.g = current.g + 1
                        neighbor.h = self.compute_h_from_cell_to_cell(neighbor.i, neighbor.j, goal_i, goal_j)
                        neighbor.f = neighbor.g + self.hweight * neighbor.h
                        # Use breakingties to ensure deterministic behavior
                        if self.breakingties:
                            # Add a tie-breaking value based on node position to ensure deterministic ordering
                            tie_breaker = neighbor.i * map_obj.width + neighbor.j
                            heapq.heappush(open_set, (neighbor.f, tie_breaker, neighbor))
                        else:
                            heapq.heappush(open_set, (neighbor.f, neighbor))
        
        end_time = time.time()
        self.sresult.time = end_time - begin_time
        self.sresult.nodescreated = len(open_set) + len(close_set)
        self.sresult.nodesexpanded = len(close_set)
        
        if self.sresult.pathfound:
            self.sresult.pathlength = current.g
            self.sresult.minF = current.f
            self.sresult.lastNode = current
            if return_path:
                self.make_primary_path(current, goal_time)
                self.make_secondary_path(map_obj)
                self.sresult.lppath = self.lppath
                self.sresult.hppath = self.hppath
        
        return self.sresult

    def search(self, start: Node, goal: Node, occupied_nodes=None):
        """保持向后兼容的search方法"""
        if occupied_nodes is None:
            occupied_nodes = set()
        
        # 使用startSearch方法
        result = self.startSearch(self.sub_map, None, start.i, start.j, goal.i, goal.j, 
                                occupied_nodes=occupied_nodes)
        
        if result.pathfound:
            return result.lppath
        return None

    def is_occupied(self, agent_set, i, j):
        """检查位置是否被agent占用"""
        if agent_set is None:
            return False
        for agent in agent_set:
            if agent.current.x == j and agent.current.y == i:
                return True
        return False

    def get_actor_id(self, agent_set, i, j):
        """获取占用位置的agent ID"""
        if agent_set is None:
            return -1
        for agent in agent_set:
            if agent.current.x == j and agent.current.y == i:
                return agent.id
        return -1

    def compute_h_from_cell_to_cell(self, start_i, start_j, fin_i, fin_j):
        """计算启发式值"""
        return abs(start_i - fin_i) + abs(start_j - fin_j)

    def find_successors(self, cur_node, map_obj, goal_i=0, goal_j=0, agent_id=-1, 
                       occupied_nodes=None, constraints=None, with_cat=False, cat=None):
        """生成后继节点"""
        if occupied_nodes is None:
            occupied_nodes = set()
        if constraints is None:
            constraints = set()
        
        successors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_i, new_j = cur_node.i + di, cur_node.j + dj
            # DEBUG: Log boundary check details
            in_bounds = map_obj.in_bounds(new_i, new_j) if hasattr(map_obj, 'in_bounds') else False
            is_traversable = map_obj.is_traversable(new_i, new_j)
            not_occupied = (new_i, new_j) not in occupied_nodes
            
            # Log when boundary is exceeded
            if not in_bounds:
                print(f"DEBUG BOUNDARY: Position ({new_i}, {new_j}) exceeds bounds - map size: {getattr(map_obj, 'width', 'unknown')}x{getattr(map_obj, 'height', 'unknown')}")
            
            if (is_traversable and not_occupied):
                neighbor = Node(new_i, new_j)
                successors.append(neighbor)
        return successors

    def check_goal(self, cur, goal_time, agent_id, constraints):
        """检查是否到达目标"""
        return goal_time == -1 or cur.g == goal_time

    def make_primary_path(self, cur_node, end_time):
        """构建主要路径"""
        self.lppath = []
        current = cur_node
        while current is not None:
            # DEBUG: Check if any path node is out of bounds
            # Use the map_obj that was passed to startSearch
            if hasattr(self, 'map_obj') and self.map_obj:
                if not self.map_obj.in_bounds(current.i, current.j):
                    print(f"DEBUG PATH: Out-of-bounds node in path: ({current.i}, {current.j}) - map size: {self.map_obj.width}x{self.map_obj.height}")
            self.lppath.insert(0, current)
            current = getattr(current, 'parent', None)
        
        # DEBUG: Log the complete path
        if self.lppath:
            path_coords = [(node.i, node.j) for node in self.lppath]
            print(f"DEBUG PATH: A* generated path with {len(path_coords)} nodes: {path_coords[:5]}...{path_coords[-5:] if len(path_coords) > 5 else path_coords}")
            # Additional check: verify all coordinates are within bounds
            if hasattr(self, 'map_obj') and self.map_obj:
                for i, (node_i, node_j) in enumerate(path_coords):
                    if not self.map_obj.in_bounds(node_i, node_j):
                        print(f"DEBUG PATH: Path node {i} ({node_i}, {node_j}) is OUT OF BOUNDS - map size: {self.map_obj.width}x{self.map_obj.height}")

    def make_secondary_path(self, map_obj):
        """构建次要路径"""
        self.hppath = []
        if not self.lppath:
            return
        
        self.hppath.append(self.lppath[0])
        for i in range(1, len(self.lppath)):
            prev = self.lppath[i-1]
            curr = self.lppath[i]
            if i + 1 < len(self.lppath):
                next_node = self.lppath[i+1]
                # 检查是否转向
                if ((curr.i - prev.i) * (next_node.j - curr.j) != 
                    (curr.j - prev.j) * (next_node.i - curr.i)):
                    self.hppath.append(curr)
            else:
                self.hppath.append(curr)

    def clear_lists(self):
        """清空列表"""
        self.lppath = []
        self.hppath = []

    def heuristic(self, node1, node2):
        """保持向后兼容的启发式方法"""
        if node2 is None:
            return 0
        return abs(node1.i - node2.i) + abs(node1.j - node2.j)

    def get_neighbors(self, node, occupied_nodes=None):
        """保持向后兼容的邻居方法"""
        if occupied_nodes is None:
            occupied_nodes = set()
        neighbors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = node.i + di, node.j + dj
            if self.sub_map.is_traversable(ni, nj) and (ni, nj) not in occupied_nodes:
                neighbors.append(Node(ni, nj))
        return neighbors

    def reconstruct_path(self, came_from, current):
        """保持向后兼容的路径重建方法"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path 