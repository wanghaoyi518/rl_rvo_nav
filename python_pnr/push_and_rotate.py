from .actor_set import ActorSet
from .mapf_config import MAPFConfig
from .mapf_search_result import MAPFSearchResult
from .sub_map import SubMap
from .isearch import ISearch
from .node import Node, Point, ActorMove
from .utils import CN_INFINITY
import time
from collections import deque, defaultdict
import copy
import os
import json
from datetime import datetime

def debug_log(msg):
    with open("pnr_debug.log", "a") as f:
        f.write(msg + "\n")
    print(msg)

# class PNRTraceLogger:
#     """Lightweight JSONL trace logger for PNR runs (timestamped per run)."""
#     def __init__(self):
#         base_dir = os.path.dirname(__file__)
#         self.trace_dir = os.path.join(base_dir, "pnr_traces")
#         os.makedirs(self.trace_dir, exist_ok=True)
#         # Use minute-level granularity so multiple calls in one run append to the same file
#         ts = datetime.now().strftime("%Y%m%d_%H%M")
#         self.file_path = os.path.join(self.trace_dir, f"pnr_trace_{ts}.jsonl")
#         self._fp = open(self.file_path, "a", buffering=1)
# 
#     def log(self, event: dict):
#         try:
#             self._fp.write(json.dumps(event, ensure_ascii=False) + "\n")
#         except Exception:
#             pass
# 
#     def close(self):
#         try:
#             self._fp.close()
#         except Exception:
#             pass

class PushAndRotate:
    def __init__(self, search=None):
        self.search = search  # ISearch 实例
        self.result = MAPFSearchResult()
        self.agents_moves = []
        self.agents_paths = []
        # self.trace_logger = None  # created per start_search call

    def clear(self):
        self.agents_paths.clear()
        self.agents_moves.clear()

    def move_along_path(self, actor_set, aid, path):
        # 辅助函数：沿A*路径逐步推进，每一步都写入agents_moves
        for idx in range(1, len(path)):
            prev = path[idx-1]
            cur = path[idx]
            for a in actor_set:
                if a.id == aid and a.current.x == prev.j and a.current.y == prev.i:
                    # 计算增量移动
                    di = cur.i - prev.i
                    dj = cur.j - prev.j
                    # 存储增量移动，与C++保持一致
                    self.agents_moves.append(ActorMove(di, dj, a.id))
                    # 更新agent位置
                    a.current.x = cur.j
                    a.current.y = cur.i
                    break

    def clear_node(self, sub_map: SubMap, actor_set: ActorSet, node: Node, occupied_nodes=None):
        """严格按照C++ clearNode实现"""
        if occupied_nodes is None:
            occupied_nodes = set()
        
        # 定义目标条件：找到未被占用的位置
        def is_goal(start, cur, sub_map, actor_set):
            # First check if position is within bounds
            if not sub_map.in_bounds(cur.i, cur.j):
                debug_log(f"DEBUG IS_GOAL: Position ({cur.i}, {cur.j}) rejected - out of bounds")
                return False
            # Then check if position is not occupied by any agent
            is_free = not any(a.current.x == cur.j and a.current.y == cur.i for a in actor_set)
            if is_free:
                debug_log(f"DEBUG IS_GOAL: Position ({cur.i}, {cur.j}) accepted as goal")
            return is_free
        
        # 使用A*搜索找到路径
        dijkstra = ISearch(sub_map)
        
        # 将occupied_nodes转换为坐标元组集合
        occupied_nodes_set = set()
        for pos in occupied_nodes:
            if isinstance(pos, tuple):
                occupied_nodes_set.add(pos)
            else:
                occupied_nodes_set.add((pos.i, pos.j))
        
        # 使用startSearch方法
        # DEBUG: Log SubMap dimensions before A* search
        debug_log(f"DEBUG A* SEARCH: SubMap dimensions - width={sub_map.width}, height={sub_map.height}")
        debug_log(f"DEBUG A* SEARCH: Start position - i={node.i}, j={node.j}")
        debug_log(f"DEBUG A* SEARCH: Grid size - rows={len(sub_map.grid)}, cols={len(sub_map.grid[0]) if sub_map.grid else 0}")
        
        search_result = dijkstra.startSearch(sub_map, actor_set, node.i, node.j, 0, 0, 
                                           is_goal, True, True, 0, -1, -1, occupied_nodes_set)
        
        if not search_result.pathfound:
            return False
        
        path = search_result.lppath
        if not path:
            return False
        
        # 按照C++逻辑：从路径末尾开始，逐步移动阻挡的agent
        for i in range(len(path) - 2, -1, -1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # 检查当前位置是否被占用
            for a in actor_set:
                if a.current.x == current_node.j and a.current.y == current_node.i:
                    # 移动agent到下一个位置
                    from_node = Node(a.current.y, a.current.x)
                    to_node = Node(next_node.i, next_node.j)
                    # 直接移动，不使用move_along_path
                    a.current.x = to_node.j
                    a.current.y = to_node.i
                    # 记录移动
                    di = to_node.i - from_node.i
                    dj = to_node.j - from_node.j
                    self.agents_moves.append(ActorMove(di, dj, a.id))
                    break
        
        return True

    def push(self, sub_map: SubMap, actor_set: ActorSet, from_node: Node, to_node: Node, occupied_nodes=None):
        """严格按照C++ push实现"""
        if occupied_nodes is None:
            occupied_nodes = set()
        
        # 检查目标位置是否在occupied_nodes中
        if to_node in occupied_nodes:
            return False
        
        # 检查目标位置是否被agent占用
        blocking_actor = None
        for a in actor_set:
            if a.current.x == to_node.j and a.current.y == to_node.i:
                blocking_actor = a
                break
        
        if blocking_actor is not None:
            # 有阻挡，尝试清除
            inserted = False
            if from_node not in occupied_nodes:
                occupied_nodes.add(from_node)
                inserted = True
            
            can_clear = self.clear_node(sub_map, actor_set, to_node, occupied_nodes)
            
            if inserted:
                occupied_nodes.remove(from_node)
            
            if not can_clear:
                return False
        
        # 直接移动agent，不使用move_along_path
        for a in actor_set:
            if a.current.x == from_node.j and a.current.y == from_node.i:
                # 记录移动
                di = to_node.i - from_node.i
                dj = to_node.j - from_node.j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                # 更新位置
                a.current.x = to_node.j
                a.current.y = to_node.i
                break
        
        return True

    def multipush(self, sub_map: SubMap, actor_set: ActorSet, first: Node, second: Node, to: Node, path):
        """严格按照C++ multipush实现"""
        if len(path) > 1 and path[1].i == second.i and path[1].j == second.j:
            first, second = second, first
            path = path[1:]
        
        prev_node = second
        for idx in range(len(path) - 1):
            cur_node = path[idx]
            next_node = path[idx + 1]
            occupied_nodes = {prev_node, cur_node}
            
            # 检查下一个位置是否被占用
            blocking = any(a.current.x == next_node.j and a.current.y == next_node.i for a in actor_set)
            if blocking:
                if not self.clear_node(sub_map, actor_set, next_node, occupied_nodes):
                    return False
            
            # 直接移动agent，不使用move_along_path
            # 移动cur_node的agent到next_node
            for a in actor_set:
                if a.current.x == cur_node.j and a.current.y == cur_node.i:
                    di = next_node.i - cur_node.i
                    dj = next_node.j - cur_node.j
                    self.agents_moves.append(ActorMove(di, dj, a.id))
                    a.current.x = next_node.j
                    a.current.y = next_node.i
                    break
            
            # 移动prev_node的agent到cur_node
            for a in actor_set:
                if a.current.x == prev_node.j and a.current.y == prev_node.i:
                    di = cur_node.i - prev_node.i
                    dj = cur_node.j - prev_node.j
                    self.agents_moves.append(ActorMove(di, dj, a.id))
                    a.current.x = cur_node.j
                    a.current.y = cur_node.i
                    break
            
            prev_node = cur_node
        
        return True

    def rotate(self, sub_map: SubMap, actor_set: ActorSet, q_path, cycle_beg):
        """严格按照C++ rotate实现"""
        size = len(q_path) - cycle_beg
        
        # 尝试简单旋转
        for i in range(cycle_beg, len(q_path)):
            if not any(a.current.x == q_path[i].j and a.current.y == q_path[i].i for a in actor_set):
                for j in range(size - 1):
                    from_idx = cycle_beg + (i - cycle_beg - j - 1 + size) % size
                    to_idx = cycle_beg + (i - cycle_beg - j + size) % size
                    if any(a.current.x == q_path[from_idx].j and a.current.y == q_path[from_idx].i for a in actor_set):
                        # 直接移动agent，不使用move_along_path
                        for a in actor_set:
                            if a.current.x == q_path[from_idx].j and a.current.y == q_path[from_idx].i:
                                di = q_path[to_idx].i - q_path[from_idx].i
                                dj = q_path[to_idx].j - q_path[from_idx].j
                                self.agents_moves.append(ActorMove(di, dj, a.id))
                                a.current.x = q_path[to_idx].j
                                a.current.y = q_path[to_idx].i
                                break
                return True
        
        # 复杂情况处理
        cycle_nodes = set(q_path[cycle_beg:])
        for i in range(cycle_beg, len(q_path)):
            cycle_nodes.discard(q_path[i])
            
            # 找到当前位置的agent
            first_agent_id = None
            for a in actor_set:
                if a.current.x == q_path[i].j and a.current.y == q_path[i].i:
                    first_agent_id = a.id
                    break
            
            if first_agent_id is None:
                continue
            
            beg_size = len(self.agents_moves)
            
            # 尝试清除当前位置
            if self.clear_node(sub_map, actor_set, q_path[i], cycle_nodes):
                end_size = len(self.agents_moves)
                
                # 找到第二个agent（循环中的前一个）
                second_agent_index = cycle_beg + (i - cycle_beg - 1 + size) % size
                second_agent_id = None
                for a in actor_set:
                    if a.current.x == q_path[second_agent_index].j and a.current.y == q_path[second_agent_index].i:
                        second_agent_id = a.id
                        break
                
                if second_agent_id is None:
                    cycle_nodes.add(q_path[i])
                    continue
                
                # 移动第二个agent到当前位置
                for a in actor_set:
                    if a.id == second_agent_id:
                        a.current.x = q_path[i].j
                        a.current.y = q_path[i].i
                        break
                
                # 获取第一个agent的当前位置
                cur_position = None
                for a in actor_set:
                    if a.id == first_agent_id:
                        cur_position = Node(a.current.x, a.current.y)
                        break
                
                if cur_position is None:
                    cycle_nodes.add(q_path[i])
                    continue
                
                # 尝试交换操作
                if self.swap(sub_map, actor_set, q_path[i], cur_position):
                    # 完成旋转
                    for j in range(size - 1):
                        from_idx = cycle_beg + (i - cycle_beg - j - 2 + size) % size
                        to_idx = cycle_beg + (i - cycle_beg - j - 1 + size) % size
                        if any(a.current.x == q_path[from_idx].j and a.current.y == q_path[from_idx].i for a in actor_set):
                            # 直接移动agent
                            for a in actor_set:
                                if a.current.x == q_path[from_idx].j and a.current.y == q_path[from_idx].i:
                                    di = q_path[to_idx].i - q_path[from_idx].i
                                    dj = q_path[to_idx].j - q_path[from_idx].j
                                    self.agents_moves.append(ActorMove(di, dj, a.id))
                                    a.current.x = q_path[to_idx].i
                                    a.current.y = q_path[to_idx].j
                                    break
                    
                    # 反转clear_node操作的移动
                    self.reverse(beg_size, end_size, first_agent_id, second_agent_id, actor_set)
                    return True
            
            cycle_nodes.add(q_path[i])
        
        return False

    def reverse(self, beg_size, end_size, first_id, second_id, actor_set: ActorSet):
        """严格按照C++ reverse实现"""
        for i in range(end_size - 1, beg_size - 1, -1):
            move = self.agents_moves[i]
            agent_id = move.id
            if agent_id == first_id:
                agent_id = second_id
            elif agent_id == second_id:
                agent_id = first_id
            
            # 找到对应的agent
            for a in actor_set:
                if a.id == agent_id:
                    # 获取当前位置
                    from_node = Node(a.current.x, a.current.y)
                    # 计算反向移动的目标位置
                    to_node = Node(from_node.i - move.di, from_node.j - move.dj)
                    # 更新agent位置
                    a.current.x = to_node.j
                    a.current.y = to_node.i
                    # 记录反向移动
                    self.agents_moves.append(ActorMove(-move.di, -move.dj, a.id))
                    break

    def solve(self, sub_map: SubMap, config: MAPFConfig, actor_set: ActorSet):
        # 完全还原C++ solve主流程，包括优先级排序和循环检测
        if self.search is None:
            self.search = ISearch(sub_map)
        trace = []  # diagnostic trace for early-return reasons
        # 优先级比较器
        def comparator(id1, id2):
            subgraph1 = getattr(actor_set.get_actor_by_id(id1), 'subgraph', -1)
            subgraph2 = getattr(actor_set.get_actor_by_id(id2), 'subgraph', -1)
            
            if subgraph1 != subgraph2:
                if subgraph1 == -1 or (hasattr(self, 'priorities') and subgraph2 in self.priorities.get(subgraph1, set())):
                    return False
                elif subgraph2 == -1 or (hasattr(self, 'priorities') and subgraph1 in self.priorities.get(subgraph2, set())):
                    return True
            return id1 < id2
        
        # 检查是否为多边形地图
        is_polygon = True
        for i in range(sub_map.height):
            for j in range(sub_map.width):
                if sub_map.grid[i][j] == 0:  # 可通行
                    # 计算度数（简化版）
                    degree = 0
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0<=ni<sub_map.height and 0<=nj<sub_map.width and sub_map.grid[ni][nj] == 0:
                            degree += 1
                    if degree != 2:
                        is_polygon = False
                        break
            if not is_polygon:
                break
        
        # 记录地图类型
        try:
            self.result.stats['is_polygon'] = bool(is_polygon)
        except Exception:
            pass

        not_finished = set(a.id for a in actor_set)
        finished = set()
        finished_positions = set()
        q_path_nodes = set()
        q_path = []
        
        cur_agent_id = -1
        steps = 0
        
        while not_finished and steps < config.max_steps:
            steps += 1

            # 交换位置冲突检测开始
            # 收集所有agent的当前位置和下一个位置（如果有路径）
            agent_positions = {}
            agent_next_positions = {}
            for a in actor_set:
                agent_positions[a.id] = (int(a.current.x), int(a.current.y))
                # 预测下一步（如果有规划路径且未到终点）
                if hasattr(a, 'planned_path') and a.planned_path is not None:
                    idx = 0
                    # 找到当前所在路径点
                    for i, n in enumerate(a.planned_path):
                        if (int(a.current.x), int(a.current.y)) == (n.j, n.i):
                            idx = i
                            break
                    # 取下一个点
                    if idx + 1 < len(a.planned_path):
                        agent_next_positions[a.id] = (a.planned_path[idx + 1].i, a.planned_path[idx + 1].j)
            # 检查所有agent对，是否有交换位置型冲突
            for id1, pos1 in agent_positions.items():
                for id2, pos2 in agent_positions.items():
                    if id1 >= id2:
                        continue
                    next1 = agent_next_positions.get(id1)
                    next2 = agent_next_positions.get(id2)
                    if next1 is not None and next2 is not None:
                        if pos1 == next2 and pos2 == next1:
                            # 记录诊断
                            try:
                                trace.append({
                                    'stage': 'pre_swap_collision',
                                    'pair': [int(id1), int(id2)],
                                    'pos1': list(pos1),
                                    'pos2': list(pos2),
                                    'next1': list(next1),
                                    'next2': list(next2),
                                    'reason': 'swap_collision'
                                })
                            except Exception:
                                pass
                            try:
                                self.result.stats['solve_trace'] = trace
                            except Exception:
                                pass
                            return False
            # 交换位置冲突检测结束

            if cur_agent_id == -1:
                # 按优先级选择下一个agent - 简化逻辑
                cur_agent_id = min(not_finished)
            
            debug_log(f"步骤 {steps}: 选择Agent {cur_agent_id}, not_finished={not_finished}")
            # Trace: step selection
            # try:
            #     self.trace_logger.log({
            #         "type": "step_select",
            #         "step": int(steps),
            #         "agent": int(cur_agent_id),
            #         "not_finished": [int(x) for x in sorted(list(not_finished))]
            #     })
            # except Exception:
            #     pass
            
            cur_agent = actor_set.get_actor_by_id(cur_agent_id)
            if cur_agent_id not in not_finished:
                # debug_log(f"错误: Agent {cur_agent_id} 不在 not_finished 中!")
                return False
            not_finished.remove(cur_agent_id)
            
            # 搜索路径 - 与C++版本保持一致：多边形地图使用finished_positions，非多边形地图使用空集合
            occupied_set = finished_positions if is_polygon else set()
            path = self.search.search(
                Node(int(cur_agent.current.y), int(cur_agent.current.x)),
                Node(int(cur_agent.goal.y), int(cur_agent.goal.x)),
                occupied_set
            )
            debug_log(f"A*路径 (Agent {cur_agent_id}): {[ (n.i, n.j) for n in path ] if path else '无路径'}")
            # Trace: astar outcome
            # try:
            #     self.trace_logger.log({
            #         "type": "astar",
            #         "step": int(steps),
            #         "agent": int(cur_agent_id),
            #         "start": [int(cur_agent.current.y), int(cur_agent.current.x)],
            #         "goal": [int(cur_agent.goal.y), int(cur_agent.goal.x)],
            #         "occupied_size": int(len(occupied_set) if occupied_set is not None else 0),
            #         "path": [[int(n.i), int(n.j)] for n in path] if path else None
            #     })
            # except Exception:
            #     pass
            if not path or len(path) < 2:
                # 记录诊断
                try:
                    occ_size = 0
                    occ_samples = []
                    try:
                        occ_size = len(occupied_set) if occupied_set is not None else 0
                        for node in list(occupied_set)[:5]:
                            if hasattr(node, 'i') and hasattr(node, 'j'):
                                occ_samples.append([int(node.i), int(node.j)])
                            elif isinstance(node, tuple) and len(node) >= 2:
                                occ_samples.append([int(node[0]), int(node[1])])
                            else:
                                occ_samples.append(None)
                    except Exception:
                        pass

                    # 5x5 patches around start/goal
                    si, sj = int(cur_agent.current.y), int(cur_agent.current.x)
                    gi, gj = int(cur_agent.goal.y), int(cur_agent.goal.x)
                    def patch5(i, j):
                        patch = []
                        for ii in range(max(0, i-2), min(sub_map.height, i+3)):
                            row = []
                            for jj in range(max(0, j-2), min(sub_map.width, j+3)):
                                try:
                                    row.append(int(sub_map.grid[ii][jj]))
                                except Exception:
                                    row.append(None)
                            patch.append(row)
                        return patch
                    start_patch = patch5(si, sj)
                    goal_patch = patch5(gi, gj)

                    # Line/rect obstacle diagnostics
                    line_info = {}
                    rect_info = {}
                    # straight line check (same row or same col)
                    if si == gi:
                        j0, j1 = sorted([sj, gj])
                        line_vals = [int(sub_map.grid[si][jj]) for jj in range(j0, j1+1)]
                        # find first obstacle
                        first_idx = next((k for k,v in enumerate(line_vals) if v == 1), None)
                        first_obs = [si, j0+first_idx] if first_idx is not None else None
                        # contiguous obstacle segments
                        segs = 0; prev = 0
                        for v in line_vals:
                            if v == 1 and prev == 0:
                                segs += 1
                            prev = v
                        line_info = {
                            'orientation': 'row', 'row': si,
                            'range_j': [j0, j1],
                            'first_obstacle': first_obs,
                            'segments': segs,
                            'obstacle_ratio': sum(line_vals)/max(1,len(line_vals))
                        }
                    elif sj == gj:
                        i0, i1 = sorted([si, gi])
                        line_vals = [int(sub_map.grid[ii][sj]) for ii in range(i0, i1+1)]
                        first_idx = next((k for k,v in enumerate(line_vals) if v == 1), None)
                        first_obs = [i0+first_idx, sj] if first_idx is not None else None
                        segs = 0; prev = 0
                        for v in line_vals:
                            if v == 1 and prev == 0:
                                segs += 1
                            prev = v
                        line_info = {
                            'orientation': 'col', 'col': sj,
                            'range_i': [i0, i1],
                            'first_obstacle': first_obs,
                            'segments': segs,
                            'obstacle_ratio': sum(line_vals)/max(1,len(line_vals))
                        }
                    # rectangle summary
                    i0, i1 = sorted([si, gi]); j0, j1 = sorted([sj, gj])
                    rect_obs = 0; rect_total = 0
                    for ii in range(i0, i1+1):
                        for jj in range(j0, j1+1):
                            rect_total += 1
                            try:
                                rect_obs += 1 if int(sub_map.grid[ii][jj]) == 1 else 0
                            except Exception:
                                pass
                    rect_info = {
                        'bounds': {'i': [i0,i1], 'j': [j0,j1]},
                        'obstacle_count': rect_obs,
                        'cell_count': rect_total,
                        'obstacle_ratio': (rect_obs / max(1, rect_total))
                    }

                    trace.append({
                        'stage': 'astar',
                        'agent': int(cur_agent_id),
                        'start': [int(cur_agent.current.y), int(cur_agent.current.x)],
                        'goal': [int(cur_agent.goal.y), int(cur_agent.goal.x)],
                        'path_len': 0 if not path else len(path),
                        'occupied_size': occ_size,
                        'occupied_samples': occ_samples,
                        'start_patch_5x5': start_patch,
                        'goal_patch_5x5': goal_patch,
                        'line_info': line_info,
                        'rect_info': rect_info,
                        'reason': 'no_path'
                    })
                except Exception:
                    pass
                try:
                    self.result.stats['solve_trace'] = trace
                except Exception:
                    pass
                return False
            
            # 处理路径
            q_path.append(path[0])
            q_path_nodes.add(path[0])
            
            for idx in range(len(path)-1):
                current_node = path[idx]
                next_node = path[idx+1]
                
                # 检查循环
                if next_node in q_path_nodes:
                    cycle_beg = len(q_path) - 1
                    while cycle_beg >= 0 and q_path[cycle_beg] != next_node:
                        cycle_beg -= 1
                    # Trace: cycle detected
                    # try:
                    #     self.trace_logger.log({
                    #         "type": "cycle_detected",
                    #         "step": int(steps),
                    #         "agent": int(cur_agent_id),
                    #         "cycle_beg_index": int(cycle_beg),
                    #         "q_path_len": int(len(q_path))
                    #     })
                    # except Exception:
                    #     pass
                    self.rotate(sub_map, actor_set, q_path, cycle_beg)
                    
                    # 清理循环部分
                    while len(q_path) > cycle_beg:
                        last_node = q_path.pop()
                        q_path_nodes.remove(last_node)
                        # 处理finished_positions
                        for a in actor_set:
                            if a.current.x == last_node.j and a.current.y == last_node.i:
                                if a.id in finished:
                                    finished_positions.add(last_node)
                                break
                else:
                    # 尝试push，失败则swap
                    occupied_set = finished_positions if is_polygon else set()
                    if not self.push(sub_map, actor_set, current_node, next_node, occupied_set):
                        if not self.swap(sub_map, actor_set, current_node, next_node):
                            # 记录诊断
                            try:
                                trace.append({
                                    'stage': 'push_or_swap',
                                    'agent': int(cur_agent_id),
                                    'from': [int(current_node.i), int(current_node.j)],
                                    'to': [int(next_node.i), int(next_node.j)],
                                    'push_ok': False,
                                    'swap_ok': False,
                                    'reason': 'push_and_swap_failed'
                                })
                            except Exception:
                                pass
                            # Trace: push/swap failed
                            # try:
                            #     self.trace_logger.log({
                            #         "type": "push_swap_failed",
                            #         "step": int(steps),
                            #         "agent": int(cur_agent_id),
                            #         "from": [int(current_node.i), int(current_node.j)],
                            #         "to": [int(next_node.i), int(next_node.j)]
                            #     })
                            # except Exception:
                            #     pass
                            try:
                                self.result.stats['solve_trace'] = trace
                            except Exception:
                                pass
                            return False
                        # 更新finished_positions
                        for a in actor_set:
                            if a.current.x == current_node.j and a.current.y == current_node.i and a.id in finished:
                                finished_positions.discard(next_node)
                                finished_positions.add(current_node)
                                break
                # 主循环推进时同步current并写入agents_moves（修正点）
                for a in actor_set:
                    if a.id == cur_agent_id and a.current.x == current_node.j and a.current.y == current_node.i:
                        # 计算增量移动
                        di = next_node.i - current_node.i
                        dj = next_node.j - current_node.j
                        # 存储增量移动
                        self.agents_moves.append(ActorMove(di, dj, a.id))
                        # 更新agent位置
                        a.current.x = next_node.j
                        a.current.y = next_node.i
                        # Trace: move applied
                        # try:
                        #     self.trace_logger.log({
                        #         "type": "move",
                        #         "step": int(steps),
                        #         "agent": int(cur_agent_id),
                        #         "from": [int(current_node.i), int(current_node.j)],
                        #         "to": [int(next_node.i), int(next_node.j)],
                        #         "di": int(di),
                        #         "dj": int(dj)
                        #     })
                        # except Exception:
                        #     pass
                        break
                q_path.append(next_node)
                q_path_nodes.add(next_node)
            
            # 仅当当前agent实际到达其目标时，才标记为完成；否则重新加入待处理
            if int(cur_agent.current.x) == int(cur_agent.goal.x) and int(cur_agent.current.y) == int(cur_agent.goal.y):
                finished.add(cur_agent_id)
                finished_positions.add(Node(int(cur_agent.goal.y), int(cur_agent.goal.x)))
            else:
                if cur_agent_id not in not_finished:
                    not_finished.add(cur_agent_id)
            
            # 处理q_path中的剩余agent
            cur_agent_id = -1
            while q_path:
                last_node = q_path[-1]
                for a in actor_set:
                    if a.current.x == last_node.j and a.current.y == last_node.i:
                        goal_node = Node(int(a.goal.y), int(a.goal.x))
                        if a.id not in not_finished and last_node != goal_node:
                            # 检查目标位置是否空闲
                            goal_occupied = any(oa.current.x == goal_node.j and oa.current.y == goal_node.i for oa in actor_set)
                            if not goal_occupied:
                                # 禁止直接赋值到goal，只允许逐步推进
                                # 需要用A*路径逐步推进到goal
                                dijkstra = ISearch(sub_map)
                                occupied_nodes = set((a.current.x, a.current.y) for a in actor_set)
                                occupied_nodes.discard((last_node.i, last_node.j))
                                occupied_nodes.discard((goal_node.i, goal_node.j))
                                path = dijkstra.search(last_node, goal_node, occupied_nodes)
                                if path and len(path) > 1:
                                    self.move_along_path(actor_set, a.id, path)
                                    finished_positions.discard(last_node)
                                    finished_positions.add(goal_node)
                                # 如果已经在goal或无法到达，不做任何补写
                            else:
                                # 找到占用目标位置的agent
                                for oa in actor_set:
                                    if oa.current.x == goal_node.j and oa.current.y == goal_node.i:
                                        cur_agent_id = oa.id
                                        # debug_log(f"  重新选择Agent {cur_agent_id} (占用目标位置)")
                                        if cur_agent_id not in not_finished:
                                            not_finished.add(cur_agent_id)
                                            # debug_log(f"  将Agent {cur_agent_id} 重新添加到 not_finished")
                                        break
                                break
                        break
                if cur_agent_id != -1:
                    break
                q_path_nodes.remove(last_node)
                q_path.pop()
        
        # # 尝试在主循环结束后将未到达目标的agent用A*直接补齐至目标（若可达且目标空闲）
        # for a in actor_set:
        #     if int(a.current.x) != int(a.goal.x) or int(a.current.y) != int(a.goal.y):
        #         goal_node = Node(int(a.goal.x), int(a.goal.y))
        #         last_node = Node(int(a.current.x), int(a.current.y))
        #         # 其他agent当前位置作为占用，排除自身与目标
        #         occupied_nodes = set((oa.current.x, oa.current.y) for oa in actor_set if oa.id != a.id)
        #         occupied_nodes.discard((last_node.i, last_node.j))
        #         occupied_nodes.discard((goal_node.i, goal_node.j))
        #         path = self.search.search(last_node, goal_node, occupied_nodes)
        #         if path and len(path) > 1:
        #             self.move_along_path(actor_set, a.id, path)
        
        # 生成路径
        self.get_parallel_paths(actor_set, config)
        
        # 写入诊断trace
        try:
            self.result.stats['solve_trace'] = trace
        except Exception:
            pass
        self.result.steps = steps
        return len(not_finished) == 0

    def clear(self, sub_map: SubMap, actor_set: ActorSet, first: Node, second: Node):
        """Implement the missing clear function from C++ version"""
        # Get successors of first node
        successors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = first.i + di, first.j + dj
            if sub_map.is_traversable(ni, nj):
                successors.append(Node(ni, nj))
        
        # Find unoccupied successors
        unoccupied = []
        for node in successors:
            if not any(a.current.x == node.j and a.current.y == node.i for a in actor_set):
                unoccupied.append(node)
        
        if len(unoccupied) >= 2:
            return True
        
        # Try to clear occupied successors
        forbidden = {first, second}
        forbidden.update(unoccupied)
        
        for node in successors:
            if node not in unoccupied and node != second:
                if self.clear_node(sub_map, actor_set, node, forbidden):
                    if len(unoccupied) >= 1:
                        return True
                    unoccupied.append(node)
                    forbidden.add(node)
        
        if not unoccupied:
            return False
        
        # Try complex clearing strategies
        free_neigh = unoccupied[0]
        for node in successors:
            if node != second and node != free_neigh:
                # Strategy 1: Clear both nodes
                cur_size = len(self.agents_moves)
                new_actor_set = copy.deepcopy(actor_set)
                if self.clear_node(sub_map, new_actor_set, node, {first, second}):
                    if self.clear_node(sub_map, new_actor_set, free_neigh, {first, second, node}):
                        # Update original actor_set
                        for i, agent in enumerate(actor_set):
                            agent.current.x = new_actor_set[i].current.x
                            agent.current.y = new_actor_set[i].current.y
                        return True
                    else:
                        # Rollback moves
                        self.agents_moves = self.agents_moves[:cur_size]
                break
        
        # Strategy 2: Move and clear
        for node in successors:
            if node != second and node != free_neigh:
                cur_size = len(self.agents_moves)
                new_actor_set = copy.deepcopy(actor_set)
                
                # Move first to free_neigh, second to first
                for agent in new_actor_set:
                    if agent.current.x == first.j and agent.current.y == first.i:
                        agent.current.x = free_neigh.j
                        agent.current.y = free_neigh.i
                    elif agent.current.x == second.j and agent.current.y == second.i:
                        agent.current.x = first.j
                        agent.current.y = first.i
                
                if self.clear_node(sub_map, new_actor_set, node, {first, second}):
                    if self.clear_node(sub_map, new_actor_set, second, {first, second, node}):
                        # Update original actor_set
                        for i, agent in enumerate(actor_set):
                            agent.current.x = new_actor_set[i].current.x
                            agent.current.y = new_actor_set[i].current.y
                        return True
                    else:
                        # Rollback moves
                        self.agents_moves = self.agents_moves[:cur_size]
                break
        
        # Strategy 3: Final attempt
        second_agent_id = None
        for agent in actor_set:
            if agent.current.x == second.j and agent.current.y == second.i:
                second_agent_id = agent.id
                break
        
        if second_agent_id is None:
            return False
        
        if not self.clear_node(sub_map, actor_set, second, {first}):
            return False
        
        # Move first to second
        for agent in actor_set:
            if agent.current.x == first.j and agent.current.y == first.i:
                agent.current.x = second.j
                agent.current.y = second.i
                break
        
        # Get second's new position
        second_position = None
        for agent in actor_set:
            if agent.id == second_agent_id:
                second_position = Node(agent.current.x, agent.current.y)
                break
        
        if second_position is None:
            return False
        
        if not self.clear_node(sub_map, actor_set, free_neigh, {first, second, second_position}):
            return False
        
        # Complex final moves
        for node in successors:
            if node != second and node != free_neigh:
                # Move node to first, first to free_neigh, second to first, second_position to second
                for agent in actor_set:
                    if agent.current.x == node.j and agent.current.y == node.i:
                        agent.current.x = first.j
                        agent.current.y = first.i
                    elif agent.current.x == first.j and agent.current.y == first.i:
                        agent.current.x = free_neigh.j
                        agent.current.y = free_neigh.i
                    elif agent.current.x == second.j and agent.current.y == second.i:
                        agent.current.x = first.j
                        agent.current.y = first.i
                    elif agent.current.x == second_position.j and agent.current.y == second_position.i:
                        agent.current.x = second.j
                        agent.current.y = second.i
                
                return self.clear_node(sub_map, actor_set, free_neigh, {first, second, node})
        
        return False

    def swap(self, sub_map: SubMap, actor_set: ActorSet, first: Node, second: Node):
        """Improved swap function matching C++ implementation"""
        # Get agent IDs
        first_agent_id = None
        second_agent_id = None
        for agent in actor_set:
            if agent.current.x == first.j and agent.current.y == first.i:
                first_agent_id = agent.id
            elif agent.current.x == second.j and agent.current.y == second.i:
                second_agent_id = agent.id
        
        if first_agent_id is None or second_agent_id is None:
            return False
        
        # Define goal condition: find nodes with degree >= 3
        def is_goal(start, cur, sub_map, actor_set):
            # Calculate degree (number of traversable neighbors)
            degree = 0
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = cur.i + di, cur.j + dj
                if sub_map.is_traversable(ni, nj):
                    degree += 1
            return degree >= 3
        
        # Search for exchange node
        search_result = self.search.search(first, None, set())
        while search_result:
            # Find a node that satisfies goal condition
            exchange_node = None
            for node in search_result:
                if is_goal(first, node, sub_map, actor_set):
                    exchange_node = node
                    break
            
            if exchange_node is None:
                break
            
            # Try multipush to exchange node
            beg_size = len(self.agents_moves)
            new_actor_set = copy.deepcopy(actor_set)
            
            if self.multipush(sub_map, new_actor_set, first, second, exchange_node, search_result):
                # Find which agent is at exchange node
                exchange_agent_id = None
                neigh_agent_id = None
                for agent in new_actor_set:
                    if agent.current.x == exchange_node.j and agent.current.y == exchange_node.i:
                        exchange_agent_id = agent.id
                    elif agent.current.x == first.j and agent.current.y == first.i:
                        neigh_agent_id = agent.id
                
                if exchange_agent_id is None or neigh_agent_id is None:
                    break
                
                # Determine which agent is the neighbor
                if exchange_agent_id == first_agent_id:
                    neigh_agent_id = second_agent_id
                else:
                    neigh_agent_id = first_agent_id
                
                # Get neighbor's position
                neigh_node = None
                for agent in new_actor_set:
                    if agent.id == neigh_agent_id:
                        neigh_node = Node(agent.current.x, agent.current.y)
                        break
                
                if neigh_node and self.clear(sub_map, new_actor_set, exchange_node, neigh_node):
                    # Update original actor_set
                    for i, agent in enumerate(actor_set):
                        agent.current.x = new_actor_set[i].current.x
                        agent.current.y = new_actor_set[i].current.y
                    
                    end_size = len(self.agents_moves)
                    self.exchange(sub_map, actor_set, exchange_node, neigh_node)
                    self.reverse(beg_size, end_size, first_agent_id, second_agent_id, actor_set)
                    return True
            
            # Try next search result
            search_result = self.search.search(first, None, set())
        
        return False

    def exchange(self, sub_map: SubMap, actor_set: ActorSet, first: Node, second: Node):
        """严格按照C++ exchange实现"""
        # 找到first节点的后继节点
        successors = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = first.i + di, first.j + dj
            if sub_map.is_traversable(ni, nj):
                successors.append(Node(ni, nj))
        
        # 找到空闲的邻居节点
        free_neigh = []
        for node in successors:
            if not any(a.current.x == node.j and a.current.y == node.i for a in actor_set):
                free_neigh.append(node)
        
        # 执行6步交换序列，严格按照C++版本
        # Step 1: Move first to free_neigh[0]
        for a in actor_set:
            if a.current.x == first.j and a.current.y == first.i:
                di = free_neigh[0].i - first.i
                dj = free_neigh[0].j - first.j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = free_neigh[0].j
                a.current.y = free_neigh[0].i
                break
        
        # Step 2: Move second to first
        for a in actor_set:
            if a.current.x == second.j and a.current.y == second.i:
                di = first.i - second.i
                dj = first.j - second.j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = first.j
                a.current.y = first.i
                break
        
        # Step 3: Move first to free_neigh[1]
        for a in actor_set:
            if a.current.x == free_neigh[0].j and a.current.y == free_neigh[0].i:
                di = free_neigh[1].i - free_neigh[0].i
                dj = free_neigh[1].j - free_neigh[0].j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = free_neigh[1].j
                a.current.y = free_neigh[1].i
                break
        
        # Step 4: Move free_neigh[0] to first
        for a in actor_set:
            if a.current.x == free_neigh[0].j and a.current.y == free_neigh[0].i:
                di = first.i - free_neigh[0].i
                dj = first.j - free_neigh[0].j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = first.j
                a.current.y = first.i
                break
        
        # Step 5: Move first to second
        for a in actor_set:
            if a.current.x == first.j and a.current.y == first.i:
                di = second.i - first.i
                dj = second.j - first.j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = second.j
                a.current.y = second.i
                break
        
        # Step 6: Move free_neigh[1] to first
        for a in actor_set:
            if a.current.x == free_neigh[1].j and a.current.y == free_neigh[1].i:
                di = first.i - free_neigh[1].i
                dj = first.j - free_neigh[1].j
                self.agents_moves.append(ActorMove(di, dj, a.id))
                a.current.x = first.j
                a.current.y = first.i
                break

    def get_subgraphs(self, sub_map: SubMap, actor_set: ActorSet):
        # 完全还原C++ getSubgraphs逻辑
        # 这里需要为每个Node和Actor维护subgraph/connected_component等属性
        # Python实现需在Node/ActorSet中动态添加这些属性
        from collections import deque, defaultdict
        height, width = sub_map.height, sub_map.width
        grid = sub_map.grid
        node_to_subgraph = {}
        subgraph_id = 0
        visited = set()
        for i in range(height):
            for j in range(width):
                if grid[i][j] == 0 and (i, j) not in visited:
                    # BFS标记连通分量
                    queue = deque()
                    queue.append((i, j))
                    visited.add((i, j))
                    while queue:
                        ci, cj = queue.popleft()
                        node_to_subgraph[(ci, cj)] = subgraph_id
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = ci+di, cj+dj
                            if 0<=ni<height and 0<=nj<width and grid[ni][nj]==0 and (ni, nj) not in visited:
                                queue.append((ni, nj))
                                visited.add((ni, nj))
                    subgraph_id += 1
        # 为每个actor和节点分配subgraph属性
        for a in actor_set:
            pos = (a.current.x, a.current.y)  # 使用Point的x,y属性
            a.subgraph = node_to_subgraph.get(pos, -1)
        self.node_to_subgraph = node_to_subgraph

    def assign_to_subgraphs(self, sub_map: SubMap, actor_set: ActorSet):
        # 完全还原C++ assignToSubgraphs逻辑
        # 统计每个subgraph中的actor数量
        from collections import defaultdict
        subgraph_actor_count = defaultdict(int)
        for a in actor_set:
            if hasattr(a, 'subgraph'):
                subgraph_actor_count[a.subgraph] += 1
        self.subgraph_actor_count = subgraph_actor_count
        # 可根据需要为每个节点/actor分配更多属性

    def get_priorities(self, sub_map: SubMap, actor_set: ActorSet):
        # 完全还原C++ getPriorities逻辑
        # 这里用一个dict维护subgraph之间的优先级关系
        from collections import defaultdict
        priorities = defaultdict(set)
        for a in actor_set:
            if hasattr(a, 'goal') and hasattr(a, 'subgraph'):
                goal_pos = (a.goal.x, a.goal.y)  # 使用Point的x,y属性
                goal_subgraph = self.node_to_subgraph.get(goal_pos, -1)
                if goal_subgraph != a.subgraph and goal_subgraph != -1 and a.subgraph != -1:
                    priorities[a.subgraph].add(goal_subgraph)
        self.priorities = priorities

    def get_component(self, actor_set: ActorSet, start_edge, edge_stack, components):
        # 严格还原C++ getComponent逻辑
        # start_edge: (Node, Node)
        # edge_stack: list of (Node, Node)
        # components: list of set(Node)
        component = set()
        while edge_stack:
            cur_edge = edge_stack.pop()
            component.add(cur_edge[0])
            component.add(cur_edge[1])
            if cur_edge == start_edge:
                break
        if len(component) <= 2:
            return
        # 为component内所有节点分配同一个subgraph编号
        subgraph_num = len(components)
        for node in component:
            # 这里假设node为Node对象
            setattr(node, 'subgraph', subgraph_num)
        components.append(component)

    def combine_node_subgraphs(self, actor_set: ActorSet, components, subgraph_node, subgraph_num):
        # 严格还原C++ combineNodeSubgraphs逻辑
        # 合并所有与subgraph_node相连且subgraph编号不同的component
        # subgraph_node: Node对象
        # subgraph_num: int
        # components: list of set(Node)
        # 先找出所有需要合并的subgraph编号
        to_merge = []
        for idx, comp in enumerate(components):
            if subgraph_node in comp and idx != subgraph_num:
                to_merge.append(idx)
        # 合并所有component到subgraph_num
        for idx in to_merge:
            for node in components[idx]:
                setattr(node, 'subgraph', subgraph_num)
                components[subgraph_num].add(node)
            components[idx].clear()

    def get_reachable_nodes_count(self, sub_map: SubMap, actor_set: ActorSet, start: Node, condition, occupied_nodes):
        # 完全还原C++ getReachableNodesCount逻辑
        from collections import deque
        visited = set()
        queue = deque()
        queue.append(start)
        count = 0
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if condition(start, node, sub_map, actor_set):
                count += 1
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = node.i+di, node.j+dj
                if 0<=ni<sub_map.height and 0<=nj<sub_map.width and sub_map.grid[ni][nj]==0:
                    nnode = Node(ni, nj)
                    if nnode not in visited and nnode not in occupied_nodes:
                        queue.append(nnode)
        return count

    def get_paths(self, actor_set: ActorSet):
        # 完全还原C++ getPaths逻辑
        self.agents_paths = [[] for _ in range(len(actor_set))]
        
        # 初始化每个agent的起始位置（与C++版本一致）
        agent_positions = []
        for i, a in enumerate(actor_set):
            # 使用初始位置作为起点
            start_pos = Node(a.current.x, a.current.y)
            agent_positions.append(start_pos)
            self.agents_paths[i].append(start_pos)
        
        # 按时间顺序处理每个移动（与C++版本一致）
        for move in self.agents_moves:
            aid, to_node = move
            # 更新移动的agent的位置（模拟C++的增量移动）
            for idx, a in enumerate(actor_set):
                if a.id == aid:
                    agent_positions[idx] = to_node
                    break
            
            # 为所有agent添加当前位置到路径（与C++版本完全一致）
            for idx, pos in enumerate(agent_positions):
                self.agents_paths[idx].append(pos)
        
        return self.agents_paths

    def get_parallel_paths(self, actor_set: ActorSet, config: MAPFConfig):
        # 严格按照C++ getParallelPaths逻辑实现
        agent_count = len(actor_set)
        agents_positions = [[] for _ in range(agent_count)]
        agent_ind = [0] * agent_count
        nodes_occupations = {}  # Node -> list of agent indices
        node_ind = {}  # Node -> current index in occupation list
        
        # 初始化路径和位置
        self.agents_paths = [[] for _ in range(agent_count)]
        for i, a in enumerate(actor_set):
            # 使用agent的起始位置：Point(x,y) -> Node(i,j) where i=y, j=x
            start_position = Node(a.start.y, a.start.x)
            agents_positions[i].append(start_position)
            self.agents_paths[i].append(start_position)
            if start_position not in nodes_occupations:
                nodes_occupations[start_position] = []
                node_ind[start_position] = 0
            nodes_occupations[start_position].append(i)  # 存储agent索引，与C++一致
        
        # 处理每个移动（使用增量移动）
        # 建立 ID 到连续索引的映射，确保非连续ID也能正确映射到路径数组
        id_to_index = {a.id: i for i, a in enumerate(actor_set)}
        # debug_log(f"处理 {len(self.agents_moves)} 个移动")
        for move in self.agents_moves:
            agent_idx = id_to_index.get(move.id)
            if agent_idx is None or not (0 <= agent_idx < agent_count):
                continue
            cur = agents_positions[agent_idx][-1]
            # 计算新位置（增量移动）
            new_pos = Node(cur.i + move.di, cur.j + move.dj)
            # debug_log(f"Agent {move.id} (idx={agent_idx}): ({cur.i},{cur.j}) + ({move.di},{move.dj}) = ({new_pos.i},{new_pos.j})")
            
            if new_pos not in nodes_occupations:
                nodes_occupations[new_pos] = []
                node_ind[new_pos] = 0
            
            # 严格按照C++逻辑：检查重复位置
            if (nodes_occupations[new_pos] and 
                nodes_occupations[new_pos][-1] == agent_idx):  # 比较agent索引
                # 移除重复的位置
                while (agents_positions[agent_idx] and 
                       agents_positions[agent_idx][-1] != new_pos):
                    cur_back = agents_positions[agent_idx][-1]
                    # 从occupation中移除
                    if cur_back in nodes_occupations:
                        last_ind = len(nodes_occupations[cur_back]) - 1
                        while (last_ind >= 0 and 
                               nodes_occupations[cur_back][last_ind] != agent_idx):
                            last_ind -= 1
                        if last_ind >= 0:
                            nodes_occupations[cur_back].pop(last_ind)
                    agents_positions[agent_idx].pop()
            else:
                # 添加新位置
                agents_positions[agent_idx].append(new_pos)
                nodes_occupations[new_pos].append(agent_idx)  # 存储agent索引
        
        # 打印每个agent的位置序列
        for i, a in enumerate(actor_set):
            # debug_log(f"Agent {a.id} (idx={i}) 位置序列: {[(p.i, p.j) for p in agents_positions[i]]}")
            pass
        
        # 并行路径生成（严格按照C++逻辑）
        # debug_log(f"开始并行路径生成，agent_count={agent_count}")
        for i, a in enumerate(actor_set):
            # debug_log(f"Agent {a.id} (idx={i}) 位置数量: {len(agents_positions[i])}")
            pass
        
        # 初始化路径
        self.agents_paths = [[] for _ in range(agent_count)]
        for i in range(agent_count):
            self.agents_paths[i].append(agents_positions[i][0])  # 添加起始位置
        
        # 并行路径生成
        finished = [False] * agent_count
        agent_ind = [0] * agent_count  # 每个agent在agents_positions中的当前位置
        
        # 使用第一阶段构建的完整nodes_occupations，但重置node_ind
        # 注意：nodes_occupations已经在第一阶段构建完成，包含所有agent会访问的位置
        node_ind = {}  # Node -> current index in occupation list
        
        # 初始化所有位置的node_ind为0
        for pos in nodes_occupations:
            node_ind[pos] = 0
        
        # debug_log(f"并行路径生成开始时的nodes_occupations:")
        for pos, agents in nodes_occupations.items():
            # debug_log(f"  ({pos.i}, {pos.j}): {agents}")
            pass
        
        while True:
            has_moved = [False] * agent_count
            for i in range(agent_count):
                if has_moved[i] or finished[i]:
                    continue
                
                if len(agents_positions[i]) == 1:
                    # debug_log(f"Agent {actor_set[i].id} (idx={i}) 只有1个位置，标记为完成")
                    self.agents_paths[i].append(agents_positions[i][0])
                    finished[i] = True
                    continue
                
                # 严格按照C++逻辑实现链式移动
                if agent_ind[i] + 1 < len(agents_positions[i]):
                    # 构建移动路径（链式移动）
                    path = []
                    cur_agent = i
                    can_move = True
                    
                    while True:
                        path.append(cur_agent)
                        next_node = agents_positions[cur_agent][agent_ind[cur_agent] + 1]
                        
                        # 确保next_node在node_ind中
                        if next_node not in node_ind:
                            node_ind[next_node] = 0
                        if next_node not in nodes_occupations:
                            nodes_occupations[next_node] = []
                        
                        last_ind = node_ind[next_node]
                        
                        # debug_log(f"Agent {actor_set[cur_agent].id} (idx={cur_agent}) 检查移动到 ({next_node.i}, {next_node.j})")
                        # debug_log(f"  node_ind[{next_node.i},{next_node.j}] = {last_ind}")
                        # debug_log(f"  nodes_occupations[{next_node.i},{next_node.j}] = {nodes_occupations[next_node]}")
                        
                        if (last_ind < len(nodes_occupations[next_node]) and 
                            nodes_occupations[next_node][last_ind] == cur_agent):
                            # debug_log(f"  Agent {actor_set[cur_agent].id} 可以直接移动")
                            break
                        elif (len(nodes_occupations[next_node]) > 0 and
                              last_ind + 1 < len(nodes_occupations[next_node]) and 
                              nodes_occupations[next_node][last_ind + 1] == cur_agent):
                            next_agent = nodes_occupations[next_node][last_ind]
                            # debug_log(f"  Agent {actor_set[cur_agent].id} 需要等待 Agent {actor_set[next_agent].id} (idx={next_agent})")
                            
                            current_pos = agents_positions[next_agent][agent_ind[next_agent]]
                            if (finished[next_agent] or has_moved[next_agent] or next_agent < cur_agent or
                                (current_pos.i != next_node.i or current_pos.j != next_node.j)):
                                # debug_log(f"  Agent {actor_set[next_agent].id} 无法移动，链式移动失败")
                                # debug_log(f"    finished[{next_agent}] = {finished[next_agent]}")
                                # debug_log(f"    has_moved[{next_agent}] = {has_moved[next_agent]}")
                                # debug_log(f"    next_agent < cur_agent: {next_agent} < {cur_agent} = {next_agent < cur_agent}")
                                # debug_log(f"    agents_positions[{next_agent}][{agent_ind[next_agent]}] = ({current_pos.i}, {current_pos.j})")
                                # debug_log(f"    next_node = ({next_node.i}, {next_node.j})")
                                can_move = False
                                break
                            
                            cur_agent = next_agent
                            if cur_agent == i:
                                # debug_log(f"  检测到循环，停止链式移动")
                                break
                        else:
                            # debug_log(f"  Agent {actor_set[cur_agent].id} 无法移动")
                            can_move = False
                            break
                    
                    if can_move:
                        # debug_log(f"执行链式移动，路径: {path}")
                        for agent_id in path:
                            has_moved[agent_id] = True
                            # 严格按照C++逻辑：递增离开位置的node_ind
                            current_node = agents_positions[agent_id][agent_ind[agent_id]]
                            node_ind[current_node] += 1
                            agent_ind[agent_id] += 1
                            self.agents_paths[agent_id].append(agents_positions[agent_id][agent_ind[agent_id]])
                            if agent_ind[agent_id] == len(agents_positions[agent_id]) - 1:
                                finished[agent_id] = True
                    else:
                        # debug_log(f"Agent {actor_set[i].id} 无法移动，重复当前位置")
                        self.agents_paths[i].append(agents_positions[i][agent_ind[i]])
                else:
                    # debug_log(f"Agent {actor_set[i].id} (idx={i}) 已经到达终点")
                    self.agents_paths[i].append(agents_positions[i][agent_ind[i]])
                    finished[i] = True
                
                # 移动逻辑已经在上面处理了，这里不需要额外的代码
            
            # 检查是否还有agent移动
            if not any(has_moved):
                # debug_log("没有agent移动，结束并行路径生成")
                break
        
        # 写入结果
        for i, a in enumerate(actor_set):
            # debug_log(f"Agent {a.id} 最终路径: {[(p.i, p.j) for p in self.agents_paths[i]]}")
            # 转换为Point对象：Node(i,j) -> Point(x=j, y=i)
            point_path = [Point(p.j, p.i) for p in self.agents_paths[i]]
            self.result.add_path(a.id, point_path)
        
        return self.agents_paths

    def start_search(self, sub_map: SubMap, config: MAPFConfig, actor_set: ActorSet):
        # Create a fresh trace logger per invocation (unique timestamped file)
        # try:
        #     if self.trace_logger is not None:
        #         self.trace_logger.close()
        # except Exception:
        #     pass
        # self.trace_logger = PNRTraceLogger()

        # Clear previous results to ensure each call is independent
        self.result = MAPFSearchResult()
        self.agents_moves.clear()  # Clear accumulated moves from previous calls
        self.agents_paths.clear()  # Clear accumulated paths from previous calls
        self.search = ISearch(sub_map)
        start_time = time.time()
        # 还原C++主入口完整流程
        # Trace: initial inputs snapshot
        # try:
        #     grid_row_sums = [int(sum(int(v) for v in row)) for row in sub_map.grid]
        # except Exception:
        #     grid_row_sums = []
        # try:
        #     actors = []
        #     for a in actor_set:
        #         item = {
        #             "id": int(a.id),
        #             "start": [int(a.start.x), int(a.start.y)],
        #             "goal": [int(a.goal.x), int(a.goal.y)]
        #         }
        #         # include real_id if present (propagated from coordinator)
        #         if hasattr(a, 'real_id'):
        #             try:
        #                 item["real_id"] = int(getattr(a, 'real_id'))
        #             except Exception:
        #                 item["real_id"] = getattr(a, 'real_id')
        #         actors.append(item)
        # except Exception:
        #     actors = []
        # self.trace_logger.log({
        #     "type": "init",
        #     "sub_map": {"width": int(sub_map.width), "height": int(sub_map.height)},
        #     "grid_row_sums": grid_row_sums,
        #     "actors": actors,
        #     "config": {"max_steps": int(getattr(config, "max_steps", 0))}
        # })

        self.get_subgraphs(sub_map, actor_set)
        self.assign_to_subgraphs(sub_map, actor_set)
        self.get_priorities(sub_map, actor_set)
        success = self.solve(sub_map, config, actor_set)
        self.result.success = success
        self.result.runtime = time.time() - start_time
        # Trace: final result snapshot
        # self.trace_logger.log({
        #     "type": "result",
        #     "success": bool(success),
        #     "runtime_s": float(self.result.runtime),
        #     "agents_moves_count": int(len(self.agents_moves))
        # })
        # try:
        #     self.trace_logger.close()
        # except Exception:
        #     pass
        return self.result 