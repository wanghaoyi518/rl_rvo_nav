from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from .actor import Actor
from .actor_set import ActorSet
from .mapf_config import MAPFConfig
from .push_and_rotate import PushAndRotate
from .sub_map import SubMap
from .node import Node, Point
from .mapf_search_result import MAPFSearchResult

from .submap_builder import SubMapBuilder
from .start_goal_selector import StartGoalSelector
from .executor import MAPFExecutor


class MAPFSession:
    """
    单个死锁组的一次 MAPF 事务：准备→构图→起终点→求解→执行准备→执行/完成或回滚。
    """

    def __init__(
        self,
        session_id: int,
        group_agent_ids: Set[int],
        env_adapter: Any,
        waypoints_dict: Dict[int, List[np.ndarray]],
    ) -> None:
        self.session_id = session_id
        self.group_agent_ids = set(group_agent_ids)
        self.env_adapter = env_adapter
        self.waypoints_dict = waypoints_dict

        # runtime states
        self.in_mapf_mode: bool = False
        self.is_finished: bool = False

        # building blocks
        self.sub_map: Optional[SubMap] = None
        self.origin_ij: Optional[Tuple[int, int]] = None
        self.resolution: float = 1.0
        self.bbox_world: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

        self.actor_set: Optional[ActorSet] = None
        self.actor_id_map: Dict[int, int] = {}  # agent_id -> actor_id (0..n-1)

        self.start_nodes: Dict[int, Node] = {}
        self.goal_nodes: Dict[int, Node] = {}
        self.paths_grid: Dict[int, List[Point]] = {}
        self.buff_skipped_waypoints: Dict[int, List[np.ndarray]] = {}

        self.executor: Optional[MAPFExecutor] = None

        # snapshots for rollback (external integration placeholders)
        self._snapshot_agent_modes: Dict[int, bool] = {}

    # ============ 主流程 ============
    def prepare_and_solve(self, agent_states: Dict[str, Dict[int, np.ndarray]], time_step: int) -> bool:
        """执行完整 setup + 求解 + 执行准备，成功则进入执行态。"""
        positions: Dict[int, np.ndarray] = agent_states.get("positions", {})
        goals: Dict[int, np.ndarray] = agent_states.get("goals", {})

        # 步骤1：触发准备（快照 + 标记待提交）
        self._snapshot_before_switch()

        # 步骤3-4：构建子图（bbox + SubMap）
        group_positions = {aid: positions[aid] for aid in self.group_agent_ids if aid in positions}
        if len(group_positions) != len(self.group_agent_ids):
            print(f"[MAPFSession] prepare failed: missing positions for agents {sorted(list(self.group_agent_ids - set(group_positions.keys())))} at t={time_step}")
            self._rollback()
            return False

        builder = SubMapBuilder(self.env_adapter)
        # 使用整数尺寸的BBox计算，确保网格大小一致且便于计算
        margin = 2.0  # meters
        resolution = 1.0  # meters/cell
        
        # 计算整数尺寸的BBox
        self.bbox_world = builder.compute_bbox_with_integer_dimensions(
            list(group_positions.values()), 
            margin_m=margin, 
            resolution=resolution
        )
        
        # 记录BBox计算过程
        base_bbox = builder.compute_bbox(list(group_positions.values()))
        (base_min_x, base_min_y), (base_max_x, base_max_y) = base_bbox
        (final_min_x, final_min_y), (final_max_x, final_max_y) = self.bbox_world
        
        print(f"[MAPFSession] BBox calculation: base=(({base_min_x:.3f}, {base_min_y:.3f}), ({base_max_x:.3f}, {base_max_y:.3f})), "
              f"final=(({final_min_x:.3f}, {final_min_y:.3f}), ({final_max_x:.3f}, {final_max_y:.3f}))")
        
        grid, origin_ij, resolution = builder.build_submap(self.bbox_world)
        self.sub_map = SubMap(grid, origin_ij[0], origin_ij[1])
        self.origin_ij = origin_ij
        self.resolution = resolution

        # 步骤5：为每个 Agent 选择起点
        selector = StartGoalSelector(builder)
        occupied: Set[Tuple[int, int]] = set()
        for aid in self.group_agent_ids:
            start_node = selector.find_close_available_node(self.sub_map, positions[aid], occupied)
            if start_node is None:
                pw = positions[aid]
                print(f"[MAPFSession] prepare failed: no start node for agent {aid} at pos {pw.tolist()} within submap at t={time_step}")
                self._rollback()
                return False
            self.start_nodes[aid] = start_node
            occupied.add((start_node.i, start_node.j))

        # 步骤6：为每个 Agent 选择终点（waypoint→可达离散节点）
        occupied_goals: Set[Tuple[int, int]] = set(occupied)
        for aid in self.group_agent_ids:
            waypoints = self.waypoints_dict.get(aid, [])
            # 使用改进的目标点选择，传入当前位置信息
            current_pos = positions.get(aid)
            target_point, skipped = selector.get_goal_point_for_mapf(waypoints, self.bbox_world, current_pos)
            self.buff_skipped_waypoints[aid] = skipped
            
            # 使用改进的可达节点搜索
            chosen_goal = selector.find_accessible_node_for_goal(self.sub_map, target_point, occupied_goals, max_search_radius=3)
            
            if chosen_goal is None:
                print(f"[MAPFSession] prepare failed: no goal node for agent {aid} target {target_point.tolist()} within submap bbox={self.bbox_world} at t={time_step}")
                self._rollback()
                return False
            
            self.goal_nodes[aid] = chosen_goal
            occupied_goals.add((chosen_goal.i, chosen_goal.j))

        # 步骤7：构建 ActorSet
        self.actor_set = ActorSet()
        self.actor_id_map.clear()
        # 确保 actor.id 从 0..n-1 连续
        for local_idx, aid in enumerate(sorted(self.group_agent_ids)):
            s_node = self.start_nodes[aid]
            g_node = self.goal_nodes[aid]
            actor = Actor(local_idx, Point(s_node.i, s_node.j), Point(g_node.i, g_node.j))
            self.actor_set.add_actor(actor)
            self.actor_id_map[aid] = local_idx

        # 步骤8-9：配置 + 求解（只用 PNR）
        config = MAPFConfig()
        solver = PushAndRotate()
        result: MAPFSearchResult = solver.start_search(self.sub_map, config, self.actor_set)
        if not result.success:
            print(f"[MAPFSession] prepare failed: PNR solve failed for session {self.session_id} at t={time_step}, group={sorted(list(self.group_agent_ids))}")
            self._rollback()
            return False

        # 步骤10：处理结果（提取路径）
        self.paths_grid = {}
        for aid in self.group_agent_ids:
            actor_id = self.actor_id_map[aid]
            path_points = result.paths.get(actor_id)
            if not path_points or len(path_points) < 1:
                print(f"[MAPFSession] prepare failed: empty path for agent {aid} (actor {actor_id}) in session {self.session_id} at t={time_step}")
                self._rollback()
                return False
            self.paths_grid[aid] = path_points

        # 步骤11：准备执行
        self.executor = MAPFExecutor(builder)
        self.executor.prepare(self.paths_grid)

        # 步骤1/2：提交切换（原子提交）
        self._commit_switch()
        self.in_mapf_mode = True
        
        # 输出详细的初始化信息到日志
        self._log_initialization_info(time_step)
        
        print(f"[MAPFSession] prepared and entered MAPF mode: session_id={self.session_id}, group={sorted(list(self.group_agent_ids))}, bbox={self.bbox_world}, grid=({self.sub_map.height}x{self.sub_map.width}) at t={time_step}")
        return True

    def step(self, current_positions: Dict[int, np.ndarray], dt: float) -> None:
        if self.is_finished or not self.in_mapf_mode:
            return
        assert self.executor is not None
        self.executor.step(current_positions, dt)
        if self.executor.is_finished():
            self.is_finished = True
            self.in_mapf_mode = False

    def next_target(self, agent_id: int) -> Optional[np.ndarray]:
        if self.executor is None:
            return None
        return self.executor.next_target(agent_id)

    def cancel(self) -> None:
        self._rollback()

    # ============ 内部：提交/回滚占位（与外部集成处） ============
    def _snapshot_before_switch(self) -> None:
        # 在集成你的 agent 容器时，记录各 agent 的 inMAPFMode 等外部状态。
        for aid in self.group_agent_ids:
            self._snapshot_agent_modes[aid] = False  # 占位：外层应读实际值

    def _commit_switch(self) -> None:
        # 在集成你的 agent 容器时，把组内 agent 的状态切换到 MAPF 模式。
        # 这里保留占位，不直接操作外部状态。
        pass

    def _rollback(self) -> None:
        # 失败回滚：恢复外部状态（占位），清理内部构件。
        self.in_mapf_mode = False
        self.is_finished = False
        self.sub_map = None
        self.origin_ij = None
        self.actor_set = None
        self.actor_id_map.clear()
        self.start_nodes.clear()
        self.goal_nodes.clear()
        self.paths_grid.clear()
        self.buff_skipped_waypoints.clear()
        self.executor = None

    def _log_initialization_info(self, time_step: int) -> None:
        """记录MAPF会话的详细初始化信息"""
        try:
            import os
            from datetime import datetime
            
            # 创建日志目录
            log_dir = "deadlock_initialization_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"mapf_init_session_{self.session_id}_t{time_step}_{timestamp}.txt"
            log_path = os.path.join(log_dir, log_filename)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("MAPF Session Initialization Log\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本信息
                f.write(f"Session ID: {self.session_id}\n")
                f.write(f"Timestep: {time_step}\n")
                f.write(f"Group Agent IDs: {sorted(list(self.group_agent_ids))}\n")
                f.write(f"Session Status: {'Active' if self.in_mapf_mode else 'Inactive'}\n\n")
                
                # BBox信息
                f.write("BBox Information:\n")
                f.write("-" * 20 + "\n")
                if self.bbox_world:
                    (min_x, min_y), (max_x, max_y) = self.bbox_world
                    width = max_x - min_x
                    height = max_y - min_y
                    area = width * height
                    
                    f.write(f"BBox World: (({min_x:.3f}, {min_y:.3f}), ({max_x:.3f}, {max_y:.3f}))\n")
                    f.write(f"BBox Width: {width:.3f} meters\n")
                    f.write(f"BBox Height: {height:.3f} meters\n")
                    f.write(f"BBox Area: {area:.3f} square meters\n")
                    
                    # 检查是否为整数尺寸
                    width_cells = width / self.resolution
                    height_cells = height / self.resolution
                    is_integer_width = abs(width_cells - round(width_cells)) < 1e-6
                    is_integer_height = abs(height_cells - round(height_cells)) < 1e-6
                    
                    f.write(f"BBox Width in Cells: {width_cells:.1f} ({'INTEGER' if is_integer_width else 'NON-INTEGER'})\n")
                    f.write(f"BBox Height in Cells: {height_cells:.1f} ({'INTEGER' if is_integer_height else 'NON-INTEGER'})\n")
                    f.write(f"Grid Resolution: {self.resolution:.3f} meters/cell\n")
                    
                    if is_integer_width and is_integer_height:
                        f.write(f"✓ BBox has integer dimensions: {int(round(width_cells))}x{int(round(height_cells))} cells\n")
                    else:
                        f.write(f"⚠ BBox has non-integer dimensions\n")
                else:
                    f.write("BBox: Not set\n")
                f.write("\n")
                
                # 子图信息
                f.write("SubMap Information:\n")
                f.write("-" * 20 + "\n")
                if self.sub_map:
                    f.write(f"Grid Size: {self.sub_map.height} x {self.sub_map.width} cells\n")
                    f.write(f"Resolution: {self.resolution:.3f} meters/cell\n")
                    f.write(f"Origin Grid: {self.origin_ij}\n")
                    f.write(f"Physical Size: {self.sub_map.height * self.resolution:.3f} x {self.sub_map.width * self.resolution:.3f} meters\n")
                    
                    # 添加详细的栅格地图信息
                    f.write(f"\nGrid Map Details:\n")
                    f.write(f"  Total Cells: {self.sub_map.height * self.sub_map.width}\n")
                    
                    # 统计障碍物和可通行区域
                    obstacle_count = 0
                    free_count = 0
                    for i in range(self.sub_map.height):
                        for j in range(self.sub_map.width):
                            if self.sub_map.grid[i][j] == 1:  # 障碍物
                                obstacle_count += 1
                            else:  # 可通行
                                free_count += 1
                    
                    f.write(f"  Free Cells: {free_count} ({free_count/(self.sub_map.height * self.sub_map.width)*100:.1f}%)\n")
                    f.write(f"  Obstacle Cells: {obstacle_count} ({obstacle_count/(self.sub_map.height * self.sub_map.width)*100:.1f}%)\n")
                    
                    # 显示栅格地图的可视化表示
                    f.write(f"\nGrid Map Visualization (0=free, 1=obstacle):\n")
                    f.write(f"  Origin: Top-left corner\n")
                    f.write(f"  Format: [row][column] = value\n")
                    
                    # 为了可读性，限制显示大小
                    max_display_size = 20
                    display_height = min(self.sub_map.height, max_display_size)
                    display_width = min(self.sub_map.width, max_display_size)
                    
                    if self.sub_map.height <= max_display_size and self.sub_map.width <= max_display_size:
                        # 显示完整地图
                        f.write(f"  Full Grid Map:\n")
                        for i in range(self.sub_map.height):
                            row_str = "    "
                            for j in range(self.sub_map.width):
                                row_str += f"{self.sub_map.grid[i][j]} "
                            f.write(f"{row_str}\n")
                    else:
                        # 显示部分地图（左上角）
                        f.write(f"  Partial Grid Map (Top-left {display_height}x{display_width}):\n")
                        for i in range(display_height):
                            row_str = "    "
                            for j in range(display_width):
                                row_str += f"{self.sub_map.grid[i][j]} "
                            f.write(f"{row_str}\n")
                        f.write(f"  ... (truncated, full size: {self.sub_map.height}x{self.sub_map.width})\n")
                    
                    # 智能体位置在地图中的标记
                    f.write(f"\nAgent Positions in Grid:\n")
                    for aid in sorted(self.group_agent_ids):
                        if aid in self.start_nodes:
                            start_node = self.start_nodes[aid]
                            # 转换为局部坐标
                            local_i = start_node.i - self.origin_ij[0]
                            local_j = start_node.j - self.origin_ij[1]
                            if 0 <= local_i < self.sub_map.height and 0 <= local_j < self.sub_map.width:
                                cell_value = self.sub_map.grid[local_i][local_j]
                                f.write(f"  Agent {aid} Start: Local({local_i}, {local_j}) = {cell_value} {'(OBSTACLE!)' if cell_value == 1 else '(FREE)'}\n")
                            else:
                                f.write(f"  Agent {aid} Start: Local({local_i}, {local_j}) = OUT_OF_BOUNDS\n")
                        
                        if aid in self.goal_nodes:
                            goal_node = self.goal_nodes[aid]
                            # 转换为局部坐标
                            local_i = goal_node.i - self.origin_ij[0]
                            local_j = goal_node.j - self.origin_ij[1]
                            if 0 <= local_i < self.sub_map.height and 0 <= local_j < self.sub_map.width:
                                cell_value = self.sub_map.grid[local_i][local_j]
                                f.write(f"  Agent {aid} Goal: Local({local_i}, {local_j}) = {cell_value} {'(OBSTACLE!)' if cell_value == 1 else '(FREE)'}\n")
                            else:
                                f.write(f"  Agent {aid} Goal: Local({local_i}, {local_j}) = OUT_OF_BOUNDS\n")
                    
                    # 路径在地图中的标记
                    f.write(f"\nPath Analysis in Grid:\n")
                    for aid in sorted(self.group_agent_ids):
                        if aid in self.paths_grid:
                            path = self.paths_grid[aid]
                            f.write(f"  Agent {aid} Path:\n")
                            
                            # 检查路径上的每个点
                            path_obstacles = 0
                            path_free = 0
                            path_out_of_bounds = 0
                            
                            for step, point in enumerate(path):
                                # 转换为局部坐标
                                # 检查point类型，Point对象有i,j属性，其他可能有x,y属性
                                if hasattr(point, 'i') and hasattr(point, 'j'):
                                    local_i = point.i - self.origin_ij[0]
                                    local_j = point.j - self.origin_ij[1]
                                elif hasattr(point, 'x') and hasattr(point, 'y'):
                                    # 如果是世界坐标，需要转换到网格坐标
                                    world_i, world_j = self.builder.world_to_grid(point.x, point.y)
                                    local_i = world_i - self.origin_ij[0]
                                    local_j = world_j - self.origin_ij[1]
                                else:
                                    # 未知类型，跳过
                                    path_out_of_bounds += 1
                                    continue
                                
                                if 0 <= local_i < self.sub_map.height and 0 <= local_j < self.sub_map.width:
                                    cell_value = self.sub_map.grid[local_i][local_j]
                                    if cell_value == 1:
                                        path_obstacles += 1
                                        f.write(f"    Step {step}: Local({local_i}, {local_j}) = {cell_value} (OBSTACLE!)\n")
                                    else:
                                        path_free += 1
                                        if step < 3 or step >= len(path) - 3:  # 只显示前3步和后3步
                                            f.write(f"    Step {step}: Local({local_i}, {local_j}) = {cell_value} (FREE)\n")
                                        elif step == 3:
                                            f.write(f"    ... (middle steps)\n")
                                else:
                                    path_out_of_bounds += 1
                                    f.write(f"    Step {step}: Local({local_i}, {local_j}) = OUT_OF_BOUNDS\n")
                            
                            f.write(f"    Path Summary: {path_free} free, {path_obstacles} obstacles, {path_out_of_bounds} out_of_bounds\n")
                    
                    # 连通性分析
                    f.write(f"\nConnectivity Analysis:\n")
                    # 检查起点到目标点的连通性
                    for aid in sorted(self.group_agent_ids):
                        if aid in self.start_nodes and aid in self.goal_nodes:
                            start_node = self.start_nodes[aid]
                            goal_node = self.goal_nodes[aid]
                            
                            # 检查起点和终点的可通行性
                            start_traversable = self.sub_map.is_traversable(start_node.i, start_node.j)
                            goal_traversable = self.sub_map.is_traversable(goal_node.i, goal_node.j)
                            
                            f.write(f"  Agent {aid}: Start traversable = {start_traversable}, Goal traversable = {goal_traversable}\n")
                            
                            if not start_traversable:
                                f.write(f"    WARNING: Agent {aid} start position is not traversable!\n")
                            if not goal_traversable:
                                f.write(f"    WARNING: Agent {aid} goal position is not traversable!\n")
                else:
                    f.write("SubMap: Not created\n")
                f.write("\n")
                
                # 起点信息
                f.write("Start Nodes Information:\n")
                f.write("-" * 25 + "\n")
                for aid in sorted(self.group_agent_ids):
                    if aid in self.start_nodes:
                        start_node = self.start_nodes[aid]
                        start_world = self._grid_to_world(start_node.i, start_node.j)
                        f.write(f"Agent {aid}: Grid({start_node.i}, {start_node.j}) -> World({start_world[0]:.3f}, {start_world[1]:.3f})\n")
                    else:
                        f.write(f"Agent {aid}: Start node not set\n")
                f.write("\n")
                
                # 目标点信息
                f.write("Goal Nodes Information:\n")
                f.write("-" * 25 + "\n")
                for aid in sorted(self.group_agent_ids):
                    if aid in self.goal_nodes:
                        goal_node = self.goal_nodes[aid]
                        goal_world = self._grid_to_world(goal_node.i, goal_node.j)
                        f.write(f"Agent {aid}: Grid({goal_node.i}, {goal_node.j}) -> World({goal_world[0]:.3f}, {goal_world[1]:.3f})\n")
                        
                        # 显示waypoint信息
                        if aid in self.waypoints_dict:
                            waypoints = self.waypoints_dict[aid]
                            f.write(f"  Waypoints: {[f'({wp[0]:.3f}, {wp[1]:.3f})' for wp in waypoints]}\n")
                            
                            # 计算目标点与waypoint的距离
                            if waypoints:
                                target_point = np.array([goal_world[0], goal_world[1]])
                                distances = [np.linalg.norm(target_point - wp) for wp in waypoints]
                                min_dist_idx = np.argmin(distances)
                                f.write(f"  Closest Waypoint: {min_dist_idx} ({waypoints[min_dist_idx][0]:.3f}, {waypoints[min_dist_idx][1]:.3f})\n")
                                f.write(f"  Distance to Closest Waypoint: {distances[min_dist_idx]:.3f} meters\n")
                    else:
                        f.write(f"Agent {aid}: Goal node not set\n")
                f.write("\n")
                
                # 路径信息
                f.write("Path Information:\n")
                f.write("-" * 15 + "\n")
                for aid in sorted(self.group_agent_ids):
                    if aid in self.paths_grid:
                        path = self.paths_grid[aid]
                        f.write(f"Agent {aid}: Path length = {len(path)} steps\n")
                        if path:
                            start_world = self._grid_to_world(path[0].i, path[0].j)
                            end_world = self._grid_to_world(path[-1].i, path[-1].j)
                            f.write(f"  Path Start: World({start_world[0]:.3f}, {start_world[1]:.3f})\n")
                            f.write(f"  Path End: World({end_world[0]:.3f}, {end_world[1]:.3f})\n")
                            f.write(f"  Path Distance: {np.linalg.norm(np.array(end_world) - np.array(start_world)):.3f} meters\n")
                    else:
                        f.write(f"Agent {aid}: Path not generated\n")
                f.write("\n")
                
                # Actor信息
                f.write("Actor Information:\n")
                f.write("-" * 18 + "\n")
                if self.actor_set:
                    f.write(f"Total Actors: {len(self.actor_set.actors)}\n")
                    for actor in self.actor_set.actors:
                        f.write(f"Actor {actor.id}: Start({actor.start.x}, {actor.start.y}) -> Goal({actor.goal.x}, {actor.goal.y})\n")
                else:
                    f.write("Actor Set: Not created\n")
                f.write("\n")
                
                # 执行器信息
                f.write("Executor Information:\n")
                f.write("-" * 20 + "\n")
                if self.executor:
                    f.write("Executor: Prepared and ready\n")
                else:
                    f.write("Executor: Not prepared\n")
                f.write("\n")
                
                # 时间戳
                f.write(f"Log Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            print(f"[MAPFSession] Initialization log saved to: {log_path}")
            
        except Exception as e:
            print(f"[MAPFSession] Failed to save initialization log: {e}")

    def _grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """将网格坐标转换为世界坐标"""
        if self.origin_ij is None or self.resolution is None:
            return (0.0, 0.0)
        
        origin_i, origin_j = self.origin_ij
        world_x = (origin_j + j + 0.5) * self.resolution
        world_y = (origin_i + i + 0.5) * self.resolution
        return (world_x, world_y)


