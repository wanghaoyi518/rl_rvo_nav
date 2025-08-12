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
            self._rollback()
            return False

        builder = SubMapBuilder(self.env_adapter)
        self.bbox_world = builder.compute_bbox(list(group_positions.values()))
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
                self._rollback()
                return False
            self.start_nodes[aid] = start_node
            occupied.add((start_node.i, start_node.j))

        # 步骤6：为每个 Agent 选择终点（waypoint→可达离散节点）
        occupied_goals: Set[Tuple[int, int]] = set(occupied)
        for aid in self.group_agent_ids:
            waypoints = self.waypoints_dict.get(aid, [])
            target_point, skipped = selector.get_goal_point_for_mapf(waypoints, self.bbox_world)
            self.buff_skipped_waypoints[aid] = skipped
            goal_node = selector.find_accessible_node_for_goal(self.sub_map, target_point, occupied_goals)
            if goal_node is None:
                self._rollback()
                return False
            self.goal_nodes[aid] = goal_node
            occupied_goals.add((goal_node.i, goal_node.j))

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
            self._rollback()
            return False

        # 步骤10：处理结果（提取路径）
        self.paths_grid = {}
        for aid in self.group_agent_ids:
            actor_id = self.actor_id_map[aid]
            path_points = result.paths.get(actor_id)
            if not path_points or len(path_points) < 1:
                self._rollback()
                return False
            self.paths_grid[aid] = path_points

        # 步骤11：准备执行
        self.executor = MAPFExecutor(builder)
        self.executor.prepare(self.paths_grid)

        # 步骤1/2：提交切换（原子提交）
        self._commit_switch()
        self.in_mapf_mode = True
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


