import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum

class DeadlockTriggerType(Enum):
    """死锁触发类型"""
    SPEED_BUFFER = "speed_buffer"      # 速度缓冲区触发
    COMMON_POINT = "common_point"      # 共同点触发
    COLLISION_RISK = "collision_risk"  # 碰撞风险触发

class AgentDeadlockDetector:
    """
    单个Agent的死锁检测器
    每个RL agent都应该有自己的死锁检测器实例
    """
    
    def __init__(self, 
                 agent_id: int,
                 speed_buffer_size: int = 20,
                 small_speed_threshold: float = 0.1,
                 neighbor_speed_threshold: float = 0.1,
                  min_neighbors_for_deadlock: int = 2,
                 sight_radius: float = 3.0,
                   detection_interval: int = 5,
                   collision_risk_distance_threshold: float = 4.0,
                   activation_hysteresis_steps: int = 3):
        """
        初始化单个agent的死锁检测器
        
        Args:
            agent_id: agent的唯一标识
            speed_buffer_size: 速度历史缓冲区大小
            small_speed_threshold: 低速阈值
            neighbor_speed_threshold: 邻居低速阈值
            min_neighbors_for_deadlock: 触发死锁的最小邻居数
            sight_radius: 视野半径
            detection_interval: 检测间隔
        """
        self.agent_id = agent_id
        
        # 速度相关参数
        self.speed_buffer_size = speed_buffer_size
        self.small_speed_threshold = small_speed_threshold
        self.neighbor_speed_threshold = neighbor_speed_threshold
        
        # 邻居相关参数
        self.min_neighbors_for_deadlock = min_neighbors_for_deadlock
        self.sight_radius = sight_radius
        self.collision_risk_distance_threshold = collision_risk_distance_threshold
        
        # 检测控制
        self.detection_interval = detection_interval
        self.last_detection_step = 0
        # 激活迟滞：需要连续满足若干步才进入死锁（首次与再次激活都适用）
        self.activation_hysteresis_steps = max(1, activation_hysteresis_steps)
        self._satisfied_streak = 0
        self._last_satisfied_trigger_type = None
        
        # 状态缓冲区
        self.speed_buffer = deque(maxlen=speed_buffer_size)
        self.mean_speed = 0.0
        
        # 邻居信息
        self.neighbors = []  # [(distance, neighbor_agent), ...]
        
        # 死锁状态
        self.deadlock_detected = False
        self.deadlock_trigger_type = None
        self.in_deadlock_mode = False
        
        # 目标信息
        self.current_position = None
        self.goal_position = None
        self.distance_to_goal = float('inf')
        
    def update_state(self, 
                    current_position: np.ndarray,
                    current_velocity: np.ndarray,
                    goal_position: np.ndarray,
                    neighbor_agents: List[Tuple[float, 'AgentDeadlockDetector']],
                    timestep: int):
        """
        更新agent状态
        
        Args:
            current_position: 当前位置
            current_velocity: 当前速度
            goal_position: 目标位置
            neighbor_agents: 邻居agent列表 [(distance, detector), ...]
            timestep: 当前时间步
        """
        self.current_position = current_position
        self.goal_position = goal_position
        self.distance_to_goal = np.linalg.norm(current_position - goal_position)
        
        # 更新速度缓冲区
        current_speed = np.linalg.norm(current_velocity)
        self.speed_buffer.append(current_speed)
        
        # 计算平均速度 (使用Kahan求和算法提高数值稳定性)
        if len(self.speed_buffer) > 0:
            sum_speed = 0.0
            c = 0.0  # 补偿项
            for speed in self.speed_buffer:
                y = speed - c
                t = sum_speed + y
                c = (t - sum_speed) - y
                sum_speed = t
            self.mean_speed = sum_speed / len(self.speed_buffer)
        
        # 更新邻居信息
        self.neighbors = neighbor_agents
        
        # 存储当前速度用于碰撞风险检测
        self.current_velocity = current_velocity
        
    def detect_deadlock(self, timestep: int) -> Tuple[bool, Optional[DeadlockTriggerType]]:
        """
        检测死锁状态
        
        Args:
            timestep: 当前时间步
            
        Returns:
            Tuple[bool, Optional[DeadlockTriggerType]]: (是否检测到死锁, 触发类型)
        """
        # 检查检测间隔
        if timestep - self.last_detection_step < self.detection_interval:
            return self.deadlock_detected, self.deadlock_trigger_type
        
        self.last_detection_step = timestep
        
        # 综合判断触发类型（单步）
        triggered_type = None
        if self._check_speed_buffer_trigger():
            triggered_type = DeadlockTriggerType.SPEED_BUFFER
        elif self._check_common_point_trigger():
            triggered_type = DeadlockTriggerType.COMMON_POINT
        elif self._check_collision_risk_trigger():
            triggered_type = DeadlockTriggerType.COLLISION_RISK

        # 维护连续满足计数
        if triggered_type is not None:
            self._satisfied_streak += 1
            self._last_satisfied_trigger_type = triggered_type
        else:
            self._satisfied_streak = 0

        # 未在死锁中：仅当连续满足达到阈值才激活
        if not self.deadlock_detected:
            if self._satisfied_streak >= self.activation_hysteresis_steps:
                self.deadlock_detected = True
                self.deadlock_trigger_type = self._last_satisfied_trigger_type
                return True, self.deadlock_trigger_type
            else:
                self.deadlock_detected = False
                self.deadlock_trigger_type = None
                return False, None

        # 已在死锁中：若当步不再满足，立即解除；否则保持
        if triggered_type is None:
            self.deadlock_detected = False
            self.deadlock_trigger_type = None
            self._satisfied_streak = 0
            return False, None
        else:
            # 维持死锁状态
            return True, self.deadlock_trigger_type
    
    def _check_speed_buffer_trigger(self) -> bool:
        """
        速度缓冲区触发检测
        参考C++实现: SingleNeighbourMeanSpeedMAPFTrigger()
        """
        # 检查邻居数量是否达到阈值
        if len(self.neighbors) < self.min_neighbors_for_deadlock:
            return False
        
        # 检查自身速度是否低于阈值
        if self.mean_speed >= self.small_speed_threshold:
            return False
        
        # 检查邻居速度是否也低于阈值
        for distance, neighbor_detector in self.neighbors:
            if neighbor_detector.mean_speed >= self.neighbor_speed_threshold:
                return False
        
        return True
    
    def _check_common_point_trigger(self) -> bool:
        """
        共同点触发检测
        参考C++实现: CommonPointMAPFTrigger()
        """
        # 检查邻居数量是否达到阈值
        if len(self.neighbors) < self.min_neighbors_for_deadlock:
            return False
        
        # 检查到目标的距离是否在视野范围内
        if self.distance_to_goal >= self.sight_radius:
            return False
        
        # 检查速度是否足够低（真正的死锁应该是低速状态）
        if self.mean_speed >= self.small_speed_threshold:
            return False
        
        # 检查邻居速度是否也足够低
        for distance, neighbor_detector in self.neighbors:
            if neighbor_detector.mean_speed >= self.neighbor_speed_threshold:
                return False
        
        return True
    
    def _check_collision_risk_trigger(self) -> bool:
        """
        碰撞风险触发检测
        """
        # 检查邻居数量是否达到阈值
        if len(self.neighbors) < self.min_neighbors_for_deadlock:
            return False
        
        # 检查是否有非常近的邻居（碰撞风险）
        for distance, neighbor_detector in self.neighbors:
            if distance < self.collision_risk_distance_threshold:
                # 检查当前速度（高速碰撞风险）
                current_speed = np.linalg.norm(self.current_velocity) if hasattr(self, 'current_velocity') else self.mean_speed
                neighbor_speed = neighbor_detector.mean_speed
                
                # 如果两个智能体都在高速移动且距离很近，触发死锁检测
                if current_speed > 0.5 and neighbor_speed > 0.5:
                    return True
                
                # 或者如果速度很低且距离很近（传统死锁）
                if self.mean_speed < self.small_speed_threshold and neighbor_detector.mean_speed < self.neighbor_speed_threshold:
                    return True
        
        return False
    
    def get_deadlock_info(self) -> Dict:
        """
        获取死锁检测信息
        """
        return {
            'agent_id': self.agent_id,
            'deadlock_detected': self.deadlock_detected,
            'trigger_type': self.deadlock_trigger_type.value if self.deadlock_trigger_type else None,
            'mean_speed': self.mean_speed,
            'neighbor_count': len(self.neighbors),
            'distance_to_goal': self.distance_to_goal,
            'speed_buffer_size': len(self.speed_buffer)
        }
    
    def reset(self):
        """
        重置检测器状态
        """
        self.deadlock_detected = False
        self.deadlock_trigger_type = None
        self.in_deadlock_mode = False
        self.speed_buffer.clear()
        self.mean_speed = 0.0
        self.neighbors.clear()
        self.last_detection_step = 0
        self._satisfied_streak = 0
        self._last_satisfied_trigger_type = None

class DistributedDeadlockManager:
    """
    分布式死锁管理器
    管理所有agent的死锁检测器，协调死锁解决
    """
    
    def __init__(self, 
                 num_agents: int,
                 config: Dict = None):
        """
        初始化分布式死锁管理器
        
        Args:
            num_agents: agent数量
            config: 配置参数
        """
        self.num_agents = num_agents
        
        # 默认配置
        default_config = {
            'speed_buffer_size': 20,
            'small_speed_threshold': 0.1,
            'neighbor_speed_threshold': 0.1,
            'min_neighbors_for_deadlock': 2,
            'sight_radius': 3.0,
            'detection_interval': 5,
            'activation_hysteresis_steps': 3
        }
        
        if config:
            default_config.update(config)
        
        # 为每个agent创建死锁检测器
        self.agent_detectors = {}
        for i in range(num_agents):
            self.agent_detectors[i] = AgentDeadlockDetector(
                agent_id=i,
                **default_config
            )
        
        # 死锁状态
        self.deadlock_agents = set()
        self.deadlock_groups = []  # 死锁组列表
        
    def update_all_agents(self, 
                         agent_positions: List[np.ndarray],
                         agent_velocities: List[np.ndarray],
                         agent_goals: List[np.ndarray],
                         timestep: int):
        """
        更新所有agent的状态
        
        Args:
            agent_positions: 所有agent的位置列表
            agent_velocities: 所有agent的速度列表
            agent_goals: 所有agent的目标列表
            timestep: 当前时间步
        """
        # 计算邻居关系
        neighbor_relations = self._calculate_neighbors(agent_positions)
        
        # 更新每个agent的检测器
        for i in range(self.num_agents):
            neighbors = [(dist, self.agent_detectors[j]) 
                        for dist, j in neighbor_relations[i]]
            
            self.agent_detectors[i].update_state(
                current_position=agent_positions[i],
                current_velocity=agent_velocities[i],
                goal_position=agent_goals[i],
                neighbor_agents=neighbors,
                timestep=timestep
            )
    
    def detect_deadlocks(self, timestep: int) -> Tuple[Set[int], List[Set[int]]]:
        """
        检测所有agent的死锁状态
        
        Args:
            timestep: 当前时间步
            
        Returns:
            Tuple[Set[int], List[Set[int]]]: (死锁agent集合, 死锁组列表)
        """
        deadlock_agents = set()
        
        # 检测每个agent的死锁状态
        for i in range(self.num_agents):
            deadlock_detected, trigger_type = self.agent_detectors[i].detect_deadlock(timestep)
            if deadlock_detected:
                deadlock_agents.add(i)
                # 额外：将风险半径内的邻居也加入死锁集合（便于二者一起进入MAPF）
                risk_r = getattr(self.agent_detectors[i], 'collision_risk_distance_threshold', 1.0)
                for dist, nei_det in self.agent_detectors[i].neighbors:
                    if dist <= risk_r:
                        deadlock_agents.add(nei_det.agent_id)
        
        # 识别死锁组（相互影响的agent组）
        deadlock_groups = self._identify_deadlock_groups(deadlock_agents)
        
        self.deadlock_agents = deadlock_agents
        self.deadlock_groups = deadlock_groups
        
        return deadlock_agents, deadlock_groups
    
    def _calculate_neighbors(self, agent_positions: List[np.ndarray]) -> List[List[Tuple[float, int]]]:
        """
        计算所有agent的邻居关系
        
        Args:
            agent_positions: agent位置列表
            
        Returns:
            List[List[Tuple[float, int]]]: 每个agent的邻居列表 [(distance, neighbor_id), ...]
        """
        neighbor_relations = [[] for _ in range(self.num_agents)]
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    if distance <= self.agent_detectors[i].sight_radius:
                        neighbor_relations[i].append((distance, j))
            
            # 按距离排序
            neighbor_relations[i].sort(key=lambda x: x[0])
        
        return neighbor_relations
    
    def _identify_deadlock_groups(self, deadlock_agents: Set[int]) -> List[Set[int]]:
        """
        识别死锁组（相互影响的agent组）
        
        Args:
            deadlock_agents: 死锁agent集合
            
        Returns:
            List[Set[int]]: 死锁组列表
        """
        if not deadlock_agents:
            return []
        
        # 使用并查集算法识别连通分量
        parent = {i: i for i in deadlock_agents}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # 检查死锁agent之间的连通性
        for i in deadlock_agents:
            for j in deadlock_agents:
                if i != j:
                    # 如果两个agent是邻居，则合并（注意 neighbors 中存的是 detector 对象而非ID）
                    neighbor_ids = [neighbor_detector.agent_id for _, neighbor_detector in self.agent_detectors[i].neighbors]
                    if j in neighbor_ids:
                        union(i, j)
        
        # 收集连通分量
        groups = defaultdict(set)
        for agent_id in deadlock_agents:
            root = find(agent_id)
            groups[root].add(agent_id)
        
        return list(groups.values())
    
    def get_deadlock_summary(self) -> Dict:
        """
        获取死锁检测摘要信息
        """
        return {
            'total_agents': self.num_agents,
            'deadlock_agents': list(self.deadlock_agents),
            'deadlock_groups': [list(group) for group in self.deadlock_groups],
            'agent_details': {
                i: detector.get_deadlock_info() 
                for i, detector in self.agent_detectors.items()
            }
        }
    
    def reset(self):
        """
        重置所有检测器
        """
        for detector in self.agent_detectors.values():
            detector.reset()
        self.deadlock_agents.clear()
        self.deadlock_groups.clear()
