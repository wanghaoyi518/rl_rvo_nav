import torch
import numpy as np
from pathlib import Path
import platform
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac
from math import pi, sin, cos, sqrt
import time 
from datetime import datetime
from rl_rvo_nav.agent_deadlock_detector import DistributedDeadlockManager
from rl_rvo_nav.python_pnr import MAPFManager, SimpleEnvAdapter

class post_train:
    def __init__(self, env, num_episodes=100, max_ep_len=150, acceler_vel = 1.0, reset_mode=3, render=True, save=False, neighbor_region=4, neighbor_num=5, args=None, **kwargs):

        self.env = env
        self.num_episodes=num_episodes
        self.max_ep_len = max_ep_len
        self.acceler_vel = acceler_vel
        self.reset_mode = reset_mode
        self.render=render
        self.save=save
        self.robot_number = self.env.ir_gym.robot_number
        self.step_time = self.env.ir_gym.step_time

        self.inf_print = kwargs.get('inf_print', True)
        self.std_factor = kwargs.get('std_factor', 0.001)
        # self.show_traj = kwargs.get('show_traj', False)
        self.show_traj = False
        self.traj_type = ''
        self.figure_format = kwargs.get('figure_format', 'png')

        self.nr = neighbor_region
        self.nm = neighbor_num
        self.args = args
        
        # 可选：多段 waypoint 支持（不影响默认流程）
        self.waypoint_sequences = kwargs.get('waypoint_sequences', None)
        self.waypoint_goal_threshold = float(kwargs.get('waypoint_goal_threshold', 0.2))
        self._waypoint_index = {}

        # 初始化碰撞记录
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collision_log_filename = f"collision_neighbor_{current_time}.txt"
        self.collision_episodes = []
        
        # 初始化失败episode历史记录
        self.failure_history = []  # 存储失败episode的历史信息
        self.history_length = 30   # 记录失败前30个timestep
        self.current_episode_history = []  # 当前episode的历史记录
        
        # 初始化VO Flag简化日志
        self.vo_flag_log_filename = f"vo_flag_log_{current_time}.txt"
        self.vo_flag_episodes = []  # 存储VO Flag信息
        # MAPF 让行冷却（避免帧间抖动）：agent_id -> remaining steps
        self._yield_cooldown = {}
        # 死锁解除后的短时安全护栏步数与目标对象
        self._post_release_guard = 0
        self._post_release_agents = []
        self._post_release_safe_streak = 0
        # 最近一次 MAPF 会话涉及的 agent 集合（用于解除后仅对相关体施加护栏）
        self._last_mapf_group = set()
        
        # 初始化分布式死锁检测器 - 优化参数以提高响应速度
        deadlock_config = {
            'speed_buffer_size': kwargs.get('deadlock_speed_buffer_size', 15),  # 减少缓冲区大小，提高响应速度
            'small_speed_threshold': kwargs.get('deadlock_speed_threshold', 0.05),
            'neighbor_speed_threshold': kwargs.get('deadlock_neighbor_speed_threshold', 0.05),
            'min_neighbors_for_deadlock': kwargs.get('min_agents_for_deadlock', 1),
            'sight_radius': kwargs.get('deadlock_distance_threshold', 3.0),  # 保持与RL训练时一致的视野半径
            'detection_interval': kwargs.get('deadlock_detection_interval', 1),  # 每步都检测
            'activation_hysteresis_steps': 3  # 减少激活延迟，从10步降到3步，提高响应速度
        }
        
        self.deadlock_manager = DistributedDeadlockManager(
            num_agents=self.robot_number,
            config=deadlock_config
        )
        # 优化碰撞风险检测参数 - 基于日志分析调整
        for det in self.deadlock_manager.agent_detectors.values():
            det.detection_interval = min(det.detection_interval, 1)
            # 降低碰撞风险距离阈值，提高检测灵敏度
            if not hasattr(det, 'collision_risk_distance_threshold'):
                setattr(det, 'collision_risk_distance_threshold', 3.0)  # 从4.0降到3.0
            else:
                det.collision_risk_distance_threshold = min(3.0, det.collision_risk_distance_threshold)  # 使用更小的值
            det.sight_radius = 3.0  # 保持与RL训练时一致的视野半径
        
        # 死锁状态
        self.deadlock_active = False
        self.deadlock_episodes = []
        self.current_episode_deadlock_events = []  # 当前episode的死锁事件
        self.deadlock_log_filename = f"deadlock_log_{current_time}.txt"

        # MAPF 管理器 & 环境适配器
        self.mapf_manager = MAPFManager()
        self.env_adapter = SimpleEnvAdapter(self.env)
        
        # MAPF执行超时机制
        self.mapf_timeout_steps = 20  # MAPF最大执行步数（减少超时时间）
        self.mapf_start_timestep = None  # MAPF开始时间步

    def _init_waypoints(self):
        """在每次环境 reset 后，按提供的 waypoint 序列设置初始目标。"""
        if not self.waypoint_sequences:
            return
        for aid in range(self.robot_number):
            seq = self.waypoint_sequences.get(aid)
            if not seq or len(seq) < 2:
                continue
            # 约定：seq[0] 为起点，seq[1] 为第一段目标
            self._waypoint_index[aid] = 1
            target = np.array(seq[1], dtype=float).reshape(2, 1)
            self.env.ir_gym.robot_list[aid].goal = target

    def _update_waypoints(self):
        """在每个 timestep 后推进到下一个 waypoint（若抵达阈值）。"""
        if not self.waypoint_sequences:
            return
        for aid in range(self.robot_number):
            seq = self.waypoint_sequences.get(aid)
            if not seq or len(seq) < 2:
                continue
            idx = self._waypoint_index.get(aid, 1)
            if idx >= len(seq):
                continue
            robot = self.env.ir_gym.robot_list[aid]
            pos = np.array([robot.state[0, 0], robot.state[1, 0]])
            cur_target = np.array(seq[idx], dtype=float)
            if np.linalg.norm(pos - cur_target) <= self.waypoint_goal_threshold and idx < len(seq) - 1:
                idx += 1
                self._waypoint_index[aid] = idx
                next_target = np.array(seq[idx], dtype=float).reshape(2, 1)
                robot.goal = next_target

    def get_collision_robots(self):
        """检测发生碰撞的机器人编号"""
        collision_robots = []
        for i, robot in enumerate(self.env.ir_gym.robot_list):
            if robot.collision_flag:
                collision_robots.append(i)
        

        
        return collision_robots
    
    def get_collision_pairs(self):
        """获取碰撞机器人对信息"""
        components = self.env.ir_gym.components
        if 'collision_pairs' in components:
            return components['collision_pairs']
        return []
    
    def update_deadlock_detection(self, timestep):
        """
        更新分布式死锁检测状态
        
        Args:
            timestep: 当前时间步
        """
        robot_list = self.env.ir_gym.robot_list
        
        # 提取所有agent的状态信息
        agent_positions = []
        agent_velocities = []
        agent_goals = []
        
        for robot in robot_list:
            pos = np.array([robot.state[0, 0], robot.state[1, 0]])
            vel = np.array([robot.vel_omni[0, 0], robot.vel_omni[1, 0]])
            goal = np.array([robot.goal[0, 0], robot.goal[1, 0]])
            
            agent_positions.append(pos)
            agent_velocities.append(vel)
            agent_goals.append(goal)
        
        # 更新分布式死锁检测器
        self.deadlock_manager.update_all_agents(
            agent_positions=agent_positions,
            agent_velocities=agent_velocities,
            agent_goals=agent_goals,
            timestep=timestep
        )
        
        # 检测死锁
        deadlock_agents, deadlock_groups = self.deadlock_manager.detect_deadlocks(timestep)
        
        # 更新死锁状态
        if deadlock_agents and not self.deadlock_active:
            self.deadlock_active = True
            self.mapf_start_timestep = timestep  # 记录MAPF开始时间
            print(f"死锁激活: 时间步{timestep}, 涉及{len(deadlock_agents)}个agent, 死锁组: {deadlock_groups}")
            self._log_deadlock_activation(timestep, deadlock_agents, deadlock_groups)
        elif not deadlock_agents and self.deadlock_active:
            self.deadlock_active = False
            self.mapf_start_timestep = None  # 重置MAPF开始时间
            print(f"死锁解除: 时间步{timestep}")
            self._log_deadlock_deactivation(timestep)
        elif deadlock_agents and self.deadlock_active:
            # 检查MAPF执行超时
            if (self.mapf_start_timestep is not None and 
                timestep - self.mapf_start_timestep > self.mapf_timeout_steps):
                print(f"MAPF执行超时: 时间步{timestep}, 已执行{timestep - self.mapf_start_timestep}步, 超过{self.mapf_timeout_steps}步限制")
                # 强制取消MAPF会话
                if hasattr(self, 'mapf_manager'):
                    try:
                        self.mapf_manager.cancel_all()
                        print(f"强制取消MAPF会话")
                    except Exception as e:
                        print(f"取消MAPF会话失败: {e}")
                # 重置死锁状态
                self.deadlock_active = False
                self.mapf_start_timestep = None
                # 清空让行冷却
                if hasattr(self, '_yield_cooldown'):
                    self._yield_cooldown = {}
            # 在死锁解除时立刻结束所有 MAPF 会话并清空让行冷却，避免继续接管
            if hasattr(self, 'mapf_manager'):
                try:
                    self.mapf_manager.cancel_all()
                except Exception:
                    pass
            if hasattr(self, '_yield_cooldown'):
                self._yield_cooldown = {}
            # 启动解除后的短时安全护栏：保持 let-yield 策略和限速，避免立刻回切RL导致贴近碰撞
            self._post_release_guard = 10  # 保护若干步，可按需调参
            # 保护对象设为解除前参与会话的活跃agent（若无法获取则保护全体）
            try:
                if getattr(self, '_last_mapf_group', None):
                    self._post_release_agents = list(self._last_mapf_group)
                else:
                    self._post_release_agents = [aid for aid in range(self.num_agents)]
            except Exception:
                self._post_release_agents = [aid for aid in range(self.robot_number)]
            # 打印解除瞬间各参与agent的位置与速度
            try:
                if self.inf_print:
                    pos = {aid: [self.env.ir_gym.robot_list[aid].state[0,0], self.env.ir_gym.robot_list[aid].state[1,0]] for aid in self._post_release_agents}
                    vel = {aid: [float(v) for v in self.env.ir_gym.robot_list[aid].vel_omni.reshape(-1)] for aid in self._post_release_agents}
                    print(f"Post-MAPF release at t={timestep}: positions={pos}, vels={vel}")
            except Exception:
                pass
        
        # 记录当前时间步的死锁状态（无论是否激活）
        if deadlock_agents:
            self._log_timestep_deadlock(timestep, deadlock_agents, deadlock_groups)
        
        return len(deadlock_agents) > 0, deadlock_agents, deadlock_groups
    
    def _log_deadlock_activation(self, timestep, deadlock_agents, deadlock_groups):
        """记录死锁激活信息"""
        # 获取详细的死锁初始化信息
        initialization_info = self._get_deadlock_initialization_info(timestep, deadlock_agents, deadlock_groups)
        
        deadlock_info = {
            'timestep': timestep,
            'type': 'activation',
            'agents': list(deadlock_agents),
            'groups': [list(group) for group in deadlock_groups],
            'summary': self.deadlock_manager.get_deadlock_summary(),
            'initialization_info': initialization_info
        }
        self.deadlock_episodes.append(deadlock_info)
        
        # 保存详细的初始化信息到文件
        self._save_deadlock_initialization_log(timestep, deadlock_agents, deadlock_groups, initialization_info)

    def _get_deadlock_initialization_info(self, timestep, deadlock_agents, deadlock_groups):
        """获取死锁初始化时的详细信息"""
        try:
            robot_list = self.env.ir_gym.robot_list
            
            initialization_info = {
                'timestep': timestep,
                'deadlock_agents': list(deadlock_agents),
                'deadlock_groups': [list(group) for group in deadlock_groups],
                'agent_states': {},
                'detector_info': {},
                'environment_info': {}
            }
            
            # 收集每个智能体的状态信息
            for aid in deadlock_agents:
                if aid < len(robot_list):
                    robot = robot_list[aid]
                    
                    # 基本状态
                    pos = np.array([robot.state[0, 0], robot.state[1, 0]])
                    vel = np.array([robot.vel_omni[0, 0], robot.vel_omni[1, 0]])
                    goal = np.array([robot.goal[0, 0], robot.goal[1, 0]])
                    
                    # 计算速度大小
                    speed = np.linalg.norm(vel)
                    
                    # 计算到目标的距离
                    distance_to_goal = np.linalg.norm(pos - goal)
                    
                    # 获取检测器信息
                    detector = self.deadlock_manager.agent_detectors.get(aid)
                    detector_info = {}
                    if detector:
                        detector_info = {
                            'trigger_type': detector.deadlock_trigger_type.value if detector.deadlock_trigger_type else None,
                            'mean_speed': detector.mean_speed,
                            'neighbor_count': len(detector.neighbors),
                            'speed_buffer_size': len(detector.speed_buffer),
                            'satisfied_streak': detector._satisfied_streak
                        }
                    
                    # 获取邻居信息
                    neighbors_info = []
                    if detector:
                        for dist, nei_det in detector.neighbors:
                            neighbors_info.append({
                                'neighbor_id': nei_det.agent_id,
                                'distance': dist,
                                'neighbor_speed': nei_det.mean_speed
                            })
                    
                    initialization_info['agent_states'][aid] = {
                        'position': pos.tolist(),
                        'velocity': vel.tolist(),
                        'speed': float(speed),
                        'goal': goal.tolist(),
                        'distance_to_goal': float(distance_to_goal),
                        'detector_info': detector_info,
                        'neighbors': neighbors_info
                    }
            
            # 收集环境信息
            if hasattr(self.env, 'ir_gym') and hasattr(self.env.ir_gym, 'world'):
                world = self.env.ir_gym.world
                initialization_info['environment_info'] = {
                    'world_size': getattr(world, 'world_size', None),
                    'obstacle_count': len(getattr(world, 'obs_list', [])),
                    'robot_count': len(robot_list)
                }
            
            return initialization_info
            
        except Exception as e:
            print(f"Error getting deadlock initialization info: {e}")
            return {'error': str(e)}

    def _save_deadlock_initialization_log(self, timestep, deadlock_agents, deadlock_groups, initialization_info):
        """保存死锁初始化详细信息到文件"""
        try:
            import os
            from datetime import datetime
            
            # 创建日志目录
            log_dir = "deadlock_initialization_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"deadlock_init_t{timestep}_agents{len(deadlock_agents)}_{timestamp}.txt"
            log_path = os.path.join(log_dir, log_filename)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("Deadlock Initialization Log\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本信息
                f.write(f"Timestep: {timestep}\n")
                f.write(f"Deadlock Agents: {list(deadlock_agents)}\n")
                f.write(f"Deadlock Groups: {[list(group) for group in deadlock_groups]}\n")
                f.write(f"Log Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 环境信息
                f.write("Environment Information:\n")
                f.write("-" * 25 + "\n")
                env_info = initialization_info.get('environment_info', {})
                for key, value in env_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # 智能体详细信息
                f.write("Agent States Information:\n")
                f.write("-" * 30 + "\n")
                agent_states = initialization_info.get('agent_states', {})
                for aid, state in agent_states.items():
                    f.write(f"Agent {aid}:\n")
                    f.write(f"  Position: ({state['position'][0]:.3f}, {state['position'][1]:.3f})\n")
                    f.write(f"  Velocity: ({state['velocity'][0]:.3f}, {state['velocity'][1]:.3f})\n")
                    f.write(f"  Speed: {state['speed']:.3f} m/s\n")
                    f.write(f"  Goal: ({state['goal'][0]:.3f}, {state['goal'][1]:.3f})\n")
                    f.write(f"  Distance to Goal: {state['distance_to_goal']:.3f} meters\n")
                    
                    # 检测器信息
                    detector_info = state.get('detector_info', {})
                    f.write(f"  Detector Info:\n")
                    f.write(f"    Trigger Type: {detector_info.get('trigger_type', 'None')}\n")
                    f.write(f"    Mean Speed: {detector_info.get('mean_speed', 0):.3f} m/s\n")
                    f.write(f"    Neighbor Count: {detector_info.get('neighbor_count', 0)}\n")
                    f.write(f"    Speed Buffer Size: {detector_info.get('speed_buffer_size', 0)}\n")
                    f.write(f"    Satisfied Streak: {detector_info.get('satisfied_streak', 0)}\n")
                    
                    # 邻居信息
                    neighbors = state.get('neighbors', [])
                    f.write(f"  Neighbors ({len(neighbors)}):\n")
                    for nei in neighbors:
                        f.write(f"    Agent {nei['neighbor_id']}: Distance={nei['distance']:.3f}m, Speed={nei['neighbor_speed']:.3f}m/s\n")
                    f.write("\n")
                
                # 死锁检测配置
                f.write("Deadlock Detection Configuration:\n")
                f.write("-" * 35 + "\n")
                if self.deadlock_manager.agent_detectors:
                    config = self.deadlock_manager.agent_detectors[0]
                    f.write(f"Speed Buffer Size: {config.speed_buffer_size}\n")
                    f.write(f"Small Speed Threshold: {config.small_speed_threshold} m/s\n")
                    f.write(f"Neighbor Speed Threshold: {config.neighbor_speed_threshold} m/s\n")
                    f.write(f"Min Neighbors for Deadlock: {config.min_neighbors_for_deadlock}\n")
                    f.write(f"Sight Radius: {config.sight_radius} meters\n")
                    f.write(f"Detection Interval: {config.detection_interval} timesteps\n")
                    f.write(f"Activation Hysteresis Steps: {config.activation_hysteresis_steps}\n")
                    f.write(f"Collision Risk Distance Threshold: {config.collision_risk_distance_threshold} meters\n")
                f.write("\n")
                
                # Waypoint信息
                f.write("Waypoint Information:\n")
                f.write("-" * 20 + "\n")
                if hasattr(self, 'waypoint_sequences'):
                    for aid in deadlock_agents:
                        if aid in self.waypoint_sequences:
                            waypoints = self.waypoint_sequences[aid]
                            f.write(f"Agent {aid} Waypoints: {[f'({wp[0]:.3f}, {wp[1]:.3f})' for wp in waypoints]}\n")
                        else:
                            f.write(f"Agent {aid}: No waypoint sequence\n")
                f.write("\n")
                
                # 触发条件分析
                f.write("Trigger Condition Analysis:\n")
                f.write("-" * 30 + "\n")
                for aid in deadlock_agents:
                    if aid in agent_states:
                        state = agent_states[aid]
                        detector_info = state.get('detector_info', {})
                        trigger_type = detector_info.get('trigger_type', 'Unknown')
                        mean_speed = detector_info.get('mean_speed', 0)
                        neighbor_count = detector_info.get('neighbor_count', 0)
                        
                        f.write(f"Agent {aid}:\n")
                        f.write(f"  Trigger Type: {trigger_type}\n")
                        f.write(f"  Mean Speed: {mean_speed:.3f} m/s")
                        if mean_speed < 0.1:
                            f.write(" (Below threshold)")
                        f.write("\n")
                        f.write(f"  Neighbor Count: {neighbor_count}")
                        if neighbor_count >= 2:
                            f.write(" (Sufficient for deadlock)")
                        f.write("\n")
                        f.write(f"  Current Speed: {state['speed']:.3f} m/s")
                        if state['speed'] < 0.1:
                            f.write(" (Low speed)")
                        f.write("\n")
                f.write("\n")
                
            print(f"[Deadlock] Initialization log saved to: {log_path}")
            
        except Exception as e:
            print(f"[Deadlock] Failed to save initialization log: {e}")
    
    def _log_timestep_deadlock(self, timestep, deadlock_agents, deadlock_groups):
        """记录时间步死锁信息"""
        timestep_info = {
            'timestep': timestep,
            'agents': list(deadlock_agents),
            'groups': [list(group) for group in deadlock_groups],
            'summary': self.deadlock_manager.get_deadlock_summary()
        }
        self.current_episode_deadlock_events.append(timestep_info)
    
    def _log_deadlock_deactivation(self, timestep):
        """记录死锁解除信息"""
        deadlock_info = {
            'timestep': timestep,
            'type': 'deactivation',
            'agents': [],
            'center': None,
            'region': None
        }
        self.deadlock_episodes.append(deadlock_info)
    
    def get_deadlock_info(self):
        """获取当前死锁信息"""
        return self.deadlock_manager.get_deadlock_summary()
    
    def save_deadlock_log(self, save_path):
        """保存死锁日志"""
        log_path = Path(save_path) / self.deadlock_log_filename
        
        with open(log_path, 'w') as f:
            f.write("Deadlock Detection Log\n")
            f.write("=" * 50 + "\n\n")
            
            # 记录死锁检测的主要配置
            f.write("Distributed Deadlock Detection Configuration:\n")
            f.write("-" * 40 + "\n")
            config = self.deadlock_manager.agent_detectors[0]  # 使用第一个agent的配置作为示例
            f.write(f"Speed Buffer Size: {config.speed_buffer_size}\n")
            f.write(f"Small Speed Threshold: {config.small_speed_threshold} m/s\n")
            f.write(f"Neighbor Speed Threshold: {config.neighbor_speed_threshold} m/s\n")
            f.write(f"Min Neighbors for Deadlock: {config.min_neighbors_for_deadlock}\n")
            f.write(f"Sight Radius: {config.sight_radius} meters\n")
            f.write(f"Detection Interval: {config.detection_interval} timesteps\n")
            f.write(f"Total Agents: {self.deadlock_manager.num_agents}\n")
            f.write("\n")
            
            # 记录检测统计信息
            f.write("Detection Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Episodes: {self.num_episodes}\n")
            f.write(f"Episodes with Deadlock: {len([ep for ep in self.deadlock_episodes if 'episode' in ep])}\n")
            f.write(f"Total Deadlock Events: {len(self.deadlock_episodes)}\n")
            f.write(f"Current Deadlock Agents: {len(self.deadlock_manager.deadlock_agents)}\n")
            f.write(f"Current Deadlock Groups: {len(self.deadlock_manager.deadlock_groups)}\n")
            f.write("\n")
            
            if self.deadlock_episodes:
                f.write("Deadlock Events by Episode:\n")
                f.write("=" * 50 + "\n")
                
                # 按episode组织信息
                episode_events = {}
                activation_events = []
                
                for event in self.deadlock_episodes:
                    if 'episode' in event:
                        # 这是episode级别的信息
                        episode_events[event['episode']] = event
                    elif 'type' in event and event['type'] == 'activation':
                        # 这是激活事件
                        activation_events.append(event)
                
                # 记录每个episode的死锁信息
                for episode_num in sorted(episode_events.keys()):
                    episode_info = episode_events[episode_num]
                    f.write(f"\nEpisode {episode_num}:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Episode Length: {episode_info['episode_length']}\n")
                    f.write(f"Total Deadlock Timesteps: {episode_info['total_deadlock_timesteps']}\n")
                    
                    if episode_info['deadlock_events']:
                        f.write(f"Deadlock Events:\n")
                        for event in episode_info['deadlock_events']:
                            f.write(f"  Timestep {event['timestep']}: Agents {event['agents']}, Groups {event['groups']}\n")
                    else:
                        f.write("No deadlock events in this episode.\n")
                    
                    f.write("\n")
                
                # 记录激活事件
                if activation_events:
                    f.write("\nDeadlock Activation Events:\n")
                    f.write("-" * 30 + "\n")
                    for event in activation_events:
                        f.write(f"Timestep {event['timestep']}: {event['type']}\n")
                        f.write(f"  Agents: {event['agents']}\n")
                        f.write(f"  Groups: {event['groups']}\n")
                        f.write("\n")
            else:
                f.write("No deadlock events detected during this test.\n")
                f.write("This could mean:\n")
                f.write("1. The test scenario did not trigger deadlock conditions\n")
                f.write("2. The detection parameters are too strict\n")
                f.write("3. The agents navigated successfully without conflicts\n")
                f.write("\n")
        
        print(f"死锁日志已保存到: {log_path}")
        if not self.deadlock_episodes:
            print("注意：未检测到死锁事件，但日志文件已创建并包含配置信息")

    def get_robot_neighbors(self, robot_id, action_list):
        """获取指定机器人正在考虑的neighbor信息"""
        ts = self.env.ir_gym.components['robots'].total_states()
        robot_state = ts[0][robot_id]  # 当前机器人状态
        nei_state_list = ts[1]  # 所有机器人作为neighbor的状态
        obs_cir_list = ts[2]  # 圆形障碍物状态
        obs_line_list = ts[3]  # 线型障碍物状态
        
        action = action_list[robot_id] if robot_id < len(action_list) else np.zeros(2)
        
        # 获取经过预处理的neighbor信息
        robot_state_proc, ns_list, oc_list, ol_list = self.env.ir_gym.rvo.preprocess(
            robot_state, nei_state_list, obs_cir_list, obs_line_list
        )
        
        # 获取排序后的观测信息
        obs_vo_list, _, _, _ = self.env.ir_gym.rvo.config_vo_inf(
            robot_state, nei_state_list, obs_cir_list, obs_line_list, action
        )
        
        # 找出在neighbors_region范围内的机器人ID
        considered_neighbors = []
        robot_pos = np.array([robot_state[0], robot_state[1]])
        
        for i, nei_state in enumerate(nei_state_list):
            if i != robot_id:  # 排除自己
                nei_pos = np.array([nei_state[0], nei_state[1]])
                distance = np.linalg.norm(robot_pos - nei_pos)
                if 0 < distance <= self.nr:
                    considered_neighbors.append(i)
        
        # 根据优先级排序，只保留前neighbors_num个
        if len(considered_neighbors) > self.nm:
            # 计算每个neighbor的优先级信息
            neighbor_priorities = []
            for nei_id in considered_neighbors:
                nei_state = nei_state_list[nei_id]
                nei_pos = np.array([nei_state[0], nei_state[1]])
                distance = np.linalg.norm(robot_pos - nei_pos)
                # 简化的优先级计算，实际应该与rvo_inter中的排序逻辑一致
                neighbor_priorities.append((nei_id, distance))
            
            # 按距离排序（距离越近优先级越高）
            neighbor_priorities.sort(key=lambda x: x[1])
            considered_neighbors = [x[0] for x in neighbor_priorities[:self.nm]]
        
        return considered_neighbors

    def record_timestep_info(self, timestep, observation_list, action_list, reward_list, done_list, info_list):
        """记录每个timestep的详细信息"""
        timestep_info = {
            'timestep': timestep,
            'robot_states': [],
            'rvo_observations': [],
            'actions': action_list.copy(),
            'rewards': reward_list.copy(),
            'done_flags': done_list.copy(),
            'info_flags': info_list.copy()
        }
        
        # 获取当前环境状态
        ts = self.env.ir_gym.components['robots'].total_states()
        robot_state_list = ts[0]
        nei_state_list = ts[1]
        obs_cir_list = ts[2]
        obs_line_list = ts[3]
        
        vo_flags = []  # 收集所有机器人的VO Flag
        
        for robot_id in range(self.robot_number):
            robot = self.env.ir_gym.robot_list[robot_id]
            robot_state = robot_state_list[robot_id]
            observation = observation_list[robot_id]
            action = action_list[robot_id]
            
            # 获取RVO详细信息
            obs_vo_list, vo_flag, min_exp_time, collision_flag = self.env.ir_gym.rvo.config_vo_inf(
                robot_state, nei_state_list, obs_cir_list, obs_line_list, action
            )
            
            vo_flags.append(vo_flag)  # 收集VO Flag
            
            # 获取经过预处理的neighbor信息
            robot_state_proc, ns_list, oc_list, ol_list = self.env.ir_gym.rvo.preprocess(
                robot_state, nei_state_list, obs_cir_list, obs_line_list
            )
            
            robot_info = {
                'robot_id': robot_id,
                'position': [round(x, 4) for x in robot_state[0:2].tolist()],
                'velocity': [round(x, 4) for x in robot_state[2:4].tolist()],
                'radius': round(robot_state[4], 4),
                'desired_velocity': [round(x, 4) for x in robot_state[5:7].tolist()],
                'orientation': round(float(robot.state[2]), 4),
                'current_velocity_omni': [round(float(x), 4) for x in robot.vel_omni.flatten().tolist()],
                'rvo_observation': [round(x, 4) for x in observation.tolist()],
                'action': [round(x, 4) for x in action.tolist()],
                'vo_flag': vo_flag,
                'min_exp_time': round(min_exp_time, 4) if min_exp_time != float('inf') else min_exp_time,
                'collision_flag': collision_flag,
                'neighbors_in_region': len(ns_list),
                'circular_obstacles_in_region': len(oc_list),
                'line_obstacles_in_region': len(ol_list),
                'vo_list': [[round(x, 4) for x in (vo if isinstance(vo, list) else vo.tolist())] for vo in obs_vo_list],
                'neighbor_states': [[round(x, 4) for x in nei.tolist()] for nei in ns_list],
                'circular_obstacle_states': [[round(x, 4) for x in obs.tolist()] for obs in oc_list],
                'line_obstacle_states': [obs for obs in ol_list]  # 线型障碍物保持原格式
            }
            
            timestep_info['robot_states'].append(robot_info)
            timestep_info['rvo_observations'].append(obs_vo_list)
        
        # 添加到当前episode历史
        self.current_episode_history.append(timestep_info)
        
        # 保持历史长度限制
        if len(self.current_episode_history) > self.history_length:
            self.current_episode_history.pop(0)
        
        # 记录VO Flag信息
        self.record_vo_flag_info(timestep, vo_flags)

    def record_vo_flag_info(self, timestep, vo_flags):
        """记录每个timestep的VO Flag信息"""
        vo_flag_info = {
            'timestep': timestep,
            'vo_flags': vo_flags.copy()
        }
        
        # 添加到当前episode的VO Flag历史
        if not hasattr(self, 'current_vo_flag_history'):
            self.current_vo_flag_history = []
        
        self.current_vo_flag_history.append(vo_flag_info)
        
        # 保持历史长度限制
        if len(self.current_vo_flag_history) > self.history_length:
            self.current_vo_flag_history.pop(0)

    def log_collision_info(self, episode, collision_robots, action_list):
        """记录碰撞信息到文件"""
        collision_pairs = self.get_collision_pairs()
        
        collision_info = {
            'episode': episode,
            'collision_robots': collision_robots,
            'collision_pairs': collision_pairs,
            'neighbors_info': {},
            'failure_history': self.current_episode_history.copy(),  # 添加失败前30个timestep的历史
            'vo_flag_history': self.current_vo_flag_history.copy() if hasattr(self, 'current_vo_flag_history') else []
        }
        
        for robot_id in collision_robots:
            neighbors = self.get_robot_neighbors(robot_id, action_list)
            collision_info['neighbors_info'][robot_id] = neighbors
        
        self.collision_episodes.append(collision_info)
        
        # 添加到VO Flag episodes
        self.vo_flag_episodes.append(collision_info)
        
        # 打印碰撞信息
        print(f"Collision detected in episode {episode}:")
        print(f"  Collision robots: {collision_robots}")
        if collision_pairs:
            print(f"  Collision pairs: {collision_pairs}")
        for robot_id in collision_robots:
            neighbors = collision_info['neighbors_info'][robot_id]
            print(f"  Robot {robot_id} considering neighbors: {neighbors}")

    def save_collision_log(self, save_path):
        """保存碰撞记录到文件"""
        # 修改：即使没有碰撞也创建日志文件
        # 直接保存到policy_test文件夹，而不是创建子文件夹
        if "policy_test" in str(save_path):
            log_path = Path(save_path) / self.collision_log_filename
        else:
            log_path = Path(save_path) / "policy_test" / self.collision_log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Collision Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total episodes: {self.num_episodes}\n")
            f.write(f"Total collision episodes: {len(self.collision_episodes)}\n")
            f.write(f"Neighbor region: {self.nr}\n")
            f.write(f"Max neighbors considered: {self.nm}\n")
            f.write("=" * 80 + "\n\n")
            
            if not self.collision_episodes:
                f.write("No collisions detected during testing.\n")
            else:
                for collision_info in self.collision_episodes:
                    f.write(f"Episode {collision_info['episode']}:\n")
                    f.write(f"  Collision robots: {collision_info['collision_robots']}\n")
                    
                    if 'collision_pairs' in collision_info and collision_info['collision_pairs']:
                        f.write(f"  Collision pairs: {collision_info['collision_pairs']}\n")
                    
                    for robot_id, neighbors in collision_info['neighbors_info'].items():
                        f.write(f"  Robot {robot_id} was considering neighbors: {neighbors}\n")
                    
                    # 添加失败前20个timestep的详细信息
                    if 'failure_history' in collision_info and collision_info['failure_history']:
                        f.write(f"\n  Failure History (Last {len(collision_info['failure_history'])} timesteps):\n")
                        f.write("  " + "="*60 + "\n")
                        
                        for timestep_info in collision_info['failure_history']:
                            f.write(f"    Timestep {timestep_info['timestep']}:\n")
                            
                            for robot_info in timestep_info['robot_states']:
                                robot_id = robot_info['robot_id']
                                f.write(f"      Robot {robot_id}:\n")
                                f.write(f"        Position: {robot_info['position']}\n")
                                f.write(f"        Velocity: {robot_info['velocity']}\n")
                                f.write(f"        Desired Velocity: {robot_info['desired_velocity']}\n")
                                f.write(f"        Action: {robot_info['action']}\n")
                                f.write(f"        VO Flag: {robot_info['vo_flag']}\n")
                                f.write(f"        Min Exp Time: {robot_info['min_exp_time']:.4f}\n")
                                f.write(f"        Collision Flag: {robot_info['collision_flag']}\n")
                                f.write(f"        Neighbors in Region: {robot_info['neighbors_in_region']}\n")
                                f.write(f"        Circular Obstacles: {robot_info['circular_obstacles_in_region']}\n")
                                f.write(f"        Line Obstacles: {robot_info['line_obstacles_in_region']}\n")
                                
                                # 记录RVO观测信息
                                if robot_info['vo_list']:
                                    f.write(f"        RVO Observations ({len(robot_info['vo_list'])} VOs):\n")
                                    for i, vo in enumerate(robot_info['vo_list']):
                                        f.write(f"          VO {i}: {vo}\n")
                                else:
                                    f.write(f"        RVO Observations: No VOs detected\n")
                                
                                # 记录邻居状态
                                if robot_info['neighbor_states']:
                                    f.write(f"        Neighbor States ({len(robot_info['neighbor_states'])} robots):\n")
                                    for i, nei in enumerate(robot_info['neighbor_states']):
                                        f.write(f"          Neighbor {i}: pos=[{nei[0]:.4f}, {nei[1]:.4f}], vel=[{nei[2]:.4f}, {nei[3]:.4f}], radius={nei[4]:.4f}\n")
                                
                                f.write("\n")
                            
                            f.write("      " + "-"*40 + "\n")
                    
                    f.write("\n")
        
        print(f"Collision log saved to: {log_path}")

    def save_vo_flag_log(self, save_path):
        """保存VO Flag简化日志到文件"""
        if "policy_test" in str(save_path):
            log_path = Path(save_path) / self.vo_flag_log_filename
        else:
            log_path = Path(save_path) / "policy_test" / self.vo_flag_log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"VO Flag Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total episodes: {self.num_episodes}\n")
            f.write(f"Total collision episodes: {len(self.vo_flag_episodes)}\n")
            f.write(f"Robot number: {self.robot_number}\n")
            f.write("=" * 80 + "\n\n")
            
            if not self.vo_flag_episodes:
                f.write("No collisions detected during testing.\n")
            else:
                for collision_info in self.vo_flag_episodes:
                    f.write(f"Episode {collision_info['episode']}:\n")
                    f.write(f"  Collision robots: {collision_info['collision_robots']}\n")
                    
                    if 'vo_flag_history' in collision_info and collision_info['vo_flag_history']:
                        f.write(f"\n  VO Flag History (Last {len(collision_info['vo_flag_history'])} timesteps):\n")
                        f.write("  " + "="*60 + "\n")
                        
                        for timestep_info in collision_info['vo_flag_history']:
                            f.write(f"    Timestep {timestep_info['timestep']}: {timestep_info['vo_flags']}\n")
                    
                    f.write("\n")
        
        print(f"VO Flag log saved to: {log_path}")

    def policy_test(self, policy_type='drl', policy_path=None, policy_name='policy', result_path=None, result_name='/result.txt', figure_save_path=None, ani_save_path=None, policy_dict=False, once=False):
        
        if policy_type == 'drl':
            model_action = self.load_policy(policy_path, self.std_factor, policy_dict=policy_dict)

        o, r, d, ep_ret, ep_len, n = self.env.reset(mode=self.reset_mode), 0, False, 0, 0, 0
        # 初始化 waypoint 目标（若提供）
        self._init_waypoints()
        # 初始化碰撞对信息
        self.env.ir_gym.reset_collision_pairs()
        ep_ret_list, speed_list, mean_speed_list, ep_len_list, sn = [], [], [], [], 0
        
        # 添加失败原因统计
        collision_failures = 0
        timeout_failures = 0

        print('Policy Test Start !')

        figure_id = 0
        while n < self.num_episodes:

            # if n == 1:
            #     self.show_traj = True

            action_time_list = []

            if self.render or self.save:
                self.env.render(save=self.save, path=figure_save_path, i = figure_id, show_traj=self.show_traj, traj_type=self.traj_type)
            
            if policy_type == 'drl': 
                abs_action_list =[]
                for i in range(self.robot_number):

                    start_time = time.time()
                    a_inc = np.round(model_action(o[i]), 2)
                    end_time = time.time()

                    temp = end_time - start_time
                    action_time_list.append(temp)

                    cur_vel = self.env.ir_gym.robot_list[i].vel_omni
                    abs_action = self.acceler_vel * a_inc + np.squeeze(cur_vel)
                    abs_action_list.append(abs_action)

            # 更新死锁检测（在执行控制前，便于本步覆盖动作）
            deadlock_detected, deadlock_agents, deadlock_groups = \
                self.update_deadlock_detection(ep_len)
            
            # 增强日志记录：记录死锁检测的详细信息
            if deadlock_detected and self.inf_print:
                print(f"死锁检测激活 at t={ep_len}: agents={deadlock_agents}, groups={deadlock_groups}")
                # 记录agent之间的距离
                positions = {aid: np.array([r.state[0, 0], r.state[1, 0]]) for aid, r in enumerate(self.env.ir_gym.robot_list)}
                for i in range(self.robot_number):
                    for j in range(i+1, self.robot_number):
                        distance = np.linalg.norm(positions[i] - positions[j])
                        print(f"  Agent {i}-{j} distance: {distance:.3f}m")

            # 保护：若当前不在死锁状态，但仍存在活跃会话，立即清理，防止解除后继续接管
            if not deadlock_detected:
                try:
                    if any(self.mapf_manager.is_active(aid) for aid in range(self.robot_number)):
                        self.mapf_manager.cancel_all()
                        if hasattr(self, '_yield_cooldown'):
                            self._yield_cooldown = {}
                except Exception:
                    pass
                # 解除后安全护栏的“安全清零”判据：若连续M步满足VO全False且最近间距>1.0m，则提前结束护栏
                if getattr(self, '_post_release_guard', 0) > 0:
                    try:
                        # 计算受保护对之间的最小间距与VO状态
                        positions = {aid: np.array([r.state[0, 0], r.state[1, 0]], dtype=float) for aid, r in enumerate(self.env.ir_gym.robot_list)}
                        min_sep = float('inf')
                        all_vo_false = True
                        guard_agents = set(getattr(self, '_post_release_agents', []))
                        protected = [aid for aid in range(self.robot_number) if (aid in guard_agents)]
                        # 获取当前VO Flags（已由上游记录），保底为False
                        current_vo_flags = []
                        if hasattr(self, 'current_vo_flag_history') and self.current_vo_flag_history:
                            current_vo_flags = self.current_vo_flag_history[-1].get('vo_flags', [])
                        for i in range(len(protected)):
                            for j in range(i+1, len(protected)):
                                ai, aj = protected[i], protected[j]
                                min_sep = min(min_sep, float(np.linalg.norm(positions[ai] - positions[aj])))
                        # 判定VO：若有缺失则保守认为存在约束
                        if current_vo_flags:
                            all_vo_false = all(flag == False for flag in current_vo_flags[:self.robot_number])
                        else:
                            all_vo_false = False
                        if all_vo_false and min_sep > 1.2:
                            self._post_release_safe_streak += 1
                            if self._post_release_safe_streak >= 2:
                                self._post_release_guard = 0
                        else:
                            self._post_release_safe_streak = 0
                    except Exception:
                        pass

            # 如果检测到死锁，尝试启动 MAPF 会话
            if deadlock_detected and deadlock_groups:
                # 组装 agent_states
                positions = {}
                goals = {}
                for aid, robot in enumerate(self.env.ir_gym.robot_list):
                    positions[aid] = np.array([robot.state[0, 0], robot.state[1, 0]], dtype=float)
                    goals[aid] = np.array([robot.goal[0, 0], robot.goal[1, 0]], dtype=float)
                agent_states = {"positions": positions, "goals": goals}

                # waypoints：复用构造时传入的序列
                waypoints_dict = self.waypoint_sequences if self.waypoint_sequences else {aid: [goals[aid]] for aid in positions.keys()}
                started_sessions = self.mapf_manager.try_start(deadlock_groups, agent_states, self.env_adapter, waypoints_dict, ep_len)
                if self.inf_print and started_sessions:
                    print(f"MAPF started at t={ep_len}, sessions={started_sessions}, groups={deadlock_groups}")
                # 记录最近一次会话涉及的 agent 集合
                if started_sessions and deadlock_groups:
                    try:
                        merged = set()
                        for g in deadlock_groups:
                            merged.update(g)
                        self._last_mapf_group = set(merged)
                    except Exception:
                        self._last_mapf_group = set()
                # 强制首帧优先级让行：同组内除最小ID外的agent先让行一拍，避免相互抢占
                if started_sessions:
                    if not hasattr(self, '_yield_cooldown'):
                        self._yield_cooldown = {}
                    for group in deadlock_groups:
                        if not group:
                            continue
                        lo_id = min(group)
                        for aid in group:
                            if aid != lo_id:
                                self._yield_cooldown[aid] = max(2, self._yield_cooldown.get(aid, 0))

            # 若 MAPF 活跃，则推进执行并覆盖动作目标
            active_any = any(self.mapf_manager.is_active(aid) for aid in range(self.robot_number))
            # 优化：即使死锁解除，也继续执行MAPF直到安全距离
            if active_any and (self.deadlock_active or self._should_continue_mapf()):
                if self.inf_print:
                    active_agents = [aid for aid in range(self.robot_number) if self.mapf_manager.is_active(aid)]
                    print(f"MAPF active at t={ep_len}, agents={active_agents}")
                cur_positions = {aid: np.array([r.state[0, 0], r.state[1, 0]], dtype=float) for aid, r in enumerate(self.env.ir_gym.robot_list)}
                # 使用底层 ir_gym 的步长
                self.mapf_manager.step_execute(cur_positions, self.env.ir_gym.step_time)
                # 碰撞护栏：基于优先级（ID升序）让行，低ID优先
                active_agents = [aid for aid in range(self.robot_number) if self.mapf_manager.is_active(aid)]
                yield_agents = set()
                yield_reason = {}

                # 预取各 agent 的下一目标
                next_targets = {aid: self.mapf_manager.next_target(aid) for aid in active_agents}
                # 应用冷却：仍在冷却期的直接让行
                for aid in active_agents:
                    remain = getattr(self, '_yield_cooldown', {}).get(aid, 0)
                    if remain > 0:
                        yield_agents.add(aid)
                        yield_reason[aid] = 'cooldown'
                        self._yield_cooldown[aid] = remain - 1

                # 规则：在近距离或对向冲突下，较大ID一律让行（优先级让行）
                for ai in active_agents:
                    for aj in active_agents:
                        if ai == aj:
                            continue
                        pi, pj = cur_positions[ai], cur_positions[aj]
                        sep = float(np.linalg.norm(pi - pj))
                        # 条件1：距离过近（放宽到1.5m）
                        near_conflict = sep < 1.5
                        # 条件2：对向冲突（方向几乎相反且较近）
                        ti, tj = next_targets.get(ai), next_targets.get(aj)
                        head_on = False
                        if ti is not None and tj is not None:
                            vi = ti - pi
                            vj = tj - pj
                            nvi = vi / np.linalg.norm(vi) if np.linalg.norm(vi) > 1e-6 else vi
                            nvj = vj / np.linalg.norm(vj) if np.linalg.norm(vj) > 1e-6 else vj
                            head_on = (sep < 1.5 and float(np.dot(nvi, nvj)) < -0.1)
                        if near_conflict or head_on:
                            # 较大ID让行
                            if aj < ai and ai not in yield_agents:
                                yield_agents.add(ai)
                                yield_reason[ai] = 'distance' if near_conflict else 'head-on'
                                # 设置冷却，避免下一帧立刻夺回
                                if not hasattr(self, '_yield_cooldown'):
                                    self._yield_cooldown = {}
                                self._yield_cooldown[ai] = max(2, self._yield_cooldown.get(ai, 0))
                # 选出当前活跃集合中的最低ID，赋予优先通行权（避免双方都停导致超时）
                min_active_id = min(active_agents) if active_agents else None
                # 将处于 MAPF 的 agent 的动作替换为改进的协调机制（覆盖 abs_action_list）
                for aid in range(self.robot_number):
                    if self.mapf_manager.is_active(aid):
                        target = next_targets.get(aid)
                        if target is not None:
                            pos = cur_positions[aid]
                            
                            # 让行逻辑
                            if aid in yield_agents:
                                # 被判为让行：完全停下
                                abs_action_list[aid] = np.array([0.0, 0.0], dtype=float)
                                if self.inf_print:
                                    reason = yield_reason.get(aid, 'priority')
                                    print(f"  override action for agent {aid} yield ({reason})")
                            else:
                                # 使用改进的协调机制
                                rl_action = abs_action_list[aid]  # 保存原始RL动作
                                coordinated_action = self._improved_action_coordination(aid, rl_action, target, pos)
                                
                                # 限制速度幅值，避免过激
                                action_norm = np.linalg.norm(coordinated_action)
                                if action_norm > 0:
                                    max_speed = min(self.acceler_vel, 0.6)
                                    abs_action_list[aid] = coordinated_action / action_norm * max_speed
                                
                                if self.inf_print:
                                    print(f"  coordinated action for agent {aid} towards {target.tolist()}")

            # 若处于解除后的安全护栏期，对高风险相邻对施加轻量让行与限速（不依赖会话）
            if getattr(self, '_post_release_guard', 0) > 0:
                try:
                    # 仅对受保护agent应用
                    guard_agents = set(getattr(self, '_post_release_agents', []))
                    # 近距离/对向冲突避让（与MAPF接管期一致但更保守）
                    positions = {aid: np.array([r.state[0, 0], r.state[1, 0]], dtype=float) for aid, r in enumerate(self.env.ir_gym.robot_list)}
                    protected = [aid for aid in range(self.robot_number) if (aid in guard_agents)]
                    for ai in protected:
                        for aj in protected:
                            if ai >= aj:
                                continue
                            pi, pj = positions[ai], positions[aj]
                            sep = float(np.linalg.norm(pi - pj))
                            if sep < 0.8:
                                # 高ID让行（速度置零），低ID限速
                                hi, lo = (ai, aj) if ai > aj else (aj, ai)
                                abs_action_list[hi] = np.array([0.0, 0.0], dtype=float)
                                # 低ID将动作幅值限制到0.5
                                v = abs_action_list[lo]
                                norm = float(np.linalg.norm(v))
                                if norm > 1e-6:
                                    abs_action_list[lo] = v / norm * min(self.acceler_vel, 0.5)
                    # 递减剩余保护步数
                    self._post_release_guard -= 1
                except Exception:
                    pass

            # 执行动作（已考虑接管覆盖与解除后护栏）
            o, r, d, info = self.env.step_ir(abs_action_list, vel_type = 'omni')

            # 推进 waypoint（若提供）
            self._update_waypoints()

            # 记录当前timestep的详细信息（记录实际下发的动作）
            self.record_timestep_info(ep_len, o, abs_action_list, r, d, info)

            robot_speed_list = [np.linalg.norm(robot.vel_omni) for robot in self.env.ir_gym.robot_list]
            avg_speed = np.average(robot_speed_list)
            speed_list.append(avg_speed)

            ep_ret += r[0]
            ep_len += 1
            figure_id += 1

            if np.max(d) or (ep_len == self.max_ep_len) or np.min(info):
                speed = np.mean(speed_list)
                figure_id = 0
                
                # 先记录碰撞信息，再重置状态
                collision_detected = False
                if np.max(d):  # 碰撞发生
                    collision_failures += 1
                    collision_detected = True
                    
                    # 在重置之前立即记录碰撞信息
                    collision_robots = self.get_collision_robots()
                    if collision_robots:
                        self.log_collision_info(n, collision_robots, abs_action_list)

                
                if np.min(info):
                    ep_len_list.append(ep_len)
                    if self.inf_print: print('Successful, Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
                else:
                    # 统计失败原因
                    if collision_detected:
                        if self.inf_print: print('Fail (Collision), Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
                    else:  # 超时
                        timeout_failures += 1
                        if self.inf_print: print('Fail (Timeout), Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
    
                ep_ret_list.append(ep_ret)
                mean_speed_list.append(speed)
                speed_list = []

                # 清理所有MAPF会话，避免跨episode残留
                self.mapf_manager.cancel_all()
                o, r, d, ep_ret, ep_len = self.env.reset(mode=self.reset_mode), 0, False, 0, 0
                # 重置后恢复到第一段 waypoint（若提供）
                self._init_waypoints()
                # 重置碰撞对信息
                self.env.ir_gym.reset_collision_pairs()
                                # 重置当前episode历史记录
                self.current_episode_history = []
                # 重置VO Flag历史记录
                if hasattr(self, 'current_vo_flag_history'):
                    self.current_vo_flag_history = []
                # 重置让行冷却
                self._yield_cooldown = {}
                # 保存当前episode的死锁信息
                if self.current_episode_deadlock_events:
                    episode_deadlock_info = {
                        'episode': n,
                        'deadlock_events': self.current_episode_deadlock_events.copy(),
                        'total_deadlock_timesteps': len(self.current_episode_deadlock_events),
                        'episode_length': ep_len
                    }
                    self.deadlock_episodes.append(episode_deadlock_info)
                
                # 重置死锁检测器
                self.deadlock_manager.reset()
                self.deadlock_active = False
                self.current_episode_deadlock_events = []

                n += 1

                if np.min(info):
                    sn+=1
                    
                    # if n == 2: 
                        
                    if once:
                        self.env.ir_gym.world_plot.save_gif_figure(figure_save_path, 0, format='eps')
                        break
                        
                    if self.save:
                        self.env.ir_gym.save_ani(figure_save_path, ani_save_path, ani_name=policy_name)
                        break

        # 保存碰撞记录
        if result_path:
            self.save_collision_log(result_path)
            self.save_vo_flag_log(result_path)
            self.save_deadlock_log(result_path)

        mean_len = 0 if len(ep_len_list) == 0 else np.round(np.mean(ep_len_list), 2)
        std_len = 0 if len(ep_len_list) == 0 else np.round(np.std(ep_len_list), 2)

        average_speed = np.round(np.mean(mean_speed_list),2)
        std_speed = np.round(np.std(mean_speed_list), 2)

        # 计算失败率
        total_failures = collision_failures + timeout_failures
        collision_rate = collision_failures / total_failures if total_failures > 0 else 0
        timeout_rate = timeout_failures / total_failures if total_failures > 0 else 0

        f = open( result_path + result_name, 'a')
        print( 'policy_name: '+ policy_name, 
               ' successful rate: {:.2%}'.format(sn/self.num_episodes),
               "average EpLen:", mean_len, 
               "std length", std_len, 
               'average speed:', average_speed, 
               'std speed', std_speed,
               '\nFailure Analysis:',
               'Collision failures:', collision_failures,
               '({:.2%})'.format(collision_rate),
               'Timeout failures:', timeout_failures,
               '({:.2%})'.format(timeout_rate),
               file = f)
        f.close() 
        
        print( 'policy_name: '+ policy_name, 
               ' successful rate: {:.2%}'.format(sn/self.num_episodes),
               "average EpLen:", mean_len, 
               'std length', std_len, 
               'average speed:', average_speed, 
               'std speed', std_speed)
        print('Failure Analysis:',
              'Collision failures:', collision_failures,
              '({:.2%})'.format(collision_rate),
              'Timeout failures:', timeout_failures,
              '({:.2%})'.format(timeout_rate))

    def load_policy(self, filename, std_factor=1, policy_dict=False):

        if policy_dict == True:
            model = rnn_ac(self.env.observation_space, self.env.action_space, self.args.state_dim, self.args.rnn_input_dim, self.args.rnn_hidden_dim, self.args.hidden_sizes_ac, self.args.hidden_sizes_v, self.args.activation, self.args.output_activation, self.args.output_activation_v, self.args.use_gpu, self.args.rnn_mode)
        
            check_point = torch.load(filename)
            model.load_state_dict(check_point['model_state'], strict=True)
            model.eval()

        else:
            model = torch.load(filename)
            model.eval()

        # model.train()
        def get_action(x):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                action = model.act(x, std_factor)
            return action

        return get_action
    
    def dis(self, p1, p2):
        return sqrt( (p2.py - p1.py)**2 + (p2.px - p1.px)**2 )
    
    def _should_continue_mapf(self):
        """判断是否应该继续执行MAPF，即使死锁已解除"""
        # 检查是否有agent距离过近
        positions = {}
        for i, robot in enumerate(self.env.ir_gym.robot_list):
            positions[i] = np.array([robot.state[0, 0], robot.state[1, 0]])
        
        # 检查任意两个agent之间的距离
        for i in range(self.robot_number):
            for j in range(i+1, self.robot_number):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < 2.0:  # 如果距离小于2米，继续MAPF
                    return True
        
        return False
    
    def _improved_action_coordination(self, aid, rl_action, mapf_target, current_position):
        """改进的动作协调机制"""
        if mapf_target is None:
            return rl_action
        
        # 计算到MAPF目标的方向
        direction_to_target = mapf_target - current_position
        distance_to_target = np.linalg.norm(direction_to_target)
        
        if distance_to_target < 0.1:
            return rl_action
        
        # 归一化方向
        direction_to_target = direction_to_target / distance_to_target
        
        # 混合RL动作和MAPF目标
        # 距离越近，MAPF权重越大
        mapf_weight = min(0.8, max(0.3, 1.0 - distance_to_target / 3.0))
        rl_weight = 1.0 - mapf_weight
        
        blended_action = mapf_weight * direction_to_target + rl_weight * rl_action
        
        # 归一化
        action_norm = np.linalg.norm(blended_action)
        if action_norm > 0:
            blended_action = blended_action / action_norm
        
        return blended_action