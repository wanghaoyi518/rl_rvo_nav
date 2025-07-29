import torch
import numpy as np
from pathlib import Path
import platform
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac
from math import pi, sin, cos, sqrt
import time 
from datetime import datetime

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

            o, r, d, info = self.env.step_ir(abs_action_list, vel_type = 'omni')

            # 记录当前timestep的详细信息
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

                o, r, d, ep_ret, ep_len = self.env.reset(mode=self.reset_mode), 0, False, 0, 0
                # 重置碰撞对信息
                self.env.ir_gym.reset_collision_pairs()
                # 重置当前episode历史记录
                self.current_episode_history = []
                # 重置VO Flag历史记录
                if hasattr(self, 'current_vo_flag_history'):
                    self.current_vo_flag_history = []

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