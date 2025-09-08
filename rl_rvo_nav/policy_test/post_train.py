import torch
import numpy as np
from pathlib import Path
import platform
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac
from math import pi, sin, cos, sqrt
import time
from rl_rvo_nav.test_logger import TestLogger 

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
        
        # Initialize test logger
        self.test_logger = TestLogger(test_type="test")
        self.test_logger.set_environment(env)

    def policy_test(self, policy_type='drl', policy_path=None, policy_name='policy', result_path=None, result_name='/result.txt', figure_save_path=None, ani_save_path=None, policy_dict=False, once=False):
        
        if policy_type == 'drl':
            model_action = self.load_policy(policy_path, self.std_factor, policy_dict=policy_dict)

        o, r, d, ep_ret, ep_len, n = self.env.reset(mode=self.reset_mode), 0, False, 0, 0, 0
        ep_ret_list, speed_list, mean_speed_list, ep_len_list, sn = [], [], [], [], 0

        print('Policy Test Start !')

        figure_id = 0
        episode_started = False
        
        while n < self.num_episodes:
            
            # Start logging episode only once per episode
            if not episode_started:
                episode_config = {
                    'policy_name': policy_name,
                    'policy_type': policy_type,
                    'reset_mode': self.reset_mode,
                    'max_ep_len': self.max_ep_len,
                    'render': self.render
                }
                self.test_logger.start_episode(n, self.robot_number, episode_config)
                episode_started = True

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

            # Log step data
            robot_positions = self.test_logger.get_robot_positions_from_env(self.env)
            robot_velocities = self.test_logger.get_robot_velocities_from_env(self.env)
            step_info = {
                'reward': float(r[0]),
                'done': bool(np.max(d)),
                'info': info.tolist() if hasattr(info, 'tolist') else info
            }
            self.test_logger.log_step(robot_positions, robot_velocities, step_info)

            robot_speed_list = [np.linalg.norm(robot.vel_omni) for robot in self.env.ir_gym.robot_list]
            avg_speed = np.average(robot_speed_list)
            speed_list.append(avg_speed)

            ep_ret += r[0]
            ep_len += 1
            figure_id += 1

            if np.max(d) or (ep_len == self.max_ep_len) or np.min(info):
                speed = np.mean(speed_list)
                figure_id = 0
                
                # Determine episode success and failure reason
                episode_success = bool(np.min(info))
                failure_reason = None
                
                if episode_success:
                    ep_len_list.append(ep_len)
                    if self.inf_print: print('Successful, Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
                else:
                    if self.inf_print: print('Fail, Episode %d \t EpRet %.3f \t EpLen %d \t EpSpeed  %.3f'%(n, ep_ret, ep_len, speed))
                    
                    # Print failure reasons and collect for logging
                    failure_reason = self._get_failure_reasons()
                    self._print_failure_reasons()
                
                # End episode logging
                self.test_logger.end_episode(
                    success=episode_success,
                    episode_reward=ep_ret,
                    episode_length=ep_len,
                    failure_reason=failure_reason
                )
    
                ep_ret_list.append(ep_ret)
                mean_speed_list.append(speed)
                speed_list = []

                o, r, d, ep_ret, ep_len = self.env.reset(mode=self.reset_mode), 0, False, 0, 0

                n += 1
                episode_started = False  # Reset for next episode

                if np.min(info):
                    sn+=1
                    
                    # if n == 2: 
                        
                    if once:
                        self.env.ir_gym.world_plot.save_gif_figure(figure_save_path, 0, format='eps')
                        break
                        
                    if self.save:
                        self.env.ir_gym.save_ani(figure_save_path, ani_save_path, ani_name=policy_name)
                        break

        mean_len = 0 if len(ep_len_list) == 0 else np.round(np.mean(ep_len_list), 2)
        std_len = 0 if len(ep_len_list) == 0 else np.round(np.std(ep_len_list), 2)

        average_speed = np.round(np.mean(mean_speed_list),2)
        std_speed = np.round(np.std(mean_speed_list), 2)

        f = open( result_path + result_name, 'a')
        print( 'policy_name: '+ policy_name, ' successful rate: {:.2%}'.format(sn/self.num_episodes), "average EpLen:", mean_len, "std length", std_len, 'average speed:', average_speed, 'std speed', std_speed, file = f)
        f.close() 
        
        print( 'policy_name: '+ policy_name, ' successful rate: {:.2%}'.format(sn/self.num_episodes), "average EpLen:", mean_len, 'std length', std_len, 'average speed:', average_speed, 'std speed', std_speed)
        
        # Save session summary
        self.test_logger.save_session_summary()


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
    
    def _get_failure_reasons(self):
        """Get detailed failure reasons for failed episode as structured data"""
        collision_robots = []
        
        # Collect collision information from all robots
        for robot in self.env.ir_gym.robot_list:
            if robot.collision_flag and robot.collision_info is not None:
                collision_robots.append(robot.collision_info)
        
        if collision_robots:
            failure_reasons = []
            for collision_info in collision_robots:
                if collision_info['type'] == 'robot_robot':
                    failure_reasons.append({
                        'type': 'robot_robot',
                        'robot_id': collision_info['robot_id'],
                        'other_robot_id': collision_info['other_robot_id']
                    })
                elif collision_info['type'] == 'robot_obstacle':
                    robot_id = collision_info['robot_id']
                    obstacle_type = collision_info['obstacle_type']
                    reason = {
                        'type': 'robot_obstacle',
                        'robot_id': robot_id,
                        'obstacle_type': obstacle_type
                    }
                    if obstacle_type in ['circular', 'line', 'polygon']:
                        reason['obstacle_position'] = collision_info['obstacle_position']
                    failure_reasons.append(reason)
            return failure_reasons
        else:
            return [{'type': 'timeout', 'reason': 'Episode reached maximum length'}]
    
    def _print_failure_reasons(self):
        """Print detailed failure reasons for failed episode"""
        failure_reasons = self._get_failure_reasons()
        
        if failure_reasons:
            print("  Failure reasons:")
            for reason in failure_reasons:
                if reason['type'] == 'robot_robot':
                    print(f"    Robot {reason['robot_id']} collided with Robot {reason['other_robot_id']}")
                elif reason['type'] == 'robot_obstacle':
                    robot_id = reason['robot_id']
                    obstacle_type = reason['obstacle_type']
                    if obstacle_type == 'circular':
                        pos = reason['obstacle_position']
                        print(f"    Robot {robot_id} collided with circular obstacle at ({pos[0]:.2f}, {pos[1]:.2f})")
                    elif obstacle_type == 'map':
                        print(f"    Robot {robot_id} collided with map obstacle")
                    elif obstacle_type == 'line':
                        pos = reason['obstacle_position']
                        print(f"    Robot {robot_id} collided with line obstacle from ({pos[0]:.2f}, {pos[1]:.2f}) to ({pos[2]:.2f}, {pos[3]:.2f})")
                    elif obstacle_type == 'polygon':
                        pos = reason['obstacle_position']
                        print(f"    Robot {robot_id} collided with polygon obstacle edge from ({pos[0]:.2f}, {pos[1]:.2f}) to ({pos[2]:.2f}, {pos[3]:.2f})")
                elif reason['type'] == 'timeout':
                    print(f"    {reason['reason']}")
        else:
            print("  No collision information available")