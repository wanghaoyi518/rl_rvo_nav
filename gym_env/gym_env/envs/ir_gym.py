from ir_sim.env import env_base
from math import sqrt, pi
from gym import spaces
from gym_env.envs.rvo_inter import rvo_inter
from gym_env.envs.safe_distance_inter import SafeDistanceInter
import numpy as np

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 1.5, vymax = 1.5, env_train=True, acceler = 0.5, **kwargs):
        super(ir_gym, self).__init__(world_name=world_name, **kwargs)

        self.radius_exp = kwargs.get('radius_exp', 0.2)
        self.env_train = env_train
        self.nr = neighbors_region
        self.nm = neighbors_num

        # Add navigation mode parameter
        self.nav_mode = kwargs.get('nav_mode', 'rvo')  # 'rvo' or 'safe_distance'
        
        # Initialize both RVO and safe distance handlers
        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax, acceler, env_train, self.radius_exp)
        self.safe_distance = SafeDistanceInter(neighbors_region, neighbors_num, vxmax, vymax, acceler, env_train, self.radius_exp)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        self.reward_parameter = kwargs.get('reward_parameter', (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)) 
        self.acceler = acceler
        self.arrive_flag_cur = False
        self.ctime_threshold = kwargs.get('ctime_threshold', 2.0)  # Default collision time threshold of 2.0 seconds

        self.rvo_state_dim = 8
        self.safe_distance_state_dim = 6
        
        # Curriculum learning parameters for random obstacles
        self.curriculum_level = kwargs.get('curriculum_level', 0)
        self.obs_curriculum_enable = kwargs.get('obs_curriculum_enable', False)

    def cal_des_omni_list(self):
        des_vel_list = [robot.cal_des_vel_omni() for robot in self.robot_list]
        return des_vel_list

    def rvo_reward_list_cal(self, action_list, **kwargs):    
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        if self.nav_mode == 'rvo':
            rvo_reward_list = [self.rvo_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs) 
                             for robot_state, action in zip(ts[0], action_list)]
        else:  # safe_distance mode
            rvo_reward_list = [self.safe_distance_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs) 
                             for robot_state, action in zip(ts[0], action_list)]

        return rvo_reward_list

    def rvo_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(robot_state, nei_state_list, obs_cir_list, obs_line_list, action, **kwargs)
        
        des_vel = np.round(np.squeeze(robot_state[-2:]), 2)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0])**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = -dis_des / max_dis_des  # (0-1)
        
        # RVO reward
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward
            
            if min_exp_time < self.ctime_threshold:
                rvo_reward = p2 + p1 * p4 * (min_exp_time / self.ctime_threshold)
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)
        
        return rvo_reward

    def safe_distance_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):
        _, collision_flag, min_dis = self.safe_distance.config_safe_distance_inf(robot_state, nei_state_list, obs_cir_list, obs_line_list, action, **kwargs)
        
        des_vel = np.round(np.squeeze(robot_state[-2:]), 2)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0])**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = -dis_des / max_dis_des  # (0-1)
        
        # Safe distance reward
        if collision_flag:
            safe_reward = p2 + p3 * dis_des_reward
            
            if min_dis < self.radius_exp:
                safe_reward = p2 + p1 * p4 * (min_dis / self.radius_exp)
        else:
            safe_reward = p5 + p6 * dis_des_reward
        
        safe_reward = np.round(safe_reward, 2)
        
        return safe_reward

    def observation_reward(self, robot, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs):
        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot.cal_des_vel_omni())
       
        done = False

        if robot.arrive() and not robot.arrive_flag:
            robot.arrive_flag = True
            arrive_reward_flag = True
        else:
            arrive_reward_flag = False

        if self.nav_mode == 'rvo':
            obs_vo_list, vo_flag, min_exp_time, collision_flag = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs)
        else:  # safe_distance mode
            obs_vo_list, collision_flag, min_dis = self.safe_distance.config_safe_distance_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs)
            min_exp_time = min_dis  # Use minimum distance as time to collision

        radian = robot.state[2]
        cur_vel = np.squeeze(robot.vel_omni)
        radius = robot.radius_collision * np.ones(1,)

        propri_obs = np.concatenate([cur_vel, des_vel, radian, radius]) 
        
        if len(obs_vo_list) == 0:
            if self.nav_mode == 'rvo':
                exter_obs = np.zeros((self.rvo_state_dim,))
            else:
                exter_obs = np.zeros((self.safe_distance_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list)
            
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        mov_reward = self.mov_reward(collision_flag, arrive_reward_flag, self.reward_parameter, min_exp_time)

        reward = mov_reward

        done = True if collision_flag else False
        info = True if robot.arrive_flag else False
        
        return [observation, reward, done, info]

    def mov_reward(self, collision_flag, arrive_reward_flag, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 15), min_exp_time=100, dis2goal=100):

        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        collision_reward = p7 if collision_flag else 0
        arrive_reward = p8 if arrive_reward_flag else 0
        time_reward = 0
        
        mov_reward = collision_reward + arrive_reward + time_reward

        return mov_reward

    def osc_reward(self, state_list):
        # to avoid oscillation
        dif_rad_list = []
        
        if len(state_list) < 3:
            return 0

        for i in range(len(state_list) - 1):
            dif = ir_gym.wraptopi(state_list[i+1][2, 0] - state_list[i][2, 0])
            dif_rad_list.append(round(dif, 2))

        for j in range(len(dif_rad_list)-3):
            
            if dif_rad_list[j] * dif_rad_list[j+1] < -0.05 and dif_rad_list[j+1] * dif_rad_list[j+2] < -0.05 and dif_rad_list[j+2] * dif_rad_list[j+3] < -0.05:
                print('osc', dif_rad_list[j], dif_rad_list[j+1], dif_rad_list[j+2], dif_rad_list[j+3])
                return -10
        return 0

    def observation(self, robot, nei_state_list, obs_circular_list, obs_line_list):

        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot_omni_state[-2:])
        
        obs_vo_list, _, min_exp_time, _ = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list)
    
        cur_vel = np.squeeze(robot.vel_omni)
        radian = robot.state[2]
        radius = robot.radius_collision* np.ones(1,)

        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list

        
        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        return observation

    def env_reset(self, reset_mode=1, **kwargs):
        # Call parent class reset to handle random obstacles regeneration
        super(ir_gym, self).reset(**kwargs)
        
        self.components['robots'].robots_reset(reset_mode, **kwargs)
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list
    
    def update_curriculum_level(self, new_level):
        """Update curriculum learning level for dynamic obstacle complexity adjustment."""
        self.curriculum_level = new_level
        if hasattr(self, 'components') and 'obs_random' in self.components and self.components['obs_random'] is not None:
            # Update obstacle complexity based on curriculum level
            # This can be used to dynamically adjust obstacle generation parameters
            pass
    
    def get_obstacle_info(self):
        """Get current obstacle information for curriculum learning monitoring."""
        if hasattr(self, 'components') and 'obs_random' in self.components:
            random_info = self.get_random_obstacles_info()
            total_static_obstacles = len(self.obs_cir_list) + len(self.obs_poly_list) + len(self.obs_line_states)
            return {
                'random_obstacles': random_info,
                'total_static_obstacles': total_static_obstacles,
                'curriculum_level': self.curriculum_level
            }
        return None

    def env_reset_one(self, id):
        self.robot_reset(id)

    def env_observation(self):
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    def obs_move_reward_list(self, action_list, **kwargs):
        """Calculate observations, rewards, done flags, and info for all robots"""
        ts = self.components['robots'].total_states()  # robot_state_list, nei_state_list, obs_circular_list, obs_line_list
        
        # Calculate observations and rewards for each robot
        results = [self.observation_reward(robot, ts[1], ts[2], ts[3], action, **kwargs) 
                  for robot, action in zip(self.robot_list, action_list)]
        
        # Unzip the results
        obs_list, reward_list, done_list, info_list = zip(*results)
        
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    