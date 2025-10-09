from ir_sim.env import env_base
from math import sqrt, pi
from gym import spaces
from gym_env.envs.rvo_inter import rvo_inter
import numpy as np
import sys
import os

# Add the rl_rvo_nav directory to the path for imports
rl_rvo_nav_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'rl_rvo_nav')
sys.path.append(rl_rvo_nav_path)

# Long-range navigation modules
try:
    from LongRangeNavi.long_range_config import LongRangeConfig
    from LongRangeNavi.waypoint_manager import WaypointManager
    from LongRangeNavi.global_path_planner import GlobalPathPlanner
except Exception as e:
    print(f"LONG-RANGE IMPORT ERROR: {e}")
    LongRangeConfig = None
    WaypointManager = None
    GlobalPathPlanner = None

class ir_gym(env_base):
    def __init__(self, world_name, neighbors_region=5, neighbors_num=10, vxmax = 1.5, vymax = 1.5, env_train=True, acceler = 0.5, enable_deadlock_resolution=False, **kwargs):
        super(ir_gym, self).__init__(world_name=world_name, **kwargs)

        # self.obs_mode = kwargs.get('obs_mode', 0)    # 0 drl_rvo, 1 drl_nrvo
        # self.reward_mode = kwargs.get('reward_mode', 0)

        self.radius_exp = kwargs.get('radius_exp', 0.2)

        self.env_train = env_train

        self.nr = neighbors_region
        self.nm = neighbors_num

        self.rvo = rvo_inter(neighbors_region, neighbors_num, vxmax, vymax, acceler, env_train, self.radius_exp)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        self.reward_parameter = kwargs.get('reward_parameter', (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)) 
        self.acceler = acceler
        self.arrive_flag_cur = False

        self.rvo_state_dim = 8
        
        # Deadlock resolution module initialization
        self.enable_deadlock_resolution = enable_deadlock_resolution
        self.deadlock_detector = None
        self.mode_controller = None
        self.par_coordinator = None
        self.state_manager = None
        self.par_executor = None
        self.deadlock_config = None
        self.deadlock_logger = None
        
        # Initialize deadlock resolution modules if enabled
        if self.enable_deadlock_resolution:
            self._initialize_deadlock_modules()
        
        # Long-range waypoint navigation (optional)
        self.enable_long_range_nav = bool(kwargs.get('enable_long_range_nav', False))
        self.long_range_config = kwargs.get('long_range_config', LongRangeConfig() if LongRangeConfig else None)
        self._global_planner = None
        self._waypoint_managers = {}
        
        # Debug: Print long-range navigation status (can be removed in production)
        if self.enable_long_range_nav:
            print(f"LONG-RANGE: Navigation enabled with GlobalPathPlanner={GlobalPathPlanner is not None}, WaypointManager={WaypointManager is not None}")
            print(f"LONG-RANGE: Config received: {self.long_range_config}")
            if isinstance(self.long_range_config, dict):
                print(f"LONG-RANGE: grid_resolution from config: {self.long_range_config.get('grid_resolution', 'NOT_FOUND')}")
            else:
                print(f"LONG-RANGE: grid_resolution from config object: {getattr(self.long_range_config, 'grid_resolution', 'NOT_FOUND')}")
        

    def cal_des_omni_list(self):
        des_vel_list = [robot.cal_des_vel_omni() for robot in self.robot_list]
        return des_vel_list

    def _get_combined_obs_lines(self):
        """Combine line obstacles with polygon edges as line segments [x1,y1,x2,y2]."""
        combined_lines = []
        try:
            if 'obs_lines' in self.components and hasattr(self.components['obs_lines'], 'obs_line_states'):
                combined_lines += list(self.components['obs_lines'].obs_line_states)
        except Exception:
            pass

        try:
            if 'obs_polygons' in self.components:
                obs_polys = self.components['obs_polygons']
                if hasattr(obs_polys, 'obs_poly_list'):
                    for poly in obs_polys.obs_poly_list:
                        if hasattr(poly, 'edge_list'):
                            for edge in poly.edge_list:
                                # edge is [x1, y1, x2, y2]
                                if isinstance(edge, (list, tuple)) and len(edge) == 4:
                                    combined_lines.append([edge[0], edge[1], edge[2], edge[3]])
        except Exception:
            pass

        return combined_lines


    def rvo_reward_list_cal(self, action_list, **kwargs):    
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        rvo_reward_list = list(map(lambda robot_state, action: self.rvo_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs), ts[0], action_list))

        return rvo_reward_list
    
    def rvo_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):
        
        # Ensure polygon edges are included as line obstacles for RVO
        combined_lines = self._get_combined_obs_lines()
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(robot_state, nei_state_list, obs_cir_list, combined_lines, action, **kwargs)

        des_vel = np.round(np.squeeze(robot_state[-2:]), 2)
        
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter

        dis_des = sqrt((action[0] - des_vel[0] )**2 + (action[1] - des_vel[1])**2)
        max_dis_des = 3
        dis_des_reward = - dis_des / max_dis_des #  (0-1)
        exp_time_reward = - 0.2/(min_exp_time+0.2) # (0-1)
        
        # rvo reward    
        if vo_flag:
            rvo_reward = p2 + p3 * dis_des_reward + p4 * exp_time_reward
            
            if min_exp_time < 0.1:
                rvo_reward = p2 + p1 * p4 * exp_time_reward
        else:
            rvo_reward = p5 + p6 * dis_des_reward
        
        rvo_reward = np.round(rvo_reward, 2)

        return rvo_reward

    def obs_move_reward_list(self, action_list, **kwargs):
        # Check if deadlock resolution is enabled
        if self.enable_deadlock_resolution:
            return self._step_with_deadlock_resolution(action_list)
        else:
            # Use original pure RL logic
            ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

            # Waypoint progression BEFORE observation/reward to align des_vel with current waypoint
            reached_flags = [False] * len(self.robot_list)
            final_flags = [False] * len(self.robot_list)
            if self.enable_long_range_nav and isinstance(self._waypoint_managers, dict) and len(self._waypoint_managers) == len(self.robot_list):
                try:
                    for aid, robot in enumerate(self.robot_list):
                        # Do not skip PAR agents; they should also progress waypoints when using RL execution
                        mode = self.get_current_mode(aid) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                        if aid in self._waypoint_managers:
                            pos = (float(robot.state[0, 0]), float(robot.state[1, 0])) if hasattr(robot, 'state') else (0.0, 0.0)
                            reached, final_reached = self._waypoint_managers[aid].update(pos)
                            reached_flags[aid] = bool(reached)
                            final_flags[aid] = bool(final_reached)
                            cur_goal = self._waypoint_managers[aid].get_current_goal()
                            if cur_goal is not None:
                                try:
                                    import numpy as np
                                    robot.goal = np.array([[float(cur_goal[0])], [float(cur_goal[1])]])
                                except Exception:
                                    pass
                            # Debug print (angle) for agent 0
                            if aid == 0 and hasattr(self, 'step_count') and cur_goal is not None:
                                if not hasattr(self, '_debug_step_count'):
                                    self._debug_step_count = 0
                                if self._debug_step_count % 100 == 0:
                                    print(f"DEBUG Agent 0: pos={pos}, goal={cur_goal}, reached={reached}, final={final_reached}")
                                    try:
                                        des_vec = np.squeeze(robot.cal_des_vel_omni())
                                        goal_vec = (float(cur_goal[0]) - float(pos[0]), float(cur_goal[1]) - float(pos[1]))
                                        dvx = float(des_vec[0]) if np.size(des_vec) > 0 else 0.0
                                        dvy = float(des_vec[1]) if np.size(des_vec) > 1 else 0.0
                                        gvx = float(goal_vec[0])
                                        gvy = float(goal_vec[1])
                                        des_norm = max(1e-8, (dvx**2 + dvy**2) ** 0.5)
                                        goal_norm = max(1e-8, (gvx**2 + gvy**2) ** 0.5)
                                        dot_val = (dvx * gvx + dvy * gvy) / (des_norm * goal_norm)
                                        dot_val = max(-1.0, min(1.0, dot_val))
                                        angle_deg = float(np.degrees(np.arccos(dot_val)))
                                        print(f"DEBUG Direction: des_vel=({dvx:.3f},{dvy:.3f}), goal_vec=({gvx:.3f},{gvy:.3f}), angle_deg={angle_deg:.1f}")
                                    except Exception:
                                        pass
                                self._debug_step_count += 1
                except Exception as e:
                    print(f"LONG-RANGE STEP ERROR: {e}")

            # Now compute observation and reward using updated robot.goal
            obs_reward_list = list(map(lambda robot, action: self.observation_reward(robot, ts[1], ts[2], ts[3], action, **kwargs), self.robot_list, action_list))

            obs_list = [l[0] for l in obs_reward_list]
            reward_list = [l[1] for l in obs_reward_list]
            done_list = [l[2] for l in obs_reward_list]
            info_list = [l[3] for l in obs_reward_list]

            # Mark done for agents that reached final waypoint this step
            if self.enable_long_range_nav and len(final_flags) == len(done_list):
                for i in range(len(done_list)):
                    if final_flags[i]:
                        done_list[i] = True

            # Log current waypoint goals into info_list and include done flag without updating managers again
            if self.enable_long_range_nav and isinstance(self._waypoint_managers, dict) and len(self._waypoint_managers) == len(self.robot_list):
                try:
                    current_goals = []
                    for aid, robot in enumerate(self.robot_list):
                        mode = self.get_current_mode(aid) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                        if mode == 'par':
                            if hasattr(robot, 'goal') and robot.goal is not None:
                                current_goals.append([float(robot.goal[0]), float(robot.goal[1])])
                            else:
                                current_goals.append([0.0, 0.0])
                            continue
                        if aid in self._waypoint_managers:
                            cur_goal = self._waypoint_managers[aid].get_current_goal()
                            if cur_goal is not None:
                                current_goals.append([float(cur_goal[0]), float(cur_goal[1])])
                            else:
                                current_goals.append([0.0, 0.0])
                        else:
                            current_goals.append([0.0, 0.0])
                    if len(current_goals) > 0:
                        for i, info in enumerate(info_list):
                            if i < len(current_goals):
                                if not isinstance(info, dict):
                                    info_list[i] = {
                                        'current_goal': current_goals[i],
                                        'done': bool(final_flags[i]) if i < len(final_flags) else False
                                    }
                                else:
                                    info['current_goal'] = current_goals[i]
                                    info['done'] = bool(final_flags[i]) if i < len(final_flags) else False
                except Exception as e:
                    print(f"LONG-RANGE STEP ERROR: {e}")

            return obs_list, reward_list, done_list, info_list

    def observation_reward(self, robot, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs):

        robot_omni_state = robot.omni_state()
        des_vel = np.squeeze(robot.cal_des_vel_omni())
       
        done = False

        if robot.arrive() and not robot.arrive_flag:
            robot.arrive_flag = True
            arrive_reward_flag = True
        else:
            arrive_reward_flag = False

        # Check if this robot is in PAR mode (kept for potential logging), but do not skip RVO
        robot_id = self.robot_list.index(robot)
        _ = self.get_current_mode(robot_id) if hasattr(self, 'get_current_mode') else 'rl_rvo'
        
        # Use combined line obstacles (lines + polygon edges) and always run RVO
        combined_lines = self._get_combined_obs_lines()
        obs_vo_list, vo_flag, min_exp_time, collision_flag = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, combined_lines, action, **kwargs)

        # If collision detected by RVO, call collision_check to record detailed collision information
        if collision_flag and not robot.collision_flag:
            robot.collision_check(self.components)

        radian = robot.state[2]
        cur_vel = np.squeeze(robot.vel_omni)
        radius = robot.radius_collision* np.ones(1,)

        propri_obs = np.concatenate([ cur_vel, des_vel, radian, radius]) 
        
        if len(obs_vo_list) == 0:
            exter_obs = np.zeros((self.rvo_state_dim,))
        else:
            exter_obs = np.concatenate(obs_vo_list) # vo list
            
        observation = np.round(np.concatenate([propri_obs, exter_obs]), 2)

        # dis2goal = sqrt( robot.state[0:2] - robot.goal[0:2])
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
        des_vel = np.squeeze(robot.cal_des_vel_omni())
        
        combined_lines = self._get_combined_obs_lines()
        obs_vo_list, _, min_exp_time, _ = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, combined_lines)
    
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
        
        self.components['robots'].robots_reset(reset_mode, **kwargs)
        
        # Reset deadlock detection for new episode
        if self.enable_deadlock_resolution and hasattr(self, 'deadlock_detector') and self.deadlock_detector:
            self.deadlock_detector.reset_episode()
        
        # Reset state manager for new episode
        if self.enable_deadlock_resolution and hasattr(self, 'state_manager') and self.state_manager:
            # Force reset all agents to rl_rvo mode
            num_agents = len(self.robot_list) if hasattr(self, 'robot_list') else 10
            self.state_manager.force_reset_all_agents_to_rl_rvo(num_agents)
        
        # Reset step counter for logger
        if hasattr(self, 'step_count'):
            self.step_count = 0
        else:
            self.step_count = 0
        
        # Long-range navigation: initialize planners and waypoint managers after robots reset
        if self.enable_long_range_nav and GlobalPathPlanner is not None and WaypointManager is not None:
            try:
                # Build occupancy grid from workspace if available via PAR environment helper
                grid, resolution, world_w, world_h = self._build_occupancy_grid_for_long_range()
                print(f"LONG-RANGE: Grid building returned resolution={resolution}")
                if grid is not None:
                    # Get waypoint spacing from config (handle both dict and object)
                    if isinstance(self.long_range_config, dict):
                        waypoint_spacing = self.long_range_config.get('waypoint_min_spacing', 5.0)
                    else:
                        waypoint_spacing = getattr(self.long_range_config, 'waypoint_min_spacing', 5.0)
                    print(f"LONG-RANGE: Initializing GlobalPathPlanner with resolution={resolution}, waypoint_spacing={waypoint_spacing}")
                    print(f"LONG-RANGE: Grid actual dimensions: {len(grid)} x {len(grid[0]) if grid else 0}")
                    self._global_planner = GlobalPathPlanner(grid, resolution, waypoint_spacing)
                    print(f"LONG-RANGE: GlobalPathPlanner initialized with _resolution={self._global_planner._resolution}")
                    print(f"LONG-RANGE: GlobalPathPlanner grid dimensions: {len(self._global_planner._grid)} x {len(self._global_planner._grid[0]) if self._global_planner._grid else 0}")
                    # Initialize per-agent waypoint managers
                    self._waypoint_managers = {}
                    waypoint_data = {}  # Store waypoint data for logging
                    
                    # First, generate waypoints for all agents
                    all_waypoint_lists = []
                    agent_info = []
                    
                    for aid, robot in enumerate(self.robot_list):
                        start_pos = (float(robot.state[0, 0]), float(robot.state[1, 0])) if hasattr(robot, 'state') else (0.0, 0.0)
                        goal = None
                        if hasattr(robot, 'goal') and robot.goal is not None:
                            # Extract only x, y coordinates (ignore theta if present)
                            if hasattr(robot.goal, 'shape') and robot.goal.shape[0] >= 2:
                                goal = (float(robot.goal[0, 0]), float(robot.goal[1, 0]))
                            else:
                                goal = tuple(robot.goal)
                        elif hasattr(robot, 'target') and robot.target is not None:
                            if hasattr(robot.target, 'shape') and robot.target.shape[0] >= 2:
                                goal = (float(robot.target[0, 0]), float(robot.target[1, 0]))
                            else:
                                goal = tuple(robot.target)
                        elif hasattr(robot, 'destination') and robot.destination is not None:
                            if hasattr(robot.destination, 'shape') and robot.destination.shape[0] >= 2:
                                goal = (float(robot.destination[0, 0]), float(robot.destination[1, 0]))
                            else:
                                goal = tuple(robot.destination)
                        if goal is None:
                            goal = (start_pos[0] + 2.0, start_pos[1])
                        
                        waypoints = self._global_planner.plan_path(start_pos, goal)
                        all_waypoint_lists.append(waypoints)
                        agent_info.append((aid, robot, start_pos, goal))
                    
                    # Separate waypoints to avoid conflicts (disabled for debugging)
                    # if isinstance(self.long_range_config, dict):
                    #     min_waypoint_distance = float(self.long_range_config.get('waypoint_separation_manhattan', 1.0))
                    # else:
                    #     min_waypoint_distance = float(getattr(self.long_range_config, 'waypoint_separation_manhattan', 1.0))
                    # print(f"LONG-RANGE: Separating waypoints with min distance {min_waypoint_distance}")
                    # separated_waypoint_lists = self._global_planner.separate_waypoints(all_waypoint_lists, min_waypoint_distance)
                    separated_waypoint_lists = all_waypoint_lists
                    
                    # Debug separation results (disabled)
                    # for i, (original, separated) in enumerate(zip(all_waypoint_lists, separated_waypoint_lists)):
                    #     if original != separated:
                    #         print(f"LONG-RANGE: Agent {i} waypoints separated: {len(original)} -> {len(separated)}")
                    #         print(f"  Original: {original[:3]}...{original[-2:] if len(original) > 3 else ''}")
                    #         print(f"  Separated: {separated[:3]}...{separated[-2:] if len(separated) > 3 else ''}")
                    
                    # Now create waypoint managers with separated waypoints
                    for i, (aid, robot, start_pos, goal) in enumerate(agent_info):
                        waypoints = separated_waypoint_lists[i]
                        # # Remove the first waypoint if it approximately coincides with the start position
                        # try:
                        #     reach_thr = getattr(self.long_range_config, 'reach_threshold', 0.2)
                        #     if isinstance(waypoints, list) and len(waypoints) > 0:
                        #         wx, wy = float(waypoints[0][0]), float(waypoints[0][1])
                        #         sx, sy = float(start_pos[0]), float(start_pos[1])
                        #         if (wx - sx) * (wx - sx) + (wy - sy) * (wy - sy) <= (reach_thr * reach_thr):
                        #             waypoints = waypoints[1:]
                        # except Exception:
                        #     pass
                        reach_thr = getattr(self.long_range_config, 'reach_threshold', 0.2)
                        # Conditionally remove the first waypoint only if it overlaps start and there are >=2 waypoints
                        try:
                            if isinstance(waypoints, list) and len(waypoints) >= 2:
                                wx, wy = float(waypoints[0][0]), float(waypoints[0][1])
                                sx, sy = float(start_pos[0]), float(start_pos[1])
                                if (wx - sx) * (wx - sx) + (wy - sy) * (wy - sy) <= (reach_thr * reach_thr):
                                    waypoints = waypoints[1:]
                        except Exception:
                            pass
                        print(f"LONG-RANGE: Agent {aid} waypoints: {waypoints}")
                        
                        # Convert numpy arrays to lists for JSON serialization
                        def convert_to_serializable(obj):
                            if hasattr(obj, 'tolist'):
                                return obj.tolist()
                            elif isinstance(obj, (list, tuple)):
                                return [convert_to_serializable(item) for item in obj]
                            else:
                                return obj
                        
                        waypoint_data[aid] = {
                            'start_position': convert_to_serializable(start_pos),
                            'final_goal': convert_to_serializable(goal),
                            'waypoints': convert_to_serializable(waypoints)
                        }
                        self._waypoint_managers[aid] = WaypointManager(aid, waypoints, reach_threshold=reach_thr)
                        cur_goal = self._waypoint_managers[aid].get_current_goal()
                        if cur_goal is not None:
                            try:
                                robot.goal = [cur_goal[0], cur_goal[1]]
                            except Exception:
                                pass
                    # Log waypoint data to episode logger if available
                    self._log_waypoint_data(waypoint_data)
                    
                    # Log discretized grid map for debugging
                    self._log_discretized_grid(grid, resolution, world_w, world_h)
            except Exception as e:
                print(f"LONG-RANGE INIT ERROR: {e}")
                self._global_planner = None
                self._waypoint_managers = {}
        
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    def env_reset_one(self, id):
        self.robot_reset(id)

    def env_observation(self):
        ts = self.components['robots'].total_states()
        obs_list = list(map(lambda robot: self.observation(robot, ts[1], ts[2], ts[3]), self.robot_list))

        return obs_list

    @staticmethod
    def wraptopi(theta):

        if theta > pi:
            theta = theta - 2*pi
        
        if theta < -pi:
            theta = theta + 2*pi

        return theta
    
    # Deadlock resolution methods
    def _initialize_deadlock_modules(self):
        """Initialize deadlock resolution modules."""
        try:
            # Add the correct path for imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rl_rvo_nav_dir = os.path.join(current_dir, '..', '..', '..')
            if rl_rvo_nav_dir not in sys.path:
                sys.path.insert(0, rl_rvo_nav_dir)
            
            from config.deadlock_config import DeadlockConfig
            from deadlock_resolution.deadlock_detector import DeadlockDetector
            from mode_management.mode_controller import ModeController
            from mode_management.state_manager import StateManager
            from deadlock_resolution.par_coordinator import PARCoordinator
            from deadlock_resolution.par_executor import PARExecutor
            from python_pnr.push_and_rotate import PushAndRotate
            
            # Initialize configuration
            self.deadlock_config = DeadlockConfig()
            
            # Initialize PNR solver
            pnr_solver = PushAndRotate()
            
            # Initialize deadlock logger first
            try:
                from rl_rvo_nav.policy_test_with_deadlock.deadlock_logger import get_deadlock_logger
                self.deadlock_logger = get_deadlock_logger()
                print("Deadlock logger initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize deadlock logger: {e}")
                self.deadlock_logger = None
            
            # Initialize deadlock resolution modules with logger
            self.deadlock_detector = DeadlockDetector(self.deadlock_config.config)
            # Set logger for deadlock detector
            if hasattr(self.deadlock_detector, 'set_logger'):
                self.deadlock_detector.set_logger(self.deadlock_logger)
            elif hasattr(self.deadlock_detector, 'logger'):
                self.deadlock_detector.logger = self.deadlock_logger
            
            self.par_coordinator = PARCoordinator(pnr_solver, self.deadlock_config.config, gym_env=self)
            # Set logger for PAR coordinator
            if hasattr(self.par_coordinator, 'set_logger'):
                self.par_coordinator.set_logger(self.deadlock_logger)
            elif hasattr(self.par_coordinator, 'logger'):
                self.par_coordinator.logger = self.deadlock_logger
            
            self.par_executor = PARExecutor(self.deadlock_config.config)
            self.state_manager = StateManager()
            self.mode_controller = ModeController(
                self.deadlock_detector, 
                self.par_coordinator, 
                self.deadlock_config.config
            )
            # Inject dependencies into PARExecutor
            if hasattr(self.par_executor, 'set_dependencies'):
                self.par_executor.set_dependencies(self.state_manager, self.par_coordinator)
            
            print("Deadlock resolution modules initialized successfully")
            
        except ImportError as e:
            print(f"Warning: Could not import deadlock resolution modules: {e}")
            print("Deadlock resolution will be disabled")
            self.enable_deadlock_resolution = False
    
    def enable_deadlock_resolution_mode(self, config_file=None):
        """Enable deadlock resolution mode for testing."""
        if not self.enable_deadlock_resolution:
            self.enable_deadlock_resolution = True
            if config_file and self.deadlock_config:
                self.deadlock_config.load_from_file(config_file)
            self._initialize_deadlock_modules()
            print("Deadlock resolution mode enabled")
    
    def disable_deadlock_resolution_mode(self):
        """Disable deadlock resolution mode (restore pure RL)."""
        self.enable_deadlock_resolution = False
        print("Deadlock resolution mode disabled")
    
    def get_current_mode(self, agent_id):
        """Get current mode of an agent."""
        if not self.enable_deadlock_resolution or not self.state_manager:
            return 'rl_rvo'
        return self.state_manager.get_agent_mode(agent_id)
    
    def is_in_deadlock_resolution_mode(self):
        """Check if in deadlock resolution mode."""
        return self.enable_deadlock_resolution
    
    def _step_with_deadlock_resolution(self, action_list):
        """Execute step with deadlock resolution logic."""
        if not self.enable_deadlock_resolution:
            return self._step_pure_rl(action_list)
        
        # Increment step counter
        if hasattr(self, 'step_count'):
            self.step_count += 1
        else:
            self.step_count = 1
        
        # Increment deadlock detector step counter (once per step, not per agent)
        if hasattr(self, 'deadlock_detector') and self.deadlock_detector:
            self.deadlock_detector.step_counter = self.step_count
        
        # Get current states
        ts = self.components['robots'].total_states()
        agent_states = self._get_agent_states_dict(ts[0])
        neighbor_states = self._get_neighbor_states_dict(ts[1])
        
        # Log step start if logger is available
        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
            self.deadlock_logger.log_step_start(self.step_count, agent_states, neighbor_states)
        
        # Process each agent
        modified_action_list = action_list.copy()
        
        # Debug: Track which agents are being processed
        print(f"DEBUG: Step {self.step_count} - Processing {len(action_list)} agents")
        
        for agent_id, action in enumerate(action_list):
            current_mode = self.get_current_mode(agent_id)
            print(f"DEBUG: Agent {agent_id} - Current mode: {current_mode}")
            
            # Check for deadlock and mode switching
            if current_mode == 'rl_rvo':
                print(f"DEBUG: Agent {agent_id} in rl_rvo mode, checking deadlock")
                # Check if should switch to PAR mode
                agent_neighbor_states = self._get_agent_neighbor_states(agent_id, agent_states, neighbor_states)
                if self.deadlock_detector.detect_deadlock(agent_id, agent_states, agent_neighbor_states):
                    print(f"DEBUG: Agent {agent_id} triggered deadlock detection")
                    deadlock_participants = self.deadlock_detector.get_deadlock_participants(agent_id, agent_states, agent_neighbor_states)
                    print(f"DEBUG: Agent {agent_id} deadlock participants: {deadlock_participants}")
                    
                    # Participants are fully determined by detector (unified mode). Upper-layer TTC/Jaccard logic removed.
                    
                    # Only proceed if we have multiple confirmed participants
                    if len(deadlock_participants) > 1:
                        try:
                            par_solution = self.par_coordinator.prepare_par_execution(agent_states, deadlock_participants)
                            
                            # Validate solution per participant: require non-empty path or already at goal
                            valid_participants = []
                            tol = 0.5
                            try:
                                tol = self.deadlock_config.get('GOAL_TOLERANCE') if self.deadlock_config else 0.5
                            except Exception:
                                tol = 0.5
                            for pid in deadlock_participants:
                                agent_state = agent_states.get(pid, {})
                                path = self.par_executor.get_agent_path_from_solution(pid, par_solution) if hasattr(self, 'par_executor') else None
                                has_path = bool(path) and len(path) > 0
                                at_goal = False
                                if 'position' in agent_state and 'goal' in agent_state and agent_state['goal'] is not None:
                                    pos = agent_state['position']
                                    goal = agent_state['goal']
                                    if isinstance(pos, (list, tuple)) and isinstance(goal, (list, tuple)) and len(pos) >= 2 and len(goal) >= 2:
                                        gx = goal[0][0] if isinstance(goal[0], list) and len(goal[0]) > 0 else goal[0]
                                        gy = goal[1][0] if isinstance(goal[1], list) and len(goal[1]) > 0 else goal[1]
                                        dx = float(pos[0]) - float(gx)
                                        dy = float(pos[1]) - float(gy)
                                        at_goal = (dx*dx + dy*dy) ** 0.5 <= float(tol)
                                if has_path or at_goal:
                                    valid_participants.append(pid)
                            
                            # Allow switching if we have a core subset (>=2) with valid paths
                            if len(valid_participants) < 2:
                                continue
                            
                            # Log PAR preparation with agent positions
                            if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                                # Log agent positions when switching from RL to MAPF
                                agent_positions = {}
                                for participant_id in valid_participants:
                                    if participant_id in agent_states:
                                        agent_state = agent_states[participant_id]
                                        if 'position' in agent_state:
                                            agent_positions[participant_id] = agent_state['position']
                                
                                self.deadlock_logger.log_par_preparation(agent_id, valid_participants, par_solution)
                                self.deadlock_logger.log_rl_to_mapf_positions(agent_positions)
                                self.deadlock_logger.log_par_solution_paths(par_solution, valid_participants)
                            
                            # Set PAR mode for all valid participants and initialize PAR waypoints
                            for participant_id in valid_participants:
                                old_mode = self.get_current_mode(participant_id)
                                self.state_manager.set_par_mode(participant_id, par_solution)
                                
                                # Build PAR waypoints and inject into WaypointManager
                                grid_path = []
                                try:
                                    grid_path = self.par_coordinator.get_agent_path(participant_id)
                                    print(f"PAR INIT: Agent {participant_id} grid path length: {len(grid_path)}")
                                    if grid_path:
                                        print(f"  Grid path: {grid_path[:3]}...{grid_path[-3:] if len(grid_path) > 6 else grid_path[3:]}")
                                except Exception as e:
                                    print(f"PAR INIT: Failed to get grid path for agent {participant_id}: {e}")
                                    grid_path = []
                                
                                cont_path = []
                                if grid_path and hasattr(self.par_coordinator, 'par_environment') and self.par_coordinator.par_environment and hasattr(self.par_coordinator.par_environment, 'grid_to_continuous'):
                                    for gp in grid_path:
                                        try:
                                            if hasattr(gp, 'x') and hasattr(gp, 'y'):
                                                grid_coord = (gp.x, gp.y)
                                            else:
                                                grid_coord = gp
                                            cont = self.par_coordinator.par_environment.grid_to_continuous(grid_coord)
                                            cont_path.append(cont)
                                        except Exception as e:
                                            print(f"PAR INIT: Failed to convert grid point {gp}: {e}")
                                            pass
                                else:
                                    print(f"PAR INIT: Cannot convert path for agent {participant_id} - missing environment or grid_to_continuous method")
                                
                                # Inject as waypoints if available
                                if cont_path and isinstance(self._waypoint_managers, dict):
                                    try:
                                        tol = 0.5
                                        try:
                                            tol = self.deadlock_config.get('GOAL_TOLERANCE') if self.deadlock_config else 0.5
                                        except Exception:
                                            tol = 0.5
                                        # Ensure save buffer exists
                                        if not hasattr(self, '_saved_lr_managers'):
                                            self._saved_lr_managers = {}
                                        # Save current LR manager if not already a PAR manager
                                        if participant_id in self._waypoint_managers:
                                            cur_mgr = self._waypoint_managers[participant_id]
                                            if not hasattr(cur_mgr, '_is_par_manager'):
                                                self._saved_lr_managers[participant_id] = cur_mgr
                                        # Replace manager for this agent with PAR waypoints and mark as PAR
                                        par_mgr = WaypointManager(participant_id, cont_path, reach_threshold=tol)
                                        setattr(par_mgr, '_is_par_manager', True)
                                        self._waypoint_managers[participant_id] = par_mgr
                                        cur_goal = par_mgr.get_current_goal()
                                        if cur_goal is not None:
                                            import numpy as np
                                            self.robot_list[participant_id].goal = np.array([[float(cur_goal[0])], [float(cur_goal[1])]])
                                        print(f"PAR INIT: Injected {len(cont_path)} continuous waypoints for agent {participant_id}")
                                    except Exception as e:
                                        print(f"PAR INIT: Failed to inject waypoints for agent {participant_id}: {e}")
                                
                                # Log mode switch
                                if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                                    self.deadlock_logger.log_mode_switch(participant_id, old_mode, 'par', f"Deadlock detected by agent {agent_id}")
                            
                            # print(f"🔴 DEADLOCK DETECTED: Agent {agent_id} triggered deadlock resolution, switching {len(deadlock_participants)} agents to PAR mode")
                            # print(f"   Participants: {deadlock_participants}")
                        except Exception as e:
                            print(f"❌ PAR preparation failed for agent {agent_id}: {e}")
                            if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                                self.deadlock_logger.log_error("PAR_PREPARATION", e, {"agent_id": agent_id, "participants": deadlock_participants})
                            # Continue with RL_RVO mode if PAR fails
                else:
                    print(f"DEBUG: Agent {agent_id} did not trigger deadlock detection")
            else:
                print(f"DEBUG: Agent {agent_id} not in rl_rvo mode, skipping deadlock detection")
            
            # Execute action based on current mode: PAR agents no longer override actions; RL path only
        
        # BEFORE dynamics: apply deferred PAR position sets with safety check, so RVO/collision sees updated positions
        # Removed legacy PAR set_position collision handling and overrides

        # Before executing RL dynamics, align waypoint goal so observation uses current waypoint
        if self.enable_long_range_nav and isinstance(self._waypoint_managers, dict) and len(self._waypoint_managers) == len(self.robot_list):
            try:
                for aid, robot in enumerate(self.robot_list):
                    # Do not skip PAR agents; they should also progress waypoints when using RL execution
                    mode = self.get_current_mode(aid) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                    if aid in self._waypoint_managers:
                        pos = (float(robot.state[0, 0]), float(robot.state[1, 0])) if hasattr(robot, 'state') else (0.0, 0.0)
                        # Progress waypoint once before observation/reward
                        reached, final_reached = self._waypoint_managers[aid].update(pos)
                        cur_goal = self._waypoint_managers[aid].get_current_goal()
                        if cur_goal is not None:
                            try:
                                import numpy as np
                                robot.goal = np.array([[float(cur_goal[0])], [float(cur_goal[1])]])
                            except Exception:
                                pass
                        # PAR quick diagnostics: print per-step tracking status
                        if mode == 'par':
                            try:
                                import numpy as np
                                des_vec = np.squeeze(robot.cal_des_vel_omni())
                                dvx = float(des_vec[0]) if np.size(des_vec) > 0 else 0.0
                                dvy = float(des_vec[1]) if np.size(des_vec) > 1 else 0.0
                                if cur_goal is not None:
                                    gvx = float(cur_goal[0]) - float(pos[0])
                                    gvy = float(cur_goal[1]) - float(pos[1])
                                    des_norm = max(1e-8, (dvx * dvx + dvy * dvy) ** 0.5)
                                    goal_norm = max(1e-8, (gvx * gvx + gvy * gvy) ** 0.5)
                                    dot_val = (dvx * gvx + dvy * gvy) / (des_norm * goal_norm)
                                    dot_val = max(-1.0, min(1.0, dot_val))
                                    angle_deg = float(np.degrees(np.arccos(dot_val)))
                                    goal_dist = float((gvx * gvx + gvy * gvy) ** 0.5)
                                    print(f"PAR DEBUG Agent {aid}: pos={pos}, goal={cur_goal}, des_vel=({dvx:.3f},{dvy:.3f}), angle_deg={angle_deg:.1f}, speed={des_norm:.3f}, goal_dist={goal_dist:.3f}, reached={reached}, final={final_reached}")
                                else:
                                    print(f"PAR DEBUG Agent {aid}: pos={pos}, goal=None, des_vel=({dvx:.3f},{dvy:.3f}), speed={(dvx*dvx + dvy*dvy) ** 0.5:.3f}, reached={reached}, final={final_reached}")
                            except Exception:
                                pass
                # Mark that we've progressed waypoint in this step to skip inside _step_pure_rl
                self._wp_progressed_in_step = True
            except Exception:
                pass

        # Execute the modified actions using pure RL logic (removed PAR set_position/yield/distance overrides)

        # Apply speed cap for PAR agents to improve tracking of PAR waypoints
        try:
            par_speed_cap = 0.1
            try:
                par_speed_cap = float(self.deadlock_config.get('PAR_TRACK_SPEED_LIMIT', 0.3)) if self.deadlock_config else 0.3
            except Exception:
                par_speed_cap = 0.3
            for aid2 in range(len(modified_action_list)):
                mode2 = self.get_current_mode(aid2) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                if mode2 != 'par':
                    continue
                act = modified_action_list[aid2]
                if act is None:
                    continue
                try:
                    import numpy as _np
                    vec = _np.asarray(act, dtype=float)
                    if vec.size >= 2:
                        vx, vy = float(vec[0]), float(vec[1])
                        speed = (vx * vx + vy * vy) ** 0.5
                        if speed > par_speed_cap and speed > 1e-8:
                            scale = par_speed_cap / speed
                            vx *= scale
                            vy *= scale
                            modified_action_list[aid2] = _np.array([vx, vy], dtype=float)
                except Exception:
                    pass
        except Exception:
            pass

        # DEBUG: PAR positions before _step_pure_rl removed (RL-only execution)

        # Now run dynamics with possibly adjusted actions
        obs_list, reward_list, done_list, info_list = self._step_pure_rl(modified_action_list)

        # DEBUG: PAR positions after _step_pure_rl removed (RL-only execution)

        # Ensure info_list carries 'done' based on waypoint final flags (align success criteria)
        try:
            if self.enable_long_range_nav and isinstance(self._waypoint_managers, dict) and len(self._waypoint_managers) == len(self.robot_list):
                # Build current final flags without updating managers again
                final_flags = []
                current_goals = []
                for aid, robot in enumerate(self.robot_list):
                    mode = self.get_current_mode(aid) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                    if mode == 'par':
                        # In PAR mode, keep current goal and mark not done here (PAR handles completion)
                        if hasattr(robot, 'goal') and robot.goal is not None:
                            current_goals.append([float(robot.goal[0]), float(robot.goal[1])])
                        else:
                            current_goals.append([0.0, 0.0])
                        final_flags.append(False)
                        continue
                    if aid in self._waypoint_managers:
                        cur_goal = self._waypoint_managers[aid].get_current_goal()
                        is_final = (cur_goal is None)
                        final_flags.append(bool(is_final))
                        if cur_goal is not None:
                            current_goals.append([float(cur_goal[0]), float(cur_goal[1])])
                        else:
                            current_goals.append([0.0, 0.0])
                    else:
                        final_flags.append(False)
                        current_goals.append([0.0, 0.0])

                # Attach to info_list
                for i, info in enumerate(info_list):
                    if not isinstance(info, dict):
                        info_list[i] = {
                            'current_goal': current_goals[i] if i < len(current_goals) else [0.0, 0.0],
                            'done': bool(final_flags[i]) if i < len(final_flags) else False
                        }
                    else:
                        info['current_goal'] = current_goals[i] if i < len(current_goals) else [0.0, 0.0]
                        info['done'] = bool(final_flags[i]) if i < len(final_flags) else False
        except Exception:
            pass

        # If all current PAR agents have completed their waypoint managers, switch them back to RL mode and restore LR managers
        try:
            current_par_agents = []
            for aid2 in range(len(self.robot_list)):
                mode2 = self.get_current_mode(aid2) if hasattr(self, 'get_current_mode') else 'rl_rvo'
                if mode2 == 'par':
                    current_par_agents.append(aid2)
            if len(current_par_agents) > 0 and isinstance(self._waypoint_managers, dict):
                all_complete = True
                for aid2 in current_par_agents:
                    if aid2 not in self._waypoint_managers:
                        all_complete = False
                        break
                    cur_goal2 = self._waypoint_managers[aid2].get_current_goal()
                    if cur_goal2 is not None:
                        all_complete = False
                        break
                if all_complete:
                    for pid in current_par_agents:
                        old_mode = self.get_current_mode(pid) if hasattr(self, 'get_current_mode') else 'par'
                        self.state_manager.set_rl_rvo_mode(pid)
                        # Restore original LR waypoint manager if saved
                        try:
                            if hasattr(self, '_saved_lr_managers') and pid in self._saved_lr_managers:
                                self._waypoint_managers[pid] = self._saved_lr_managers[pid]
                                del self._saved_lr_managers[pid]
                                # Update robot.goal to restored manager current goal
                                cur_goal_restored = self._waypoint_managers[pid].get_current_goal()
                                if cur_goal_restored is not None:
                                    import numpy as np
                                    self.robot_list[pid].goal = np.array([[float(cur_goal_restored[0])], [float(cur_goal_restored[1])]])
                        except Exception:
                            pass
                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                            self.deadlock_logger.log_mode_switch(pid, old_mode, 'rl_rvo', 'PAR participants completed (manager-final)')
        except Exception:
            pass

        return obs_list, reward_list, done_list, info_list
    
    def _step_pure_rl(self, action_list):
        """Execute step with pure RL logic (original behavior)."""
        # Enforce boundary constraints before robot movement
        action_list = self._enforce_boundaries(action_list)
        
        # DEBUG: Log actions before robot_step
        print(f"ROBOT_STEP DEBUG: About to call robot_step with actions: {action_list}")
        
        # Execute robot movement with the modified actions
        self.robot_step(action_list, vel_type='omni', stop=True)
        self.obs_cirs_step()
        
        # DEBUG: Log actions before observation_reward
        print(f"OBSERVATION_REWARD DEBUG: About to call observation_reward with actions: {action_list}")
        
        # This is the original step logic
        ts = self.components['robots'].total_states()
        obs_reward_list = list(map(lambda robot, action: self.observation_reward(robot, ts[1], ts[2], ts[3], action), self.robot_list, action_list))
        
        obs_list = [l[0] for l in obs_reward_list]
        reward_list = [l[1] for l in obs_reward_list]
        done_list = [l[2] for l in obs_reward_list]
        info_list = [l[3] for l in obs_reward_list]
        
        # Long-range waypoint progression moved earlier in deadlock path; avoid double-update
        if hasattr(self, '_wp_progressed_in_step') and self._wp_progressed_in_step:
            try:
                del self._wp_progressed_in_step
            except Exception:
                pass
        
        return obs_list, reward_list, done_list, info_list
    
    def _get_agent_states_dict(self, robot_state_list):
        """Convert robot state list to dictionary format."""
        agent_states = {}
        for i, robot_state in enumerate(robot_state_list):
            # Get goal position from robot
            goal = None
            if hasattr(self.robot_list[i], 'goal') and self.robot_list[i].goal is not None:
                goal = self.robot_list[i].goal
            elif hasattr(self.robot_list[i], 'target') and self.robot_list[i].target is not None:
                goal = self.robot_list[i].target
            elif hasattr(self.robot_list[i], 'destination') and self.robot_list[i].destination is not None:
                goal = self.robot_list[i].destination
            
            # If no goal found, use a default goal (e.g., current position + offset)
            if goal is None:
                current_pos = robot_state[0:2]
                # Create a simple goal: move 2 units in x direction
                goal = [current_pos[0] + 2.0, current_pos[1]]
                print(f"⚠️ Agent {i}: No goal found, using default goal: {goal}")
            
            agent_states[i] = {
                'position': robot_state[0:2],
                'velocity': robot_state[2:4],
                'goal': goal
            }
        return agent_states
    
    def _get_neighbor_states_dict(self, nei_state_list):
        """Build nested neighbor dict per agent using agent positions (distance-based).
        Returns: {agent_id: {neighbor_id: {position, velocity}}}
        """
        nested = {}
        # Reuse current agent states to compute neighbors by distance
        try:
            ts = self.components['robots'].total_states()
            agent_states = self._get_agent_states_dict(ts[0])
        except Exception:
            agent_states = {}
        # Radius from config
        try:
            radius = float(self.deadlock_config.get('COLLISION_WARNING_DISTANCE', 2.0)) if self.deadlock_config else 2.0
        except Exception:
            radius = 2.0
        for agent_id, state in agent_states.items():
            cur = state.get('position')
            if not isinstance(cur, (list, tuple, np.ndarray)) or len(cur) < 2:
                nested[agent_id] = {}
                continue
            cx, cy = float(cur[0]), float(cur[1])
            neighbors = {}
            for other_id, other in agent_states.items():
                if other_id == agent_id:
                    continue
                pos = other.get('position')
                if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) < 2:
                    continue
                ox, oy = float(pos[0]), float(pos[1])
                dx = ox - cx
                dy = oy - cy
                if dx*dx + dy*dy <= radius * radius:
                    neighbors[other_id] = {
                        'position': other['position'],
                        'velocity': other.get('velocity', [0.0, 0.0])
                    }
            nested[agent_id] = neighbors
        return nested

    # --- Long-range helper: build occupancy grid consistent with PAR workspace ---
    def _build_occupancy_grid_for_long_range(self):
        try:
            # Read grid_resolution from config (same as PAR)
            try:
                if isinstance(self.long_range_config, dict):
                    resolution = float(self.long_range_config.get('grid_resolution', 0.5))
                else:
                    resolution = float(getattr(self.long_range_config, 'grid_resolution', 0.5))
                print(f"LONG-RANGE GRID: Read grid_resolution={resolution} from config")
            except Exception as e:
                resolution = 0.5
                print(f"LONG-RANGE GRID: Failed to read grid_resolution, using default={resolution}, error={e}")
            
            # Get world bounds (same as PAR workspace bounds)
            world_w = float(getattr(self, '_env_base__width', 10))
            world_h = float(getattr(self, '_env_base__height', 10))
            offset_x = float(getattr(self, 'offset_x', 0.0))
            offset_y = float(getattr(self, 'offset_y', 0.0))
            
            # Check for map_matrix (same as PAR)
            map_matrix = None
            if hasattr(self, 'components') and isinstance(self.components, dict):
                map_matrix = self.components.get('map_matrix', None)
            
            # If we have map_matrix, use it directly (same as PAR)
            if map_matrix is not None:
                import numpy as _np
                arr = _np.array(map_matrix)
                bin_grid = (arr != 0).astype(int).tolist()
                print(f"LONG-RANGE GRID MAP: Using map_matrix with user-configured resolution={resolution}")
                # Optional: inflate obstacles by dilation (post-raster step)
                try:
                    enable_dilate = True
                    dilate_iters = 1
                    if isinstance(self.long_range_config, dict):
                        enable_dilate = bool(self.long_range_config.get('enable_obstacle_dilation', True))
                        dilate_iters = int(self.long_range_config.get('obstacle_dilation_cells', 1))
                    else:
                        enable_dilate = bool(getattr(self.long_range_config, 'enable_obstacle_dilation', True))
                        dilate_iters = int(getattr(self.long_range_config, 'obstacle_dilation_cells', 1))
                    if enable_dilate and dilate_iters > 0:
                        bin_grid = self._dilate_obstacle_grid(bin_grid, dilate_iters)
                        print(f"LONG-RANGE: Applied obstacle dilation (iters={dilate_iters}) on map_matrix grid")
                except Exception:
                    pass
                return bin_grid, resolution, int(world_w), int(world_h)
            
            # Build grid from workspace bounds (exactly like PAR)
            import math as _math
            bx0, by0 = offset_x, offset_y
            bx1, by1 = offset_x + world_w, offset_y + world_h
            res = resolution
            
            # Calculate grid dimensions (same as PAR)
            full_w = max(1, int(_math.floor((bx1 - bx0) / res)))
            full_h = max(1, int(_math.floor((by1 - by0) / res)))
            print(f"LONG-RANGE GRID: World bounds: ({bx0}, {by0}) to ({bx1}, {by1})")
            print(f"LONG-RANGE GRID: Grid dimensions: {full_w} x {full_h}")
            
            # Initialize grid (same as PAR)
            grid = [[0 for _ in range(full_w)] for _ in range(full_h)]
            
            # Set up coordinate system (same as PAR)
            min_x, min_y = bx0, by0
            max_x, max_y = bx1, by1
            
            # Get obstacles (same as PAR)
            obstacles = self._get_environment_obstacles_for_long_range()
            print(f"LONG-RANGE: Found {len(obstacles)} obstacles")
            
            # Rasterize obstacles using exact PAR algorithm
            if obstacles:
                self._populate_obstacles_in_grid_par_style(grid, obstacles, res, min_x, min_y, full_w, full_h)
                print(f"LONG-RANGE: Populated {len(obstacles)} obstacles into grid")
            else:
                print(f"LONG-RANGE: No obstacles found to populate")
            
            # Optional: inflate obstacles by dilation (post-raster step)
            try:
                enable_dilate = True
                dilate_iters = 1
                if isinstance(self.long_range_config, dict):
                    enable_dilate = bool(self.long_range_config.get('enable_obstacle_dilation', True))
                    dilate_iters = int(self.long_range_config.get('obstacle_dilation_cells', 1))
                else:
                    enable_dilate = bool(getattr(self.long_range_config, 'enable_obstacle_dilation', True))
                    dilate_iters = int(getattr(self.long_range_config, 'obstacle_dilation_cells', 1))
                if enable_dilate and dilate_iters > 0:
                    grid = self._dilate_obstacle_grid(grid, dilate_iters)
                    print(f"LONG-RANGE: Applied obstacle dilation (iters={dilate_iters}) on analytic grid")
            except Exception:
                pass

            return grid, resolution, int(world_w), int(world_h)
        except Exception as e:
            print(f"LONG-RANGE GRID ERROR: {e}")
            return None, None

    def _dilate_obstacle_grid(self, grid, iterations=1):
        """Apply 8-neighborhood binary dilation to obstacle grid for N iterations.
        1-marked cells are obstacles. Returns a new grid (same shape).
        """
        try:
            if grid is None or iterations <= 0:
                return grid
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            if h == 0 or w == 0:
                return grid
            import copy as _copy
            cur = [row[:] for row in grid]
            nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
            for _ in range(int(iterations)):
                nxt = [row[:] for row in cur]
                for i in range(h):
                    rowi = cur[i]
                    for j in range(w):
                        if rowi[j] == 1:
                            # mark neighbors as obstacle
                            for di,dj in nbrs:
                                ni = i + di
                                nj = j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    nxt[ni][nj] = 1
                cur = nxt
            return cur
        except Exception:
            return grid

    def _get_environment_obstacles_for_long_range(self):
        """Get obstacles from the environment using the same method as PAR."""
        try:
            obstacles = []
            if hasattr(self, 'components') and isinstance(self.components, dict):
                comp = self.components
                # Polygons
                try:
                    if 'obs_polygons' in comp and hasattr(comp['obs_polygons'], 'obs_poly_list'):
                        obstacles.extend(list(comp['obs_polygons'].obs_poly_list))
                except Exception:
                    pass
                # Circles
                try:
                    if 'obs_circles' in comp and hasattr(comp['obs_circles'], 'obs_cir_list'):
                        obstacles.extend(list(comp['obs_circles'].obs_cir_list))
                except Exception:
                    pass
                # If an explicit obstacles list exists, include it as well
                try:
                    if 'obstacles' in comp and isinstance(comp['obstacles'], list):
                        obstacles.extend(comp['obstacles'])
                except Exception:
                    pass
            
            return obstacles
            
        except Exception as e:
            print(f"LONG-RANGE OBSTACLE ERROR: {e}")
            return []
    
    def _populate_obstacles_in_grid_par_style(self, grid, obstacles, resolution, min_x, min_y, grid_width, grid_height):
        """Populate obstacles in grid using exact PAR algorithm."""
        for obstacle in obstacles:
            self._add_obstacle_to_grid_par_style(grid, obstacle, resolution, min_x, min_y, grid_width, grid_height)
    
    def _add_obstacle_to_grid_par_style(self, grid, obstacle, resolution, min_x, min_y, grid_width, grid_height):
        """Add a single obstacle to the grid using exact PAR algorithm."""
        try:
            if hasattr(obstacle, 'pos') and hasattr(obstacle, 'radius'):
                # Circular obstacle (same as PAR)
                center_x, center_y = obstacle.pos[0], obstacle.pos[1]
                radius = obstacle.radius
                self._add_circular_obstacle_par_style(grid, center_x, center_y, radius, resolution, min_x, min_y, grid_width, grid_height)
                
            elif hasattr(obstacle, 'vertices'):
                # Polygon obstacle (same as PAR)
                vertices = obstacle.vertices
                self._add_polygon_obstacle_par_style(grid, vertices, resolution, min_x, min_y, grid_width, grid_height)
            elif hasattr(obstacle, 'vertexes'):
                # Polygon obstacle (ir_sim obs_polygon uses 'vertexes' 2xN) (same as PAR)
                try:
                    verts = obstacle.vertexes
                    # Expect ndarray shape (2, N)
                    if hasattr(verts, 'shape') and len(verts.shape) == 2 and verts.shape[0] == 2:
                        vertices = [(float(verts[0, i]), float(verts[1, i])) for i in range(verts.shape[1])]
                    else:
                        # Fallback: attempt to iterate columns
                        vertices = [(float(v[0]), float(v[1])) for v in getattr(obstacle, 'vertexes')]
                    self._add_polygon_obstacle_par_style(grid, vertices, resolution, min_x, min_y, grid_width, grid_height)
                except Exception:
                    pass
                
            elif isinstance(obstacle, (list, tuple)) and len(obstacle) >= 2:
                # Point obstacle (same as PAR)
                x, y = obstacle[0], obstacle[1]
                self._add_point_obstacle_par_style(grid, x, y, resolution, min_x, min_y, grid_width, grid_height)
                
        except Exception as e:
            pass
    
    def _add_circular_obstacle_par_style(self, grid, center_x, center_y, radius, resolution, min_x, min_y, grid_width, grid_height):
        """Add a circular obstacle to the grid (exact PAR algorithm)."""
        # Convert to grid coordinates (same as PAR)
        grid_center_x = int((center_x - min_x) / resolution)
        grid_center_y = int((center_y - min_y) / resolution)
        grid_radius = int(radius / resolution) + 1
        
        # Mark grid cells within the circle as obstacles (same as PAR)
        for i in range(max(0, grid_center_y - grid_radius), min(grid_height, grid_center_y + grid_radius + 1)):
            for j in range(max(0, grid_center_x - grid_radius), min(grid_width, grid_center_x + grid_radius + 1)):
                # Check if cell is within circle (same as PAR)
                if (i - grid_center_y) ** 2 + (j - grid_center_x) ** 2 <= grid_radius ** 2:
                    if 0 <= i < grid_height and 0 <= j < grid_width:
                        grid[i][j] = 1  # Mark as obstacle
    
    def _add_polygon_obstacle_par_style(self, grid, vertices, resolution, min_x, min_y, grid_width, grid_height):
        """Add a polygon obstacle to the grid using exact PAR algorithm."""
        if len(vertices) < 3:
            return
        
        # Use grid cell overlap method (same as PAR)
        self._fill_polygon_grid_cell_overlap_par_style(grid, vertices, resolution, min_x, min_y, grid_width, grid_height)
    
    def _add_point_obstacle_par_style(self, grid, x, y, resolution, min_x, min_y, grid_width, grid_height):
        """Add a point obstacle to the grid (exact PAR algorithm)."""
        grid_x = int((x - min_x) / resolution)
        grid_y = int((y - min_y) / resolution)
        
        if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
            grid[grid_y][grid_x] = 1  # Mark as obstacle
    
    def _fill_polygon_grid_cell_overlap_par_style(self, grid, vertices, resolution, min_x, min_y, grid_width, grid_height):
        """Fill polygon area using exact PAR grid cell overlap method."""
        # Calculate polygon bounding box in continuous coordinates (same as PAR)
        min_poly_x = min(v[0] for v in vertices)
        max_poly_x = max(v[0] for v in vertices)
        min_poly_y = min(v[1] for v in vertices)
        max_poly_y = max(v[1] for v in vertices)
        
        # Calculate grid cell range that might overlap with polygon (same as PAR)
        grid_min_x = max(0, int((min_poly_x - min_x) / resolution))
        grid_max_x = min(grid_width, int((max_poly_x - min_x) / resolution) + 1)
        grid_min_y = max(0, int((min_poly_y - min_y) / resolution))
        grid_max_y = min(grid_height, int((max_poly_y - min_y) / resolution) + 1)
        
        filled_count = 0
        for i in range(grid_min_y, grid_max_y):
            for j in range(grid_min_x, grid_max_x):
                # Calculate grid cell corners in continuous coordinates (same as PAR)
                cell_x = min_x + j * resolution
                cell_y = min_y + i * resolution
                cell_corners = [
                    (cell_x, cell_y),                           # bottom-left
                    (cell_x + resolution, cell_y),              # bottom-right
                    (cell_x, cell_y + resolution),              # top-left
                    (cell_x + resolution, cell_y + resolution)  # top-right
                ]
                
                # Check if grid cell overlaps with polygon (same as PAR)
                if self._grid_cell_overlaps_polygon_par_style(cell_corners, vertices):
                    grid[i][j] = 1  # Mark as obstacle
                    filled_count += 1
    
    def _grid_cell_overlaps_polygon_par_style(self, cell_corners, vertices):
        """Check if a grid cell overlaps with polygon (exact PAR algorithm)."""
        # Check if any corner is inside the polygon (same as PAR)
        corners_inside = 0
        corners_on_boundary = 0
        
        for corner in cell_corners:
            inside, on_boundary = self._point_in_polygon_with_boundary_par_style(corner[0], corner[1], vertices)
            if inside and not on_boundary:
                corners_inside += 1
            elif on_boundary:
                corners_on_boundary += 1
        
        # If any corner is inside (not on boundary), the cell overlaps (same as PAR)
        if corners_inside > 0:
            return True
        
        # If all corners are on boundary, the cell doesn't overlap (same as PAR)
        if corners_inside == 0 and corners_on_boundary == 4:
            return False
        
        # Only mark as obstacle if there are corners inside the polygon (same as PAR)
        return corners_inside > 0
    
    def _point_in_polygon_with_boundary_par_style(self, x, y, vertices):
        """Check if a point is inside a polygon, with boundary detection (exact PAR algorithm)."""
        n = len(vertices)
        inside = False
        on_boundary = False
        
        # Check if point is exactly on a vertex (same as PAR)
        for vx, vy in vertices:
            if abs(x - vx) < 1e-10 and abs(y - vy) < 1e-10:
                return True, True
        
        # Ray casting algorithm with boundary detection (same as PAR)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            # Check if point is on the edge
            if min(xi, xj) <= x <= max(xi, xj) and min(yi, yj) <= y <= max(yi, yj):
                # Check if point is on the line
                if abs((y - yi) * (xj - xi) - (x - xi) * (yj - yi)) < 1e-10:
                    on_boundary = True
            
            # Ray casting
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        
        return inside, on_boundary

    def _populate_obstacles_in_grid(self, grid, obstacles, resolution, world_width, world_height):
        """Populate the grid with obstacles using the same method as PAR."""
        try:
            for i, obstacle in enumerate(obstacles):
                print(f"LONG-RANGE: Processing obstacle {i}: {type(obstacle)}")
                self._add_obstacle_to_grid(grid, obstacle, resolution, world_width, world_height)
        except Exception as e:
            print(f"LONG-RANGE OBSTACLE POPULATION ERROR: {e}")
    
    def _add_obstacle_to_grid(self, grid, obstacle, resolution, world_width, world_height):
        """Add a single obstacle to the grid using the same method as PAR."""
        try:
            if hasattr(obstacle, 'pos') and hasattr(obstacle, 'radius'):
                # Circular obstacle
                center_x, center_y = obstacle.pos[0], obstacle.pos[1]
                radius = obstacle.radius
                print(f"LONG-RANGE: Adding circular obstacle at ({center_x}, {center_y}) with radius {radius}")
                self._add_circular_obstacle_to_grid(grid, center_x, center_y, radius, resolution, world_width, world_height)
                
            elif hasattr(obstacle, 'vertices'):
                # Polygon obstacle
                vertices = obstacle.vertices
                print(f"LONG-RANGE: Adding polygon obstacle with {len(vertices)} vertices: {vertices}")
                self._add_polygon_obstacle_to_grid(grid, vertices, resolution, world_width, world_height)
            elif hasattr(obstacle, 'vertexes'):
                # Polygon obstacle (ir_sim obs_polygon uses 'vertexes' 2xN)
                try:
                    verts = obstacle.vertexes
                    # Expect ndarray shape (2, N)
                    if hasattr(verts, 'shape') and len(verts.shape) == 2 and verts.shape[0] == 2:
                        vertices = [(float(verts[0, i]), float(verts[1, i])) for i in range(verts.shape[1])]
                    else:
                        # Fallback: attempt to iterate columns
                        vertices = [(float(v[0]), float(v[1])) for v in getattr(obstacle, 'vertexes')]
                    print(f"LONG-RANGE: Adding polygon obstacle (vertexes) with {len(vertices)} vertices: {vertices}")
                    self._add_polygon_obstacle_to_grid(grid, vertices, resolution, world_width, world_height)
                except Exception:
                    print(f"LONG-RANGE: Failed to process obstacle with vertexes")
                    pass
                    
        except Exception as e:
            print(f"LONG-RANGE OBSTACLE ADD ERROR: {e}")
    
    def _add_circular_obstacle_to_grid(self, grid, center_x, center_y, radius, resolution, world_width, world_height):
        """Add a circular obstacle to the grid using the same method as PAR."""
        # Convert to grid coordinates with proper rounding
        grid_center_x = int(round(center_x / resolution))
        grid_center_y = int(round(center_y / resolution))
        grid_radius = int(round(radius / resolution)) + 1
        
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0
        
        # Mark grid cells within the circle as obstacles
        for i in range(max(0, grid_center_y - grid_radius), min(grid_height, grid_center_y + grid_radius + 1)):
            for j in range(max(0, grid_center_x - grid_radius), min(grid_width, grid_center_x + grid_radius + 1)):
                # Check if cell is within circle
                if (i - grid_center_y) ** 2 + (j - grid_center_x) ** 2 <= grid_radius ** 2:
                    if 0 <= i < grid_height and 0 <= j < grid_width:
                        grid[i][j] = 1  # Mark as obstacle
    
    def _add_polygon_obstacle_to_grid(self, grid, vertices, resolution, world_width, world_height):
        """Add a polygon obstacle to the grid using the same method as PAR."""
        if len(vertices) < 3:
            return
        
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid_height > 0 else 0
        
        # Convert vertices to grid coordinates using floor to preserve obstacle thickness
        grid_vertices = [(int(v[0] / resolution), int(v[1] / resolution)) for v in vertices]
        print(f"LONG-RANGE: Grid vertices: {grid_vertices}")
        
        # Find bounding box
        min_x = min(v[0] for v in grid_vertices)
        max_x = max(v[0] for v in grid_vertices)
        min_y = min(v[1] for v in grid_vertices)
        max_y = max(v[1] for v in grid_vertices)
        print(f"LONG-RANGE: Bounding box: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        # Mark grid cells within the polygon as obstacles
        obstacle_count = 0
        for i in range(max(0, min_y), min(grid_height, max_y + 1)):
            for j in range(max(0, min_x), min(grid_width, max_x + 1)):
                if self._point_in_polygon(j, i, grid_vertices):
                    grid[i][j] = 1  # Mark as obstacle
                    obstacle_count += 1
        
        # PATCH: Ensure obstacles starting from x=0.0 occupy grid column index 0
        for v in vertices:
            if v[0] == 0.0:  # If obstacle starts from x=0.0
                for i in range(max(0, min_y), min(grid_height, max_y + 1)):
                    if grid[i][0] == 0:  # If column 0 cell is not already marked
                        grid[i][0] = 1
                        obstacle_count += 1
                break  # Only need to check once per obstacle

        # PATCH: Ensure obstacles starting from y=0.0 occupy grid row index 0
        for v in vertices:
            if v[1] == 0.0:  # If obstacle starts from y=0.0
                for j in range(max(0, min_x), min(grid_width, max_x + 1)):
                    if grid[0][j] == 0:  # If row 0 cell is not already marked
                        grid[0][j] = 1
                        obstacle_count += 1
                break  # Only need to check once per obstacle
        
        print(f"LONG-RANGE: Marked {obstacle_count} grid cells as obstacles")
    
    def _point_in_polygon(self, x, y, vertices):
        """Check if a point is inside a polygon using ray casting algorithm."""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _add_map_boundaries_as_obstacles(self, grid, world_width, world_height, resolution):
        """Add map boundaries as obstacles to prevent path planning outside the map.
        
        This method is now deprecated. Boundary checking is handled dynamically
        in the GlobalPathPlanner._is_grid_position_valid method.
        """
        try:
            # Note: We no longer pre-mark boundaries as obstacles.
            # Instead, boundary checking is done dynamically during path planning.
            print(f"LONG-RANGE: Boundary checking is now handled dynamically (grid: {len(grid[0])}x{len(grid)}, world: {world_width}x{world_height})")
            
        except Exception as e:
            print(f"LONG-RANGE BOUNDARY ERROR: {e}")

    def _log_waypoint_data(self, waypoint_data):
        """Log waypoint data to episode logger if available."""
        try:
            # Try to find and use the test logger from post_train
            if hasattr(self, '_test_logger') and self._test_logger:
                # Add waypoint data to current episode data
                if hasattr(self._test_logger, 'current_episode_data'):
                    self._test_logger.current_episode_data['waypoint_data'] = waypoint_data
                    print(f"LONG-RANGE: Logged waypoint data for {len(waypoint_data)} agents")
            else:
                # Try to find logger through environment chain
                if hasattr(self, 'env') and hasattr(self.env, 'test_logger'):
                    self.env.test_logger.current_episode_data['waypoint_data'] = waypoint_data
                    print(f"LONG-RANGE: Logged waypoint data for {len(waypoint_data)} agents")
        except Exception as e:
            print(f"LONG-RANGE LOGGING ERROR: {e}")

    def _log_discretized_grid(self, grid, resolution, world_width, world_height):
        """Log discretized grid map for debugging."""
        try:
            if hasattr(self, '_test_logger') and self._test_logger:
                print(f"DEBUG: Logging discretized grid map")
                
                # Create grid info
                grid_info = {
                    'grid_resolution': resolution,
                    'world_width': world_width,
                    'world_height': world_height,
                    'grid_width': len(grid[0]) if grid else 0,
                    'grid_height': len(grid) if grid else 0,
                    'grid_data': grid
                }
                
                # Add to episode data
                if hasattr(self._test_logger, 'current_episode_data'):
                    self._test_logger.current_episode_data['discretized_grid'] = grid_info
                    print(f"LONG-RANGE: Logged discretized grid map")
                
                # Print grid visualization for debugging
                print(f"LONG-RANGE GRID MAP:")
                print(f"  World: {world_width}x{world_height}, Grid: {len(grid[0])}x{len(grid)}, Resolution: {resolution}")
                print(f"  Grid visualization (0=free, 1=obstacle):")
                for i, row in enumerate(grid):
                    row_str = ''.join(['0' if cell == 0 else '1' for cell in row])
                    print(f"    Row {i:2d}: {row_str}")
                    
            else:
                print(f"DEBUG: No test logger available for grid map")
        except Exception as e:
            print(f"LONG-RANGE GRID LOG ERROR: {e}")

    def _get_agent_neighbor_states(self, agent_id, agent_states, neighbor_states_nested):
        """Return neighbor state mapping for a specific agent.
        Prefer precomputed neighbor_states; if empty, build from agent_states by distance.
        """
        # Use precomputed neighbors if available
        precomputed = neighbor_states_nested.get(agent_id, {}) if isinstance(neighbor_states_nested, dict) else {}
        if isinstance(precomputed, dict) and len(precomputed) > 0:
            return precomputed
        
        # Build from agent_states based on distance threshold
        result = {}
        if agent_id not in agent_states:
            return result
        current = agent_states[agent_id]
        if 'position' not in current or current['position'] is None:
            return result
        cx, cy = current['position'][0], current['position'][1]
        # Use collision warning distance as neighbor radius
        try:
            radius = float(self.deadlock_config.get('COLLISION_WARNING_DISTANCE', 2.0)) if self.deadlock_config else 2.0
        except Exception:
            radius = 2.0
        for other_id, other in agent_states.items():
            if other_id == agent_id:
                continue
            if 'position' not in other or other['position'] is None:
                continue
            ox, oy = other['position'][0], other['position'][1]
            dx = ox - cx
            dy = oy - cy
            dist_sq = dx*dx + dy*dy
            if dist_sq <= radius * radius:
                result[other_id] = {
                    'position': other['position'],
                    'velocity': other.get('velocity', [0.0, 0.0])
                }
        return result
    
    def _force_par_agent_exit(self, agent_id):
        """
        Force a PAR agent to exit PAR mode and switch back to RL mode.
        
        Args:
            agent_id: ID of the agent to force exit
        """
        try:
            if self.state_manager:
                # Switch agent back to RL mode
                self.state_manager.set_rl_rvo_mode(agent_id)
                print(f"FORCED PAR EXIT: Agent {agent_id} switched from PAR to RL mode")
                
                # Clear any PAR execution state
                if hasattr(self, 'par_executor') and self.par_executor:
                    if hasattr(self.par_executor, 'agent_paths') and agent_id in self.par_executor.agent_paths:
                        del self.par_executor.agent_paths[agent_id]
                    if hasattr(self.par_executor, 'agent_substep_index') and agent_id in self.par_executor.agent_substep_index:
                        del self.par_executor.agent_substep_index[agent_id]
                
                # Legacy set_position queue no longer used
                        
        except Exception as e:
            print(f"Error forcing PAR agent {agent_id} exit: {e}")
    
    def _enforce_boundaries(self, action_list):
        """限制动作，防止机器人移动到地图边界外"""
        try:
            # 获取地图边界
            world_width = getattr(self, '_env_base__width', 15)  # 默认15米
            world_height = getattr(self, '_env_base__height', 10)  # 默认10米
            
            # 检查每个机器人的动作
            for i, robot in enumerate(self.robot_list):
                if i < len(action_list) and hasattr(robot, 'state') and robot.state is not None:
                    current_x = float(robot.state[0, 0])
                    current_y = float(robot.state[1, 0])
                    action = action_list[i]
                    
                    if action is not None and len(action) >= 2:
                        # 预测下一步位置
                        dt = getattr(self, 'step_time', 0.1)
                        next_x = current_x + action[0] * dt
                        next_y = current_y + action[1] * dt
                        
                        # 检查是否会超出边界
                        if next_x < 0 or next_x > world_width or next_y < 0 or next_y > world_height:
                            # 限制动作，使下一步位置在边界内
                            max_vx = (world_width - current_x) / dt if next_x > world_width else (0 - current_x) / dt if next_x < 0 else action[0]
                            max_vy = (world_height - current_y) / dt if next_y > world_height else (0 - current_y) / dt if next_y < 0 else action[1]
                            
                            # 更新动作
                            action_list[i] = [max_vx, max_vy]
                            
                            # print(f"BOUNDARY ENFORCE: Agent {i} action limited from [{action[0]:.3f}, {action[1]:.3f}] to [{max_vx:.3f}, {max_vy:.3f}] (bounds: [0,{world_width}] x [0,{world_height}])")
        except Exception as e:
            print(f"Error enforcing boundaries: {e}")
        
        return action_list
    