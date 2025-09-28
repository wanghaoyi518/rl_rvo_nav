from ir_sim.env import env_base
from math import sqrt, pi
from gym import spaces
from gym_env.envs.rvo_inter import rvo_inter
import numpy as np
import sys
import os

# Add the rl_rvo_nav directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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

            obs_reward_list = list(map(lambda robot, action: self.observation_reward(robot, ts[1], ts[2], ts[3], action, **kwargs), self.robot_list, action_list))

            obs_list = [l[0] for l in obs_reward_list]
            reward_list = [l[1] for l in obs_reward_list]
            done_list = [l[2] for l in obs_reward_list]
            info_list = [l[3] for l in obs_reward_list]

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

        # Check if this robot is in PAR mode
        robot_id = self.robot_list.index(robot)
        current_mode = self.get_current_mode(robot_id) if hasattr(self, 'get_current_mode') else 'rl_rvo'
        
        # Use combined line obstacles (lines + polygon edges)
        combined_lines = self._get_combined_obs_lines()
        
        # In PAR mode, skip RVO processing to avoid overriding PAR actions
        if current_mode == 'par':
            print(f"PAR MODE: Skipping RVO for robot {robot_id}, using PAR action directly")
            obs_vo_list = []
            vo_flag = False
            min_exp_time = float('inf')
            collision_flag = False
        else:
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
        des_vel = np.squeeze(robot_omni_state[-2:])
        
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
                            
                            # Set PAR mode for all valid participants and initialize PAR executor
                            for participant_id in valid_participants:
                                old_mode = self.get_current_mode(participant_id)
                                self.state_manager.set_par_mode(participant_id, par_solution)
                                
                                # Initialize PAR executor with solution data
                                if par_solution and hasattr(par_solution, 'agents_moves'):
                                    # Set start position (use current continuous position to avoid long pre-align phase)
                                    agent_state = agent_states.get(participant_id, {})
                                    if 'position' in agent_state:
                                        start_pos = agent_state['position']
                                        if len(start_pos) >= 2:
                                            self.par_executor.set_agent_start_position(participant_id, (float(start_pos[0]), float(start_pos[1])))
                                    
                                    # Set goal position (use original goal)
                                    if 'goal' in agent_state and agent_state['goal'] is not None:
                                        goal = agent_state['goal']
                                        if isinstance(goal, list) and len(goal) >= 2:
                                            goal_x = goal[0][0] if isinstance(goal[0], list) and len(goal[0]) > 0 else goal[0]
                                            goal_y = goal[1][0] if isinstance(goal[1], list) and len(goal[1]) > 0 else goal[1]
                                            self.par_executor.set_agent_goal_position(participant_id, (float(goal_x), float(goal_y)))
                                    
                                    # Use coordinator path (grid) and map to continuous coordinates
                                    grid_path = []
                                    try:
                                        grid_path = self.par_coordinator.get_agent_path(participant_id)
                                        print(f"PAR INIT: Agent {participant_id} grid path length: {len(grid_path)}")
                                        if grid_path:
                                            print(f"  Grid path: {grid_path[:3]}...{grid_path[-3:] if len(grid_path) > 6 else grid_path[3:]}")
                                    except Exception as e:
                                        print(f"PAR INIT: Failed to get grid path for agent {participant_id}: {e}")
                                        grid_path = []
                                    
                                    if grid_path and hasattr(self.par_coordinator, 'par_environment') and self.par_coordinator.par_environment and hasattr(self.par_coordinator.par_environment, 'grid_to_continuous'):
                                        cont_path = []
                                        for gp in grid_path:
                                            try:
                                                # Convert Point object to (x, y) tuple if needed
                                                if hasattr(gp, 'x') and hasattr(gp, 'y'):
                                                    grid_coord = (gp.x, gp.y)
                                                else:
                                                    grid_coord = gp
                                                
                                                cont = self.par_coordinator.par_environment.grid_to_continuous(grid_coord)
                                                cont_path.append(cont)
                                            except Exception as e:
                                                print(f"PAR INIT: Failed to convert grid point {gp}: {e}")
                                                pass
                                        if cont_path:
                                            self.par_executor.set_agent_path(participant_id, cont_path)
                                            print(f"PAR INIT: Set continuous path for agent {participant_id}, length: {len(cont_path)}")
                                            print(f"  Continuous path: {cont_path[:3]}...{cont_path[-3:] if len(cont_path) > 6 else cont_path[3:]}")
                                        else:
                                            print(f"PAR INIT: No continuous path generated for agent {participant_id}")
                                    else:
                                        print(f"PAR INIT: Cannot convert path for agent {participant_id} - missing environment or grid_to_continuous method")
                                
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
            
            # Execute action based on current mode
            if current_mode == 'par':
                try:
                    par_action = self.par_executor.execute_par_step(agent_id, agent_states)
                    if par_action and 'action' in par_action:
                        modified_action_list[agent_id] = par_action['action']
                        print(f"PAR ACTION APPLIED: Agent {agent_id} action: {par_action['action']}")
                        
                        # Handle direct position setting if specified
                        if 'set_position' in par_action:
                            # Defer actual write until after dynamics update to avoid being overwritten
                            if not hasattr(self, '_par_last_set_positions'):
                                self._par_last_set_positions = {}
                            self._par_last_set_positions[agent_id] = {
                                'pos': par_action['set_position'],
                                'path_index': par_action.get('path_index'),
                                'path_length': par_action.get('path_length')
                            }
                            print(f"PAR POSITION QUEUED: Agent {agent_id} -> {par_action['set_position']}")
                        
                        # Log PAR execution
                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                            self.deadlock_logger.log_par_execution(agent_id, par_action['action'], 'executing')
                    
                    # Group exit policy: only exit PAR when ALL current PAR participants are complete
                    progress = self.par_executor.get_path_progress(agent_id) if hasattr(self.par_executor, 'get_path_progress') else {
                        'is_complete': False
                    }
                    all_participants_complete = False
                    try:
                        current_par_agents = []
                        for aid2 in range(len(self.robot_list)):
                            if self.get_current_mode(aid2) == 'par':
                                current_par_agents.append(aid2)
                        if len(current_par_agents) > 0:
                            all_participants_complete = True
                            for pid in current_par_agents:
                                other_progress = self.par_executor.get_path_progress(pid) if hasattr(self.par_executor, 'get_path_progress') else {'is_complete': False}
                                if not other_progress.get('is_complete', False):
                                    all_participants_complete = False
                                    break
                    except Exception:
                        all_participants_complete = False

                    if progress.get('is_complete', False) and all_participants_complete:
                        old_mode = self.get_current_mode(agent_id)
                        self.state_manager.set_rl_rvo_mode(agent_id)
                        
                        # Log agent positions when switching from MAPF back to RL
                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                            agent_positions = {}
                            for participant_id in current_par_agents:
                                if participant_id in agent_states:
                                    agent_state = agent_states[participant_id]
                                    if 'position' in agent_state:
                                        agent_positions[participant_id] = agent_state['position']
                            
                            self.deadlock_logger.log_mode_switch(agent_id, old_mode, 'rl_rvo', "All PAR participants completed")
                            agent_idx = progress.get('current_index', 0)
                            self.deadlock_logger.log_par_completion(agent_id, agent_idx)
                            self.deadlock_logger.log_mapf_to_rl_positions(agent_positions)
                except Exception as e:
                    print(f"❌ PAR execution failed for agent {agent_id}: {e}")
                    if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                        self.deadlock_logger.log_error("PAR_EXECUTION", e, {"agent_id": agent_id})
                        self.deadlock_logger.log_par_execution(agent_id, None, 'failure')
                    
                    # Fall back to RL_RVO mode
                    old_mode = self.get_current_mode(agent_id)
                    self.state_manager.set_rl_rvo_mode(agent_id)
                    if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                        self.deadlock_logger.log_mode_switch(agent_id, old_mode, 'rl_rvo', "PAR execution failed")
        
        # BEFORE dynamics: apply deferred PAR position sets with safety check, so RVO/collision sees updated positions
        try:
            if hasattr(self, '_par_last_set_positions') and isinstance(self._par_last_set_positions, dict):
                # Init per-agent blocked counters and cooldown maps
                if not hasattr(self, '_par_blocked_counts'):
                    self._par_blocked_counts = {}
                if not hasattr(self, '_par_cooldown_until'):
                    self._par_cooldown_until = {}
                # thresholds
                K_block_steps = 3
                cooldown_steps = 10
                # Cache current positions for collision check
                current_positions = {}
                radii = {}
                for idx, r in enumerate(self.robot_list):
                    try:
                        current_positions[idx] = (float(r.state[0, 0]), float(r.state[1, 0]))
                        radii[idx] = float(getattr(r, 'radius_collision', 0.2))
                    except Exception:
                        continue
                applied_positions = {}

                # Disabled: Immediate early-exit if current overlap already exists
                # This was causing premature PAR exits. PAR algorithm should handle overlaps.
                # try:
                #     for aid in range(len(self.robot_list)):
                #         try:
                #             if self.get_current_mode(aid) != 'par':
                #                 continue
                #         except Exception:
                #             continue
                #         r_i = radii.get(aid, 0.2)
                #         pos_i = current_positions.get(aid)
                #         if pos_i is None:
                #             continue
                #         overlapped = False
                #         for other_id in range(len(self.robot_list)):
                #             if other_id == aid:
                #                 continue
                #             pos_j = current_positions.get(other_id)
                #             if pos_j is None:
                #                 continue
                #             r_j = radii.get(other_id, 0.2)
                #             dx = pos_i[0] - pos_j[0]
                #             dy = pos_i[1] - pos_j[1]
                #             # Add tolerance to avoid premature exit due to minor overlaps
                #             overlap_tolerance = 0.1  # 10cm tolerance
                #             if (dx*dx + dy*dy) ** 0.5 < (r_i + r_j + overlap_tolerance):
                #                 overlapped = True
                #                 blocker = other_id
                #                 break
                #         if overlapped:
                #             # Early exit PAR immediately for this agent
                #             try:
                #                 old_mode = self.get_current_mode(aid)
                #                 if hasattr(self, 'step_count'):
                #                     self._par_cooldown_until[aid] = int(self.step_count) + cooldown_steps
                #                 if hasattr(self, 'state_manager') and self.state_manager:
                #                     self.state_manager.set_rl_rvo_mode(aid)
                #                 if hasattr(self, 'par_executor') and hasattr(self.par_executor, 'reset_agent'):
                #                     self.par_executor.reset_agent(aid)
                #                 if aid in self._par_last_set_positions:
                #                     del self._par_last_set_positions[aid]
                #                 if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                #                     self.deadlock_logger.log_mode_switch(aid, old_mode, 'rl_rvo', 'par_early_exit_overlap')
                #                 print(f"PAR EARLY EXIT: Agent {aid} -> rl_rvo due to existing overlap with agent {blocker}")
                #             except Exception:
                #                 pass
                # except Exception:
                #     pass

                for aid, meta in list(self._par_last_set_positions.items()):
                    pos = meta['pos'] if isinstance(meta, dict) else meta
                    if aid < len(self.robot_list) and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        proposed = (float(pos[0]), float(pos[1]))
                        # Collision check vs all agents (use applied pos if available, else current)
                        blocked = False
                        blocker = None
                        r_i = radii.get(aid, 0.2)
                        for other_id in range(len(self.robot_list)):
                            if other_id == aid:
                                continue
                            other_pos = applied_positions.get(other_id, current_positions.get(other_id))
                            if other_pos is None:
                                continue
                            r_j = radii.get(other_id, 0.2)
                            dx = proposed[0] - other_pos[0]
                            dy = proposed[1] - other_pos[1]
                            if (dx*dx + dy*dy) ** 0.5 < (r_i + r_j):
                                blocked = True
                                blocker = other_id
                                break

                        if blocked:
                            # In PAR algorithm, blocking should be handled by push-and-rotate mechanism
                            # Don't roll back path index - let PAR algorithm resolve the conflict
                            print(f"PAR BLOCKED: Agent {aid} at waypoint {meta.get('path_index', '?')} by agent {blocker}, but continuing PAR execution")
                            
                            # Even if blocked, still set position to prevent RL dynamics from modifying it
                            # Keep the agent at its current PAR position to maintain PAR trajectory
                            if 'set_position' in par_action:
                                if not hasattr(self, '_par_last_set_positions'):
                                    self._par_last_set_positions = {}
                                # Use current position instead of planned position to avoid collision
                                current_pos = self.get_agent_position(agent_states.get(aid, {}))
                                if current_pos is not None:
                                    self._par_last_set_positions[aid] = {
                                        'pos': current_pos,
                                        'path_index': meta.get('path_index'),
                                        'path_length': meta.get('path_length')
                                    }
                                    print(f"PAR BLOCKED POSITION: Agent {aid} kept at current position {current_pos}")
                            # Only count/early-exit if blocker is RL; if blocker is PAR, do not trigger early exit
                            is_rl_blocker = True
                            try:
                                is_rl_blocker = (self.get_current_mode(blocker) != 'par')
                            except Exception:
                                is_rl_blocker = True

                            if is_rl_blocker:
                                # Increase blocked count; if exceed threshold, early-exit PAR
                                self._par_blocked_counts[aid] = int(self._par_blocked_counts.get(aid, 0)) + 1
                                if self._par_blocked_counts[aid] >= K_block_steps:
                                    try:
                                        old_mode = self.get_current_mode(aid)
                                        # Cooldown until
                                        if hasattr(self, 'step_count'):
                                            self._par_cooldown_until[aid] = int(self.step_count) + cooldown_steps
                                        # Switch back to RL
                                        if hasattr(self, 'state_manager') and self.state_manager:
                                            self.state_manager.set_rl_rvo_mode(aid)
                                        # Reset executor state for this agent
                                        if hasattr(self, 'par_executor') and hasattr(self.par_executor, 'reset_agent'):
                                            self.par_executor.reset_agent(aid)
                                        # Remove any queued set for this agent
                                        if aid in self._par_last_set_positions:
                                            del self._par_last_set_positions[aid]
                                        # Log
                                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                                            self.deadlock_logger.log_mode_switch(aid, old_mode, 'rl_rvo', 'par_early_exit_blocked')
                                        print(f"PAR EARLY EXIT: Agent {aid} -> rl_rvo due to repeated blocking")
                                    except Exception:
                                        pass
                            else:
                                # Blocked by another PAR agent: do not accumulate toward early-exit
                                self._par_blocked_counts[aid] = 0
                            continue

                        # Position will be applied before _step_pure_rl (moved above)
                        applied_positions[aid] = proposed
        except Exception:
            pass

        # Execute the modified actions using pure RL logic
        # Pre-yield: prevent RL agents from moving into PAR agents (predictive one-step check)
        try:
            # Collect current PAR agent positions and radii
            par_positions = []
            for aid in range(len(self.robot_list)):
                try:
                    if self.get_current_mode(aid) == 'par':
                        rx = float(self.robot_list[aid].state[0, 0])
                        ry = float(self.robot_list[aid].state[1, 0])
                        rr = float(getattr(self.robot_list[aid], 'radius_collision', 0.2))
                        par_positions.append((aid, rx, ry, rr))
                except Exception:
                    continue

            # For each RL agent, if predicted next pos would collide with any PAR, freeze this step
            if isinstance(modified_action_list, list) and len(par_positions) > 0:
                dt = float(getattr(self, 'step_time', 0.1))
                for rid in range(min(len(modified_action_list), len(self.robot_list))):
                    try:
                        if self.get_current_mode(rid) == 'par':
                            continue
                        act = modified_action_list[rid]
                        if act is None or not hasattr(self.robot_list[rid], 'state'):
                            continue
                        vx = float(act[0]) if isinstance(act, (list, tuple)) else float(act[0])
                        vy = float(act[1]) if isinstance(act, (list, tuple)) else float(act[1])
                        cx = float(self.robot_list[rid].state[0, 0])
                        cy = float(self.robot_list[rid].state[1, 0])
                        # Predict one-step advance (omni model)
                        nx = cx + vx * dt
                        ny = cy + vy * dt
                        r_rl = float(getattr(self.robot_list[rid], 'radius_collision', 0.2))
                        will_collide = False
                        for (paid, px, py, pr) in par_positions:
                            dx = nx - px
                            dy = ny - py
                            if (dx*dx + dy*dy) ** 0.5 < (r_rl + pr):
                                will_collide = True
                                break
                        if will_collide:
                            import numpy as np
                            modified_action_list[rid] = np.array([0.0, 0.0])
                            print(f"RL YIELD: Agent {rid} frozen to avoid PAR collision")
                    except Exception:
                        continue
        except Exception:
            pass

        # PAR-RL distance detection and forced exit logic
        try:
            # Dynamic safety threshold based on agent radii
            par_radius = 0.2  # Default PAR agent radius
            rl_radius = 0.2   # Default RL agent radius
            safety_threshold = (par_radius + rl_radius) + 0.5  # Distance threshold for PAR-RL collision detection
            par_agents = []
            rl_agents = []
            
            # Collect PAR and RL agent positions
            for aid in range(len(self.robot_list)):
                try:
                    if self.get_current_mode(aid) == 'par':
                        pos = (float(self.robot_list[aid].state[0, 0]), float(self.robot_list[aid].state[1, 0]))
                        par_agents.append((aid, pos))
                    else:
                        pos = (float(self.robot_list[aid].state[0, 0]), float(self.robot_list[aid].state[1, 0]))
                        rl_agents.append((aid, pos))
                except Exception:
                    continue
            
            # Debug: Log PAR-RL agent counts
            if par_agents or rl_agents:
                print(f"PAR-RL CHECK: Step {self.step_count}, PAR agents: {len(par_agents)}, RL agents: {len(rl_agents)}, safety_threshold: {safety_threshold:.3f}")
            
            # Check distances between PAR and RL agents
            for par_id, par_pos in par_agents:
                for rl_id, rl_pos in rl_agents:
                    distance = ((par_pos[0] - rl_pos[0])**2 + (par_pos[1] - rl_pos[1])**2)**0.5
                    if distance < safety_threshold:
                        print(f"PAR-RL COLLISION DETECTED: Agent {par_id} (PAR) and Agent {rl_id} (RL) distance={distance:.3f} < {safety_threshold}")
                        self._force_par_agent_exit(par_id)
                        break
        except Exception as e:
            print(f"Error in PAR-RL distance detection: {e}")

        # Apply PAR positions BEFORE _step_pure_rl to ensure they are not overridden
        try:
            if hasattr(self, '_par_last_set_positions') and self._par_last_set_positions:
                for aid, pos_info in self._par_last_set_positions.items():
                    proposed = pos_info['pos']
                    robot = self.robot_list[aid]
                    if hasattr(robot, 'state'):
                        try:
                            robot.state[0, 0] = proposed[0]
                            robot.state[1, 0] = proposed[1]
                        except Exception:
                            import numpy as np
                            robot.state = np.array([[proposed[0]], [proposed[1]]])
                    try:
                        import numpy as np
                        if hasattr(robot, 'vel_omni'):
                            robot.vel_omni = np.zeros((2, 1))
                        if hasattr(robot, 'vel_diff'):
                            robot.vel_diff = np.zeros((2, 1))
                    except Exception:
                        pass
                    print(f"PAR POSITION SET: Agent {aid} position set to {proposed}")
                # Clear after applying
                self._par_last_set_positions.clear()
        except Exception:
            pass

        # DEBUG: Check PAR positions before _step_pure_rl
        par_positions_before = {}
        for aid in range(len(self.robot_list)):
            if self.get_current_mode(aid) == 'par':
                try:
                    pos = (float(self.robot_list[aid].state[0, 0]), float(self.robot_list[aid].state[1, 0]))
                    par_positions_before[aid] = pos
                    print(f"PAR DEBUG BEFORE: Agent {aid} position before _step_pure_rl: {pos}")
                except Exception:
                    pass

        # Now run dynamics with possibly adjusted actions
        obs_list, reward_list, done_list, info_list = self._step_pure_rl(modified_action_list)

        # DEBUG: Check PAR positions after _step_pure_rl
        for aid in par_positions_before:
            try:
                pos = (float(self.robot_list[aid].state[0, 0]), float(self.robot_list[aid].state[1, 0]))
                if pos != par_positions_before[aid]:
                    print(f"PAR DEBUG AFTER: Agent {aid} position CHANGED after _step_pure_rl: {par_positions_before[aid]} -> {pos}")
                else:
                    print(f"PAR DEBUG AFTER: Agent {aid} position unchanged after _step_pure_rl: {pos}")
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
                
                # Clear any queued PAR positions
                if hasattr(self, '_par_last_set_positions') and self._par_last_set_positions:
                    if agent_id in self._par_last_set_positions:
                        del self._par_last_set_positions[agent_id]
                        
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
                            
                            print(f"BOUNDARY ENFORCE: Agent {i} action limited from [{action[0]:.3f}, {action[1]:.3f}] to [{max_vx:.3f}, {max_vy:.3f}] (bounds: [0,{world_width}] x [0,{world_height}])")
        except Exception as e:
            print(f"Error enforcing boundaries: {e}")
        
        return action_list
    