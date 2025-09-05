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


    def rvo_reward_list_cal(self, action_list, **kwargs):    
        ts = self.components['robots'].total_states() # robot_state_list, nei_state_list, obs_circular_list, obs_line_list

        rvo_reward_list = list(map(lambda robot_state, action: self.rvo_reward_cal(robot_state, ts[1], ts[2], ts[3], action, self.reward_parameter, **kwargs), ts[0], action_list))

        return rvo_reward_list
    
    def rvo_reward_cal(self, robot_state, nei_state_list, obs_cir_list, obs_line_list, action, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20), **kwargs):
        
        vo_flag, min_exp_time, min_dis = self.rvo.config_vo_reward(robot_state, nei_state_list, obs_cir_list, obs_line_list, action, **kwargs)

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

        obs_vo_list, vo_flag, min_exp_time, collision_flag = self.rvo.config_vo_inf(robot_omni_state, nei_state_list, obs_circular_list, obs_line_list, action, **kwargs)

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
        
        self.components['robots'].robots_reset(reset_mode, **kwargs)
        
        # Reset deadlock detection for new episode
        if self.enable_deadlock_resolution and hasattr(self, 'deadlock_detector') and self.deadlock_detector:
            self.deadlock_detector.reset_episode()
        
        # Reset state manager for new episode
        if self.enable_deadlock_resolution and hasattr(self, 'state_manager') and self.state_manager:
            self.state_manager.reset_episode()
        
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
        
        # Get current states
        ts = self.components['robots'].total_states()
        agent_states = self._get_agent_states_dict(ts[0])
        neighbor_states = self._get_neighbor_states_dict(ts[1])
        
        # Log step start if logger is available
        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
            self.deadlock_logger.log_step_start(self.step_count, agent_states, neighbor_states)
        
        # Process each agent
        modified_action_list = action_list.copy()
        
        for agent_id, action in enumerate(action_list):
            current_mode = self.get_current_mode(agent_id)
            
            # Check for deadlock and mode switching
            if current_mode == 'rl_rvo':
                # Check if should switch to PAR mode
                if self.deadlock_detector.detect_deadlock(agent_id, agent_states, neighbor_states):
                    deadlock_participants = self.deadlock_detector.get_deadlock_participants(agent_id, neighbor_states)
                    
                    # Only proceed if we have multiple participants
                    if len(deadlock_participants) > 1:
                        try:
                            par_solution = self.par_coordinator.prepare_par_execution(agent_states, deadlock_participants)
                            
                            # Log PAR preparation with agent positions
                            if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                                # Log agent positions when switching from RL to MAPF
                                agent_positions = {}
                                for participant_id in deadlock_participants:
                                    if participant_id in agent_states:
                                        agent_state = agent_states[participant_id]
                                        if 'position' in agent_state:
                                            agent_positions[participant_id] = agent_state['position']
                                
                                self.deadlock_logger.log_par_preparation(agent_id, deadlock_participants, par_solution)
                                self.deadlock_logger.log_rl_to_mapf_positions(agent_positions)
                                self.deadlock_logger.log_par_solution_paths(par_solution, deadlock_participants)
                            
                            # Set PAR mode for all participants and initialize PAR executor
                            for participant_id in deadlock_participants:
                                old_mode = self.get_current_mode(participant_id)
                                self.state_manager.set_par_mode(participant_id, par_solution)
                                
                                # Initialize PAR executor with solution data
                                if par_solution and hasattr(par_solution, 'agents_moves'):
                                    # Set start and goal positions for the participant
                                    agent_state = agent_states.get(participant_id, {})
                                    if 'position' in agent_state:
                                        start_pos = agent_state['position']
                                        if len(start_pos) >= 2:
                                            self.par_executor.set_agent_start_position(participant_id, (start_pos[0], start_pos[1]))
                                    
                                    # Set goal position (use original goal)
                                    if 'goal' in agent_state and agent_state['goal'] is not None:
                                        goal = agent_state['goal']
                                        if isinstance(goal, list) and len(goal) >= 2:
                                            if isinstance(goal[0], list) and len(goal[0]) > 0:
                                                goal_x = goal[0][0]
                                            else:
                                                goal_x = goal[0]
                                            if isinstance(goal[1], list) and len(goal[1]) > 0:
                                                goal_y = goal[1][0]
                                            else:
                                                goal_y = goal[1]
                                            self.par_executor.set_agent_goal_position(participant_id, (goal_x, goal_y))
                                    
                                    # Set path for the participant
                                    path = self.par_executor.get_agent_path_from_solution(participant_id, par_solution)
                                    if path:
                                        self.par_executor.set_agent_path(participant_id, path)
                                
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
            
            # Execute action based on current mode
            if current_mode == 'par':
                try:
                    par_action = self.par_executor.execute_par_step(agent_id, agent_states)
                    if par_action and 'action' in par_action:
                        modified_action_list[agent_id] = par_action['action']
                        
                        # Log PAR execution
                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                            self.deadlock_logger.log_par_execution(agent_id, par_action['action'], 'executing')
                    
                    # Check if PAR is complete
                    if self.par_coordinator.is_par_complete(agent_id):
                        old_mode = self.get_current_mode(agent_id)
                        self.state_manager.set_rl_rvo_mode(agent_id)
                        
                        # Log agent positions when switching from MAPF back to RL
                        if hasattr(self, 'deadlock_logger') and self.deadlock_logger:
                            agent_positions = {}
                            for participant_id in self.deadlock_detector.get_deadlock_participants(agent_id, {}):
                                if participant_id in agent_states:
                                    agent_state = agent_states[participant_id]
                                    if 'position' in agent_state:
                                        agent_positions[participant_id] = agent_state['position']
                            
                            self.deadlock_logger.log_mode_switch(agent_id, old_mode, 'rl_rvo', "PAR execution completed")
                            self.deadlock_logger.log_par_completion(agent_id, 0)  # TODO: track actual steps
                            self.deadlock_logger.log_mapf_to_rl_positions(agent_positions)
                        
                        # print(f"✅ PAR COMPLETED: Agent {agent_id} finished PAR execution, switching back to RL_RVO")
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
        
        # Execute the modified actions using pure RL logic
        return self._step_pure_rl(modified_action_list)
    
    def _step_pure_rl(self, action_list):
        """Execute step with pure RL logic (original behavior)."""
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
        """Convert neighbor state list to dictionary format."""
        neighbor_states = {}
        for i, nei_states in enumerate(nei_state_list):
            neighbor_states[i] = {}
            for j, nei_state in enumerate(nei_states):
                # Handle different neighbor state formats
                if isinstance(nei_state, (list, np.ndarray)) and len(nei_state) >= 4:
                    neighbor_states[i][j] = {
                        'position': nei_state[0:2],
                        'velocity': nei_state[2:4]
                    }
                else:
                    # Skip invalid neighbor states
                    continue
        return neighbor_states
    