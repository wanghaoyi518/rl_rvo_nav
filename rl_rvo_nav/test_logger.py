import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from .visualizer import TestVisualizer


class TestLogger:
    """
    Logger for recording test case execution with agent positions at each time step
    """
    
    def __init__(self, test_type="test", base_log_dir="/home/haoyiwang/Desktop/RL_RVO/logs"):
        """
        Initialize the test logger
        
        Args:
            test_type: Type of test ("test" or "test_with_deadlock")
            base_log_dir: Base directory for logs
        """
        self.test_type = test_type
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_log_dir / f"{timestamp}_{test_type}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Current episode data
        self.current_episode = None
        self.current_episode_data = {}
        self.episode_step_count = 0
        self.env = None  # Initialize env attribute
        
        # Session metadata
        self.session_metadata = {
            "test_type": test_type,
            "start_time": timestamp,
            "episodes": {}
        }
        
        # Initialize visualizer (can be disabled via env var: DISABLE_TEST_VIS=1/true/yes)
        disable_vis_env = os.environ.get('DISABLE_TEST_VIS', '0')
        self._enable_visualization = str(disable_vis_env).lower() not in ('1', 'true', 'yes')
        if self._enable_visualization:
            self.visualizer = TestVisualizer(test_type=test_type)
        else:
            self.visualizer = None
        
        # print(f"Test logger initialized. Session directory: {self.session_dir}")
    
    def start_episode(self, episode_id, robot_number, episode_config=None):
        """
        Start logging a new episode
        
        Args:
            episode_id: Episode number
            robot_number: Number of robots
            episode_config: Additional episode configuration
        """
        self.current_episode = episode_id
        self.episode_step_count = 0
        
        # Get start and goal positions from environment
        start_positions = self.get_robot_start_positions_from_env()
        goal_positions = self.get_robot_goal_positions_from_env()
        
        # Preserve existing waypoint_data and discretized_grid if they exist
        existing_waypoint_data = self.current_episode_data.get('waypoint_data', None) if hasattr(self, 'current_episode_data') else None
        existing_discretized_grid = self.current_episode_data.get('discretized_grid', None) if hasattr(self, 'current_episode_data') else None
        
        # Prefer final goals from waypoint_data (config goals) over robot.goal
        # Rationale: robot.goal may be overwritten by long-range current waypoint before logging starts
        try:
            if existing_waypoint_data:
                recovered_goals = []
                for v in existing_waypoint_data.values():
                    if isinstance(v, dict) and 'final_goal' in v and isinstance(v['final_goal'], (list, tuple)) and len(v['final_goal']) >= 2:
                        recovered_goals.append([float(v['final_goal'][0]), float(v['final_goal'][1])])
                if recovered_goals:
                    goal_positions = recovered_goals
        except Exception:
            pass

        # Create episode data with waypoint_data in the correct position (after goal_positions, before steps)
        self.current_episode_data = {
            "episode_id": episode_id,
            "robot_number": robot_number,
            "start_time": datetime.now().isoformat(),
            "start_positions": start_positions,
            "goal_positions": goal_positions,
            "steps": [],
            "config": episode_config or {}
        }
        
        # Insert waypoint_data and discretized_grid in the correct position if they existed
        if existing_waypoint_data is not None or existing_discretized_grid is not None:
            # Create a new ordered dictionary with waypoint_data and discretized_grid in the right place
            ordered_data = {}
            for key, value in self.current_episode_data.items():
                ordered_data[key] = value
                if key == "goal_positions":
                    if existing_waypoint_data is not None:
                        ordered_data["waypoint_data"] = existing_waypoint_data
                    if existing_discretized_grid is not None:
                        ordered_data["discretized_grid"] = existing_discretized_grid
            self.current_episode_data = ordered_data
        
        # print(f"Started logging episode {episode_id} with {robot_number} robots")
    
    def log_step(self, robot_positions, robot_velocities=None, additional_info=None):
        """
        Log a single time step with all robot positions
        
        Args:
            robot_positions: List of robot positions [(x1, y1), (x2, y2), ...]
            robot_velocities: Optional list of robot velocities
            additional_info: Optional additional information to log
        """
        if self.current_episode is None:
            # print("Warning: No episode started. Call start_episode() first.")
            return
        
        step_data = {
            "step": self.episode_step_count,
            "timestamp": datetime.now().isoformat(),
            "robot_positions": robot_positions,
        }
        
        if robot_velocities is not None:
            step_data["robot_velocities"] = robot_velocities
        
        if additional_info is not None:
            step_data["additional_info"] = additional_info
        
        self.current_episode_data["steps"].append(step_data)
        self.episode_step_count += 1
    
    def end_episode(self, success=False, episode_reward=0, episode_length=0, failure_reason=None):
        """
        End the current episode and save data
        
        Args:
            success: Whether the episode was successful
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
            failure_reason: Reason for failure if applicable
        """
        if self.current_episode is None:
            # print("Warning: No episode to end.")
            return
        
        # Add episode summary
        self.current_episode_data.update({
            "end_time": datetime.now().isoformat(),
            "success": success,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "total_steps": self.episode_step_count,
            "failure_reason": failure_reason
        })
        
        # Save episode data to session metadata
        self.session_metadata["episodes"][str(self.current_episode)] = self.current_episode_data
        
        # Save individual episode file
        episode_file = self.session_dir / f"episode_{self.current_episode:03d}.json"
        with open(episode_file, 'w') as f:
            json.dump(self.current_episode_data, f, indent=2)
        
        # Create visualization for this episode (if enabled)
        if self.visualizer is not None:
            try:
                self.visualizer.create_episode_visualization(self.current_episode_data, self.current_episode)
            except Exception as e:
                print(f"Warning: Could not create visualization for episode {self.current_episode}: {e}")
        
        # print(f"Episode {self.current_episode} completed. Success: {success}, Length: {episode_length}, Steps: {self.episode_step_count}")
        
        # Reset for next episode
        self.current_episode = None
        self.current_episode_data = {}
        self.episode_step_count = 0
    
    def save_session_summary(self):
        """
        Save session summary and metadata
        """
        self.session_metadata["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate session statistics
        episodes = list(self.session_metadata["episodes"].values())
        if episodes:
            successful_episodes = sum(1 for ep in episodes if ep.get("success", False))
            total_episodes = len(episodes)
            avg_reward = np.mean([ep.get("episode_reward", 0) for ep in episodes])
            avg_length = np.mean([ep.get("episode_length", 0) for ep in episodes])
            avg_steps = np.mean([ep.get("total_steps", 0) for ep in episodes])
            
            self.session_metadata["statistics"] = {
                "total_episodes": total_episodes,
                "successful_episodes": successful_episodes,
                "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0,
                "average_reward": float(avg_reward),
                "average_length": float(avg_length),
                "average_steps": float(avg_steps)
            }
        
        # Save session summary
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        # print(f"Session summary saved to {summary_file}")
    
    def get_robot_positions_from_env(self, env):
        """
        Extract robot positions from environment
        
        Args:
            env: Environment object with robot_list
            
        Returns:
            List of robot positions
        """
        positions = []
        for robot in env.ir_gym.robot_list:
            positions.append([float(robot.state[0, 0]), float(robot.state[1, 0])])
        return positions
    
    def get_robot_velocities_from_env(self, env):
        """
        Extract robot velocities from environment
        
        Args:
            env: Environment object with robot_list
            
        Returns:
            List of robot velocities
        """
        velocities = []
        for robot in env.ir_gym.robot_list:
            velocities.append([float(robot.vel_omni[0]), float(robot.vel_omni[1])])
        return velocities
    
    def get_agent_modes_from_env(self, env):
        """
        Extract agent modes from environment (for deadlock resolution)
        
        Args:
            env: Environment object with deadlock resolution capability
            
        Returns:
            List of agent modes ('rl_rvo' or 'par')
        """
        modes = []
        if hasattr(env, 'get_current_mode'):
            for i in range(len(env.ir_gym.robot_list)):
                mode = env.get_current_mode(i)
                modes.append(mode)
        else:
            # If no deadlock resolution, all agents are in RL mode
            for i in range(len(env.ir_gym.robot_list)):
                modes.append('rl_rvo')
        return modes
    
    def get_current_goals_from_env(self, env):
        """
        Extract current goal points from environment (for long-range navigation)
        
        Args:
            env: Environment object with long-range navigation capability
            
        Returns:
            List of current goal points [(x1, y1), (x2, y2), ...]
        """
        goals = []
        if hasattr(env.ir_gym, 'enable_long_range_nav') and env.ir_gym.enable_long_range_nav:
            if hasattr(env.ir_gym, '_waypoint_managers') and isinstance(env.ir_gym._waypoint_managers, dict):
                for aid in range(len(env.ir_gym.robot_list)):
                    if aid in env.ir_gym._waypoint_managers:
                        cur_goal = env.ir_gym._waypoint_managers[aid].get_current_goal()
                        if cur_goal is not None:
                            goals.append([float(cur_goal[0]), float(cur_goal[1])])
                        else:
                            goals.append([0.0, 0.0])
                    else:
                        goals.append([0.0, 0.0])
            else:
                # Fallback: get goals from robot.goal
                for robot in env.ir_gym.robot_list:
                    if hasattr(robot, 'goal') and robot.goal is not None:
                        goals.append([float(robot.goal[0]), float(robot.goal[1])])
                    else:
                        goals.append([0.0, 0.0])
        else:
            # Fallback: get goals from robot.goal
            for robot in env.ir_gym.robot_list:
                if hasattr(robot, 'goal') and robot.goal is not None:
                    goals.append([float(robot.goal[0]), float(robot.goal[1])])
                else:
                    goals.append([0.0, 0.0])
        return goals
    
    def get_robot_start_positions_from_env(self):
        """
        Extract robot start positions from environment
        
        Returns:
            List of start positions [(x1, y1), (x2, y2), ...]
        """
        if self.env is None:
            return []
        
        start_positions = []
        try:
            for i, robot in enumerate(self.env.ir_gym.robot_list):
                if hasattr(robot, 'init_state'):
                    # init_state is typically [[x], [y], [theta]]
                    start_pos = [float(robot.init_state[0, 0]), float(robot.init_state[1, 0])]
                    start_positions.append(start_pos)
                else:
                    # Fallback to current state
                    start_pos = [float(robot.state[0, 0]), float(robot.state[1, 0])]
                    start_positions.append(start_pos)
        except Exception as e:
            print(f"Warning: Could not extract start positions: {e}")
            return []
        
        return start_positions
    
    def get_robot_goal_positions_from_env(self):
        """
        Extract robot goal positions from environment
        
        Returns:
            List of goal positions [(x1, y1), (x2, y2), ...]
        """
        if self.env is None:
            return []
        
        goal_positions = []
        try:
            for i, robot in enumerate(self.env.ir_gym.robot_list):
                if hasattr(robot, 'goal') and robot.goal is not None:
                    g = robot.goal
                    # Handle numpy array like [[x],[y]] or [x, y]
                    try:
                        if hasattr(g, 'shape') and getattr(g, 'shape', [0])[0] >= 2:
                            goal_pos = [float(g[0, 0]), float(g[1, 0])]
                        elif isinstance(g, (list, tuple)) and len(g) >= 2:
                            goal_pos = [float(g[0]), float(g[1])]
                        else:
                            goal_pos = [float(robot.state[0, 0]), float(robot.state[1, 0])]
                    except Exception:
                        goal_pos = [float(robot.state[0, 0]), float(robot.state[1, 0])]
                    goal_positions.append(goal_pos)
                else:
                    # Fallback to current state if no goal available
                    goal_pos = [float(robot.state[0, 0]), float(robot.state[1, 0])]
                    goal_positions.append(goal_pos)
        except Exception as e:
            print(f"Warning: Could not extract goal positions: {e}")
            return []
        
        return goal_positions
    
    def set_environment(self, env):
        """
        Set environment reference for visualization and logging
        
        Args:
            env: Environment object
        """
        self.env = env
        if getattr(self, 'visualizer', None) is not None:
            self.visualizer.set_environment(env)
