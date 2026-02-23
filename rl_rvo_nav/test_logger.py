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
            "start_time_epoch": time.time(),
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
        end_time_epoch = time.time()
        start_time_epoch = self.current_episode_data.get("start_time_epoch", None)
        episode_wall_time = None
        if isinstance(start_time_epoch, (int, float)):
            episode_wall_time = float(end_time_epoch - start_time_epoch)

        self.current_episode_data.update({
            "end_time": datetime.now().isoformat(),
            "end_time_epoch": end_time_epoch,
            "episode_wall_time": episode_wall_time,
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

            # Build compact stats for quick review and save to session_stats.json
            timeout_episodes = 0
            rl_rl_collision_eps = 0
            rl_par_collision_eps = 0
            par_par_collision_eps = 0
            any_agent_obstacle_eps = 0

            for ep in episodes:
                if ep.get("success", False):
                    continue
                reasons = ep.get("failure_reason", []) or []
                if isinstance(reasons, dict):
                    reasons = [reasons]

                # Episode-level flags (avoid duplicate counting)
                ep_timeout = False
                ep_rl_rl = False
                ep_rl_par = False
                ep_par_par = False
                ep_agent_obstacle = False

                for r in reasons:
                    rtype = r.get('type', '')
                    if rtype == 'timeout':
                        ep_timeout = True
                    elif rtype == 'robot_robot':
                        m1 = str(r.get('robot_mode', 'rl_rvo')).lower()
                        m2 = str(r.get('other_robot_mode', 'rl_rvo')).lower()
                        if m1 == 'mapf' and m2 == 'mapf':
                            ep_par_par = True
                        elif (m1 == 'mapf' and m2 != 'mapf') or (m2 == 'mapf' and m1 != 'mapf'):
                            ep_rl_par = True
                        else:
                            ep_rl_rl = True
                    elif rtype == 'robot_obstacle':
                        ep_agent_obstacle = True

                timeout_episodes += 1 if ep_timeout else 0
                rl_rl_collision_eps += 1 if ep_rl_rl else 0
                rl_par_collision_eps += 1 if ep_rl_par else 0
                par_par_collision_eps += 1 if ep_par_par else 0
                any_agent_obstacle_eps += 1 if ep_agent_obstacle else 0

            session_stats = {
                "total_episodes": total_episodes,
                "success_episodes": successful_episodes,
                "timeout_episodes": timeout_episodes,
                "collision_rl_rl_episodes": rl_rl_collision_eps,
                "collision_rl_par_episodes": rl_par_collision_eps,
                "collision_par_par_episodes": par_par_collision_eps,
                "collision_agent_obstacle_episodes": any_agent_obstacle_eps
            }

            def _safe_stats(values):
                if not values:
                    return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
                arr = np.array(values, dtype=float)
                return {
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 95))
                }

            nav_episode_metrics = []
            travel_times_all = []
            path_lengths_all = []
            avg_speeds_all = []

            step_wall_times = []
            rl_inference_means = []
            rl_inference_sums = []
            deadlock_check_times = []
            mapf_solve_times = []
            episode_wall_times = []

            for ep in episodes:
                steps = ep.get("steps", []) or []
                robot_number = int(ep.get("robot_number", 0) or 0)
                step_time = 1.0
                try:
                    step_time = float(ep.get("config", {}).get("step_time", 1.0))
                except Exception:
                    step_time = 1.0

                if isinstance(ep.get("episode_wall_time", None), (int, float)):
                    episode_wall_times.append(float(ep.get("episode_wall_time")))

                arrival_steps = [None for _ in range(robot_number)]
                path_lengths = [0.0 for _ in range(robot_number)]
                prev_positions = None

                for step in steps:
                    positions = step.get("robot_positions", []) or []
                    if prev_positions is not None and positions:
                        count = min(len(prev_positions), len(positions), robot_number)
                        for i in range(count):
                            dx = float(positions[i][0]) - float(prev_positions[i][0])
                            dy = float(positions[i][1]) - float(prev_positions[i][1])
                            path_lengths[i] += (dx * dx + dy * dy) ** 0.5
                    prev_positions = positions if positions else prev_positions

                    step_info = step.get("additional_info", {}) if isinstance(step.get("additional_info", {}), dict) else {}
                    info = step_info.get("info", None)
                    if isinstance(info, list):
                        for i, item in enumerate(info):
                            if i >= robot_number:
                                break
                            if arrival_steps[i] is None and isinstance(item, dict) and bool(item.get("done", False)):
                                arrival_steps[i] = int(step.get("step", 0))

                    timing = step_info.get("timing", {}) if isinstance(step_info, dict) else {}
                    if isinstance(timing, dict):
                        if isinstance(timing.get("step_wall_time"), (int, float)):
                            step_wall_times.append(float(timing.get("step_wall_time")))
                        if isinstance(timing.get("rl_inference_mean"), (int, float)):
                            rl_inference_means.append(float(timing.get("rl_inference_mean")))
                        if isinstance(timing.get("rl_inference_sum"), (int, float)):
                            rl_inference_sums.append(float(timing.get("rl_inference_sum")))
                        if isinstance(timing.get("deadlock_check_time"), (int, float)):
                            deadlock_check_times.append(float(timing.get("deadlock_check_time")))
                        if isinstance(timing.get("mapf_solve_time"), (int, float)):
                            mapf_solve_times.append(float(timing.get("mapf_solve_time")))

                travel_times = []
                for i in range(robot_number):
                    if arrival_steps[i] is not None:
                        travel_time = (arrival_steps[i] + 1) * step_time
                    else:
                        travel_time = len(steps) * step_time
                    travel_times.append(float(travel_time))
                    travel_times_all.append(float(travel_time))
                    path_lengths_all.append(float(path_lengths[i]))
                    avg_speed = float(path_lengths[i]) / float(travel_time) if travel_time > 0 else 0.0
                    avg_speeds_all.append(avg_speed)

                makespan = max(travel_times) if travel_times else 0.0
                flowtime = sum(travel_times)
                avg_travel_time = (flowtime / len(travel_times)) if travel_times else 0.0
                avg_path_length = float(np.mean(path_lengths)) if path_lengths else 0.0
                avg_speed = float(np.mean([pl / tt if tt > 0 else 0.0 for pl, tt in zip(path_lengths, travel_times)])) if travel_times else 0.0

                nav_episode_metrics.append({
                    "makespan": makespan,
                    "flowtime": flowtime,
                    "avg_travel_time": avg_travel_time,
                    "avg_path_length": avg_path_length,
                    "avg_speed": avg_speed
                })

            nav_summary = {
                "makespan_mean": float(np.mean([m["makespan"] for m in nav_episode_metrics])) if nav_episode_metrics else 0.0,
                "flowtime_mean": float(np.mean([m["flowtime"] for m in nav_episode_metrics])) if nav_episode_metrics else 0.0,
                "avg_travel_time_mean": float(np.mean([m["avg_travel_time"] for m in nav_episode_metrics])) if nav_episode_metrics else 0.0,
                "avg_path_length_mean": float(np.mean([m["avg_path_length"] for m in nav_episode_metrics])) if nav_episode_metrics else 0.0,
                "avg_speed_mean": float(np.mean([m["avg_speed"] for m in nav_episode_metrics])) if nav_episode_metrics else 0.0
            }

            runtime_summary = {
                "step_wall_time": _safe_stats(step_wall_times),
                "rl_inference_mean": _safe_stats(rl_inference_means),
                "rl_inference_sum": _safe_stats(rl_inference_sums),
                "deadlock_check_time": _safe_stats(deadlock_check_times),
                "mapf_solve_time": _safe_stats([v for v in mapf_solve_times if v > 0.0]),
                "episode_wall_time": _safe_stats(episode_wall_times)
            }

            deadlock_summary = {}
            try:
                if self.env is not None and hasattr(self.env, 'ir_gym'):
                    logger = getattr(self.env.ir_gym, 'deadlock_logger', None)
                    if logger and hasattr(logger, 'get_session_metrics'):
                        deadlock_summary = logger.get_session_metrics()
            except Exception:
                deadlock_summary = {}

            metrics_summary = {
                "navigation": nav_summary,
                "runtime": runtime_summary,
                "deadlock": deadlock_summary
            }
            self.session_metadata["metrics"] = metrics_summary
        
        # Save session summary
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        # Save compact session stats (if built)
        try:
            stats_file = self.session_dir / "session_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(session_stats, f, indent=2)
        except Exception:
            pass

        try:
            metrics_file = self.session_dir / "metrics_summary.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.session_metadata.get("metrics", {}), f, indent=2)
        except Exception:
            pass
        
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
            List of agent modes ('rl_rvo' or 'mapf')
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
