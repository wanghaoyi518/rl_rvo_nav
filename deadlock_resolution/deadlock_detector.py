"""
Deadlock Detector Module

This module provides deadlock detection functionality for RL_RVO navigation system. 
Including trigger mechanisms: speed buffer trigger, 
common point trigger, hybrid trigger, collision prevention trigger, immediate speed trigger

"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import sys
import os

# Add the policy_test_with_deadlock directory to the path for logger import
try:
    # Try relative import first
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rl_rvo_nav', 'policy_test_with_deadlock'))
    from deadlock_logger import get_deadlock_logger
except ImportError:
    try:
        # Try absolute import
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'rl_rvo_nav', 'policy_test_with_deadlock'))
        from deadlock_logger import get_deadlock_logger
    except ImportError:
        # If all imports fail, set to None and handle gracefully
        get_deadlock_logger = None
        print("Warning: Could not import deadlock_logger, logging will be disabled")


class DeadlockDetector:
    """
    Deadlock detector that monitors agent states and detects deadlock situations.
    
    Supports two trigger mechanisms:
    1. Speed Buffer Trigger: Detects deadlock based on low average velocity
    2. Common Point Trigger: Detects deadlock based on multiple agents near common goal
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the deadlock detector.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.trigger_type = config.get('TRIGGER_TYPE', 'SPEED_BUFFER')
        self.small_speed = config.get('SMALL_SPEED', 0.2)
        self.mapf_num = config.get('MAPF_NUM', 5)
        self.sight_radius = config.get('SIGHT_RADIUS', 7.0)
        self.velocity_window_size = config.get('VELOCITY_WINDOW_SIZE', 5)
        
        
        # Velocity history for each agent
        self.velocity_history = defaultdict(list)
        
        # Deadlock detection state
        self.deadlock_detection_enabled = config.get('DEADLOCK_DETECTION_ENABLED', True)
        
        # Episode counter for debugging
        self.episode_counter = 0
        
        # Episode start delay for deadlock detection
        self.episode_start_delay = config.get('EPISODE_START_DELAY', 5)
        self.step_counter = 0
        
        # Cooldown mechanism for deadlock detection
        self.deadlock_detection_cooldown = config.get('DEADLOCK_DETECTION_COOLDOWN', 5)
        self.last_deadlock_detection = {}  # Track last detection time for each agent
        
        # Participant lock to avoid oscillation in participant selection
        self.participant_lock_steps = config.get('PARTICIPANT_LOCK_STEPS', 0)
        self.locked_participants: Dict[int, Tuple[List[int], int]] = {}
        
        # Goal tolerance for arrival check
        self.goal_tolerance = config.get('GOAL_TOLERANCE', 0.1)
        
        # Global velocity history update guard per step
        self._velhist_step = -1

        # Risk gating thresholds for SPEED_BUFFER (fallback defaults if absent in config)
        try:
            self.ttc_threshold = float(config.get('TTC_THRESHOLD', 2.0))
        except Exception:
            self.ttc_threshold = 2.0
        try:
            self.dmin_threshold = float(config.get('DMIN_THRESHOLD', 0.4))
        except Exception:
            self.dmin_threshold = 0.4
        try:
            self.required_non_progress_neighbors = int(config.get('REQUIRED_NON_PROGRESS_NEIGHBORS', 1))
        except Exception:
            self.required_non_progress_neighbors = 1

        # Waypoint-stuck trigger (long-range): optional independent trigger
        self.enable_wp_stuck = bool(config.get('ENABLE_WAYPOINT_STUCK_TRIGGER', True))
        self.wp_stuck_steps = int(config.get('WAYPOINT_STUCK_STEPS', 100))
        # Track each agent's last observed waypoint index and when it last changed
        self._last_wp_index: Dict[int, int] = {}
        self._last_wp_change_step: Dict[int, int] = {}
        
        # Initialize logger
        try:
            if get_deadlock_logger is not None:
                self.logger = get_deadlock_logger()
                self.logger.logger.info(f"🔧 DeadlockDetector initialized with config: {config}")
            else:
                self.logger = None
                print("Deadlock logger not available, logging disabled")
        except Exception as e:
            print(f"Warning: Failed to initialize deadlock logger: {e}")
            self.logger = None
    
    def set_logger(self, logger):
        """Set the logger instance for this detector."""
        self.logger = logger
        if self.logger:
            self.logger.logger.info(f"🔧 Logger set for DeadlockDetector")
    
    def detect_deadlock(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Detect if a deadlock situation exists for the given agent.
        
        Args:
            agent_id: ID of the agent to check
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if deadlock is detected, False otherwise
        """
        if not self.deadlock_detection_enabled:
            return False
        
        # Note: step_counter is incremented externally by the gym environment
        # This ensures it's incremented once per step, not once per agent
        
        # Check episode start delay - don't detect deadlock too early
        if self.step_counter < self.episode_start_delay:
            if self.logger:
                self.logger.logger.debug(f"🔍 EPISODE START DELAY: Agent {agent_id}, Step {self.step_counter} < {self.episode_start_delay}")
            return False
        
        # Check cooldown period - prevent frequent deadlock detection for the same agent
        if agent_id in self.last_deadlock_detection:
            steps_since_last_detection = self.step_counter - self.last_deadlock_detection[agent_id]
            if steps_since_last_detection < self.deadlock_detection_cooldown:
                if self.logger:
                    self.logger.logger.debug(f"🔍 COOLDOWN: Agent {agent_id}, Step {self.step_counter}, Last detection at step {self.last_deadlock_detection[agent_id]}, Cooldown: {self.deadlock_detection_cooldown}")
                return False
        
        # Log deadlock detection attempt
        if self.logger:
            self.logger.logger.debug(f"🔍 DEADLOCK DETECTION: Agent {agent_id}, Trigger type: {self.trigger_type}")
        
        # Early exit: do not detect deadlock if agent is already at goal
        try:
            if agent_id in agent_states:
                distance_to_goal = self.calculate_distance_to_goal(agent_states[agent_id])
                if distance_to_goal <= float(self.goal_tolerance):
                    if self.logger:
                        self.logger.logger.debug(f"🟢 AT GOAL: Agent {agent_id}, distance {distance_to_goal:.3f} <= {self.goal_tolerance}")
                    return False
        except Exception:
            pass

        # Update velocity history for all agents once per step, plus neighbors for completeness
        if self._velhist_step != self.step_counter:
            self._velhist_step = self.step_counter
            # Global per-agent velocity history update (ensures neighbors_comm have history)
            for aid, st in agent_states.items():
                v = self.calculate_current_velocity(st)
                self.update_velocity_history(aid, v)
        # Also update neighbor velocity histories from provided neighbor_states
        self._update_all_velocity_histories(agent_id, agent_states, neighbor_states)
            
        if self.trigger_type == 'SPEED_BUFFER':
            result_speed = self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        else:
            # Default to speed buffer trigger
            result_speed = self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)

        # Independent waypoint-stuck trigger (does not modify speed buffer logic)
        result_wp = self.check_waypoint_stuck_trigger(agent_id, agent_states)
        result = bool(result_speed or result_wp)

        # Single-agent fallback trigger: when only one unfinished agent remains
        try:
            single_enabled = bool(self.config.get('SINGLE_AGENT_TRIGGER_ENABLED', True)) if isinstance(self.config, dict) else True
        except Exception:
            single_enabled = True
        if not result and single_enabled:
            try:
                tol = float(self.goal_tolerance)
            except Exception:
                tol = 0.5
            unfinished = []
            for aid, st in agent_states.items():
                try:
                    if self.calculate_distance_to_goal(st) > tol:
                        unfinished.append(aid)
                except Exception:
                    pass
            if len(unfinished) == 1 and unfinished[0] == agent_id:
                # Last unfinished agent: allow relaxed trigger based on slow/not-progressing or waypoint-stuck
                is_not_prog = self._is_not_progressing_toward_goal(agent_states.get(agent_id, {}))
                avg_v = self.calculate_average_velocity(agent_id)
                slow_enough = (avg_v < self.small_speed)
                # Also allow time-based release if needed
                time_ok = (self.step_counter >= int(self.config.get('SINGLE_AGENT_TIME_THRESHOLD', 50)))
                if result_wp or (is_not_prog and (slow_enough or time_ok)):
                    result = True
        
        # Log detection result and update cooldown
        if result:
            # Update cooldown time for this agent
            self.last_deadlock_detection[agent_id] = self.step_counter
            if self.logger:
                meta = {
                    'agent_states': {k: v for k, v in agent_states.items() if k == agent_id},
                    'neighbor_count': len(neighbor_states),
                    'trigger_type': self.trigger_type
                }
                # Annotate source if waypoint-stuck triggered
                if result_wp and not result_speed:
                    meta['trigger_source'] = 'WAYPOINT_STUCK'
                elif result_speed and not result_wp:
                    meta['trigger_source'] = 'SPEED_BUFFER'
                else:
                    meta['trigger_source'] = 'COMBINED'
                # If single-agent condition holds, annotate
                try:
                    tol = float(self.goal_tolerance)
                except Exception:
                    tol = 0.5
                try:
                    unfinished = [aid for aid, st in agent_states.items() if self.calculate_distance_to_goal(st) > tol]
                except Exception:
                    unfinished = []
                if len(unfinished) == 1 and unfinished[0] == agent_id:
                    meta['single_agent_fallback'] = True
                try:
                    self.logger.log_deadlock_detection(agent_id, self.trigger_type, meta)
                except Exception:
                    pass
        else:
            if self.logger:
                self.logger.logger.debug(f"✅ NO DEADLOCK: Agent {agent_id}")
        
        return result

    def update_waypoint_history(self, agent_id: int, waypoint_index: int) -> None:
        """Update waypoint index history for an agent.
        Call this once per step (only in long-range rl_rvo mode).
        """
        try:
            prev = self._last_wp_index.get(agent_id)
            if prev is None or int(prev) != int(waypoint_index):
                self._last_wp_index[agent_id] = int(waypoint_index)
                self._last_wp_change_step[agent_id] = int(self.step_counter)
            elif agent_id not in self._last_wp_change_step:
                # Initialize change step if absent
                self._last_wp_change_step[agent_id] = int(self.step_counter)
        except Exception:
            pass

    def check_waypoint_stuck_trigger(self, agent_id: int, agent_states: Dict) -> bool:
        """Independent trigger: agent stuck on the same waypoint index for too long.
        Only effective if environment reports waypoint indices each step.
        """
        try:
            if not self.enable_wp_stuck:
                return False
            # If no history for this agent, cannot trigger
            if agent_id not in self._last_wp_change_step:
                return False
            # Do not trigger if agent is effectively at goal
            if agent_id in agent_states:
                dist = self.calculate_distance_to_goal(agent_states[agent_id])
                if dist <= float(self.goal_tolerance):
                    return False
            # Check duration since last waypoint change
            last_change = self._last_wp_change_step.get(agent_id, self.step_counter)
            if (self.step_counter - last_change) >= int(self.wp_stuck_steps):
                if self.logger:
                    try:
                        self.logger.logger.debug(
                            f"🔍 WAYPOINT-STUCK TRIGGER: Agent {agent_id} stuck_steps={self.step_counter - last_change} >= {self.wp_stuck_steps}")
                    except Exception:
                        pass
                return True
        except Exception:
            return False
        return False
    
    def check_speed_buffer_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using speed buffer trigger mechanism.
        
        Based on the orca_mapf paper implementation:
        - Only trigger if we have enough velocity history data
        - Only trigger if neighbors also have enough velocity history
        - Avoid triggering at episode start when agents are stationary
        
        Args:
            agent_id: ID of the agent to check
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if deadlock is detected based on speed
        """
        # Get current agent state
        if agent_id not in agent_states:
            return False
            
        agent_state = agent_states[agent_id]
        
        # Check if we have enough velocity history data (at least 80% of window size for more stability)
        min_history_required = int(self.velocity_window_size * 0.8)
        if len(self.velocity_history[agent_id]) < min_history_required:
            if self.logger:
                self.logger.logger.debug(f"🔍 INSUFFICIENT HISTORY: Agent {agent_id} has {len(self.velocity_history[agent_id])} velocity samples, need at least {min_history_required}")
            return False
        
        # Calculate average velocity over history window
        avg_velocity = self.calculate_average_velocity(agent_id)

        # Condition 1: current agent is slow
        if avg_velocity < self.small_speed:
            # Build neighbors within communication range (not just passed-in neighbor_states)
            comm_range = float(self.config.get('COMMUNICATION_RANGE', 3.0))
            neighbors_comm = self._get_neighbors_in_range(agent_id, agent_states, comm_range)

            # Consider neighbors with sufficient history, not arrived, and also slow
            neighbors_with_sufficient_history = 0
            slow_neighbors = []
            non_progress_neighbors = []
            for neighbor_id in neighbors_comm:
                # Skip arrived neighbors
                try:
                    if self.calculate_distance_to_goal(agent_states.get(neighbor_id, {})) <= float(self.goal_tolerance):
                        continue
                except Exception:
                    pass
                if neighbor_id in self.velocity_history and len(self.velocity_history[neighbor_id]) >= min_history_required:
                    neighbors_with_sufficient_history += 1
                    n_avg = self.calculate_average_velocity(neighbor_id)
                    if n_avg < self.small_speed:
                        slow_neighbors.append(neighbor_id)
                        # Check non-progress toward goal based on current velocity projection
                        if self._is_not_progressing_toward_goal(agent_states.get(neighbor_id, {})):
                            non_progress_neighbors.append(neighbor_id)

            # Condition 2: neighbors also slow (at least one)
            cond_neighbors_slow = len(slow_neighbors) > 0 and neighbors_with_sufficient_history > 0
            # Condition 3: at least N adjacent neighbors cannot progress toward goal
            cond_non_progress = len(non_progress_neighbors) >= int(self.required_non_progress_neighbors)

            # Condition 4: at least one non-progress neighbor presents imminent risk (ttc/dmin)
            risk_neighbors = []
            if len(non_progress_neighbors) > 0:
                for nid in non_progress_neighbors:
                    m = self._compute_pair_metrics(agent_state, agent_states.get(nid, {}))
                    ttc = m.get('ttc', float('inf'))
                    dmin = m.get('dmin', float('inf'))
                    if (math.isfinite(ttc) and ttc <= float(self.ttc_threshold)) or (math.isfinite(dmin) and dmin <= float(self.dmin_threshold)):
                        risk_neighbors.append(nid)
            cond_risk = len(risk_neighbors) >= 1

            if cond_neighbors_slow and cond_non_progress and cond_risk:
                if self.logger:
                    self.logger.log_deadlock_check(agent_id, avg_velocity, self.small_speed, len(slow_neighbors))
                    self.logger.logger.debug(f"   Slow neighbors: {slow_neighbors}")
                    self.logger.logger.debug(f"   Non-progress neighbors (>= {self.required_non_progress_neighbors}): {non_progress_neighbors}")
                    self.logger.logger.debug(f"   Risk neighbors (ttc<= {self.ttc_threshold} or dmin<= {self.dmin_threshold}): {risk_neighbors}")
                return True
            # Fallback for sparse neighborhoods: disabled per latest rule (require explicit at least 1 non-progress neighbor)
            # if cond_neighbors_slow and neighbors_with_sufficient_history < 2 and len(non_progress_neighbors) >= 1:
            #     if self.logger:
            #         self.logger.logger.debug(
            #             f"   Sparse neighborhood fallback: considered_neighbors={neighbors_with_sufficient_history}, non_progress_neighbors={len(non_progress_neighbors)} -> trigger=True"
            #         )
            #     return True
            else:
                if self.logger:
                    self.logger.logger.debug(
                        f"🔍 SPEED BUFFER NEW RULES: agent {agent_id} avg={avg_velocity:.3f} < {self.small_speed}, "
                        f"neighbors_with_history={neighbors_with_sufficient_history}, slow_neighbors={len(slow_neighbors)}, "
                        f"non_progress_neighbors={len(non_progress_neighbors)}, risk_neighbors={len(risk_neighbors)} "
                        f"-> trigger={cond_neighbors_slow and cond_non_progress and cond_risk}"
                    )
                
                # Single agent trigger as fallback: if agent is slow, not progressing, has few neighbors, and has been stuck for long time
                if not (cond_neighbors_slow and cond_non_progress and cond_risk):
                    single_agent_time_threshold = self.config.get('SINGLE_AGENT_TIME_THRESHOLD', 50)  # Default 50 steps
                    if self.step_counter >= single_agent_time_threshold:
                        # Check if agent is not progressing toward goal
                        if self._is_not_progressing_toward_goal(agent_state):
                            # Count active neighbors (not arrived at goal)
                            active_neighbors = []
                            for neighbor_id in neighbors_comm:
                                try:
                                    if self.calculate_distance_to_goal(agent_states.get(neighbor_id, {})) > float(self.goal_tolerance):
                                        active_neighbors.append(neighbor_id)
                                except Exception:
                                    pass
                            
                            # Check if agent has few active neighbors
                            max_neighbors_for_single_trigger = self.config.get('MAX_NEIGHBORS_FOR_SINGLE_TRIGGER', 1)
                            if len(active_neighbors) <= max_neighbors_for_single_trigger:
                                if self.logger:
                                    self.logger.logger.debug(f"🔍 SINGLE AGENT FALLBACK TRIGGER: Agent {agent_id} avg_velocity={avg_velocity:.3f} < {self.small_speed}, "
                                                           f"step={self.step_counter} >= {single_agent_time_threshold}, "
                                                           f"active_neighbors={len(active_neighbors)} <= {max_neighbors_for_single_trigger}, "
                                                           f"not_progressing=True")
                                return True
                            else:
                                if self.logger:
                                    self.logger.logger.debug(f"🔍 SINGLE AGENT FALLBACK: Agent {agent_id} has too many active neighbors: {len(active_neighbors)} > {max_neighbors_for_single_trigger}")
                        else:
                            if self.logger:
                                self.logger.logger.debug(f"🔍 SINGLE AGENT FALLBACK: Agent {agent_id} is progressing toward goal")
                    else:
                        if self.logger:
                            self.logger.logger.debug(f"🔍 SINGLE AGENT FALLBACK: Agent {agent_id} step {self.step_counter} < {single_agent_time_threshold}")
        else:
            if self.logger:
                self.logger.logger.debug(f"🔍 VELOCITY CHECK: Agent {agent_id} velocity={avg_velocity:.3f} >= {self.small_speed}")
        
        return False

    def _get_neighbors_in_range(self, agent_id: int, agent_states: Dict, radius: float) -> List[int]:
        """Return list of neighbor ids within given Euclidean radius from agent_id."""
        if agent_id not in agent_states or 'position' not in agent_states[agent_id]:
            return []
        pos = agent_states[agent_id]['position']
        if not isinstance(pos, (list, np.ndarray)) or len(pos) < 2:
            return []
        ax, ay = float(pos[0]), float(pos[1])
        r2 = radius * radius
        neighbors: List[int] = []
        for other_id, st in agent_states.items():
            if other_id == agent_id:
                continue
            if 'position' not in st:
                continue
            op = st['position']
            if not isinstance(op, (list, np.ndarray)) or len(op) < 2:
                continue
            dx = float(op[0]) - ax
            dy = float(op[1]) - ay
            if dx*dx + dy*dy <= r2:
                neighbors.append(other_id)
        return neighbors

    def _is_not_progressing_toward_goal(self, agent_state: Dict) -> bool:
        """Return True if the agent's current velocity does not point toward its goal (or near zero)."""
        try:
            if not isinstance(agent_state, dict):
                return False
            # Exclude if no valid position/goal
            if 'position' not in agent_state or 'goal' not in agent_state or 'velocity' not in agent_state:
                return False
            pos = agent_state['position']
            vel = agent_state['velocity']
            goal = agent_state['goal']
            if not (isinstance(pos, (list, np.ndarray)) and len(pos) >= 2):
                return False
            if not (isinstance(vel, (list, np.ndarray)) and len(vel) >= 2):
                return False
            # Parse goal that may be [x,y] or [[x],[y],[theta]]
            gx = goal[0][0] if isinstance(goal[0], (list, np.ndarray)) and len(goal[0]) > 0 else goal[0]
            gy = goal[1][0] if isinstance(goal[1], (list, np.ndarray)) and len(goal[1]) > 0 else goal[1]
            dx = float(gx) - float(pos[0])
            dy = float(gy) - float(pos[1])
            gnorm = math.sqrt(dx*dx + dy*dy)
            # If extremely close to goal, treat as progressing False to avoid blocking
            if gnorm <= float(self.goal_tolerance):
                return False
            # Current velocity magnitude small -> not progressing
            vx = float(vel[0])
            vy = float(vel[1])
            speed = math.sqrt(vx*vx + vy*vy)
            if speed < self.small_speed:
                return True
            # Check projection of velocity on goal direction
            proj = (vx * dx + vy * dy) / max(gnorm, 1e-6)
            return proj <= 0.0
        except Exception:
            return False
    
    def get_agent_position(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        """
        Extract agent position from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Optional[Tuple[float, float]]: Agent position (x, y) or None if not available
        """
        if 'position' in agent_state:
            position = agent_state['position']
            if isinstance(position, (list, np.ndarray)) and len(position) >= 2:
                return (float(position[0]), float(position[1]))
        
        # Try alternative position fields
        for field in ['pos', 'location', 'pose']:
            if field in agent_state:
                pos_data = agent_state[field]
                if isinstance(pos_data, (list, np.ndarray)) and len(pos_data) >= 2:
                    return (float(pos_data[0]), float(pos_data[1]))
        
        return None
    
    def get_deadlock_participants(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> List[int]:
        """
        Get list of agents that should participate in deadlock resolution.
        
        Only includes agents that are actually at risk of collision or deadlock.
        
        Args:
            agent_id: ID of the agent that detected deadlock
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            List[int]: List of agent IDs that should participate in PAR
        """
        # If locked, return the locked set until expiry
        if self.participant_lock_steps > 0 and agent_id in self.locked_participants:
            locked_set, expires_at = self.locked_participants[agent_id]
            if self.step_counter <= expires_at:
                return locked_set
            else:
                self.locked_participants.pop(agent_id, None)
        
        # If current agent is at goal, exclude from participants
        try:
            if agent_id in agent_states:
                distance_to_goal = self.calculate_distance_to_goal(agent_states[agent_id])
                if distance_to_goal <= float(self.goal_tolerance):
                    return []
        except Exception:
            pass

        # 单人兜底：若仅剩该agent未到达，则直接返回单人集合
        try:
            tol = float(self.goal_tolerance)
        except Exception:
            tol = 0.5
        try:
            single_enabled = bool(self.config.get('SINGLE_AGENT_TRIGGER_ENABLED', True)) if isinstance(self.config, dict) else True
        except Exception:
            single_enabled = True
        if single_enabled:
            try:
                unfinished = [aid for aid, st in agent_states.items() if self.calculate_distance_to_goal(st) > tol]
            except Exception:
                unfinished = []
            if len(unfinished) == 1 and unfinished[0] == agent_id:
                return [agent_id]

        # 基于局部冲突图选择参与者：构图→取连通分量→规模上限裁剪→失败回退配对
        participants = [agent_id]
        if agent_id not in agent_states:
            return participants

        # 1) 构建局部冲突图（通信半径内，双向不前进且满足TTC/DMIN风险门槛）
        try:
            graph = self._build_local_conflict_graph(agent_id, agent_states)
        except Exception:
            graph = {}

        # 2) 取包含seed的连通分量
        try:
            component = self._extract_component(graph, agent_id)
        except Exception:
            component = set([agent_id])

        # 3) 去除已到达目标的节点
        filtered = []
        for nid in component:
            try:
                if self.calculate_distance_to_goal(agent_states.get(nid, {})) <= float(self.goal_tolerance):
                    continue
            except Exception:
                pass
            filtered.append(nid)

        # 4) 规模上限裁剪与优先级排序
        max_participants = int(self.config.get('MAX_PAR_PARTICIPANTS', 10))
        ordered = self._prioritize_component(filtered, agent_states) if len(filtered) > 0 else []
        clipped = ordered[:max(2, max_participants)] if len(ordered) > 0 else []

        # 5) 若裁剪后>=2，采用该集合；若==1且允许单人兜底，直接返回；否则回退到“最佳邻居配对/超时最近邻”
        if len(clipped) >= 2:
            participants = clipped
        elif len(clipped) == 1 and single_enabled:
            participants = clipped
        else:
            # 回退逻辑：沿用原有二人配对策略
            comm_range = float(self.config.get('COMMUNICATION_RANGE', 7.0))
            candidates = self._get_neighbors_in_range(agent_id, agent_states, comm_range)
            best_neighbor = None
            best_score = -1.0
            for neighbor_id in candidates:
                if neighbor_id == agent_id:
                    continue
                # Skip arrived neighbors
                try:
                    if self.calculate_distance_to_goal(agent_states.get(neighbor_id, {})) <= float(self.goal_tolerance):
                        continue
                except Exception:
                    pass
                # Require neighbor slow and not progressing
                n_avg = self.calculate_average_velocity(neighbor_id) if neighbor_id in self.velocity_history else float('inf')
                if not math.isfinite(n_avg) or n_avg >= self.small_speed:
                    continue
                if not self._is_not_progressing_toward_goal(agent_states.get(neighbor_id, {})):
                    continue
                # Prefer the closest-in-time collision (smallest TTC)
                m = self._compute_pair_metrics(agent_states.get(agent_id, {}), agent_states.get(neighbor_id, {}))
                ttc = m.get('ttc', float('inf'))
                score = (1.0 / max(ttc, 1e-6)) if math.isfinite(ttc) else 0.0
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor_id
            participants = [agent_id]
            if best_neighbor is not None:
                participants.append(best_neighbor)

            if self.logger:
                self.logger.logger.debug(f"🔍 PARTICIPANTS CHECK: Agent {agent_id}, participants={participants}, len={len(participants)}, step_counter={self.step_counter}, threshold={self.config.get('SINGLE_AGENT_TIME_THRESHOLD', 50)}")

            # 单体回退：如仍仅1人且超时，强制加入通信半径内最近者
            if len(participants) == 1 and self.step_counter >= self.config.get('SINGLE_AGENT_TIME_THRESHOLD', 50):
                candidates = self._get_neighbors_in_range(agent_id, agent_states, comm_range)
                closest_neighbor = None
                min_distance = float('inf')
                for neighbor_id in candidates:
                    if neighbor_id == agent_id:
                        continue
                    try:
                        if self.calculate_distance_to_goal(agent_states.get(neighbor_id, {})) <= float(self.goal_tolerance):
                            continue
                    except Exception:
                        pass
                    try:
                        agent_pos = self.get_agent_position(agent_states.get(agent_id, {}))
                        neighbor_pos = self.get_agent_position(agent_states.get(neighbor_id, {}))
                        if agent_pos and neighbor_pos:
                            distance = math.sqrt((agent_pos[0] - neighbor_pos[0])**2 + (agent_pos[1] - neighbor_pos[1])**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_neighbor = neighbor_id
                    except Exception:
                        pass
                if closest_neighbor is not None:
                    participants.append(closest_neighbor)
                    if self.logger:
                        self.logger.logger.debug(f"🔍 SINGLE AGENT FALLBACK: Forced inclusion of neighbor {closest_neighbor} for PAR execution")

        # 锁定若干步避免振荡
        if self.participant_lock_steps > 0 and len(participants) > 1:
            expires_at = self.step_counter + self.participant_lock_steps
            for pid in participants:
                self.locked_participants[pid] = (participants.copy(), expires_at)

        return list(dict.fromkeys(participants))

    def _build_local_conflict_graph(self, seed_id: int, agent_states: Dict) -> Dict[int, List[int]]:
        """基于多跳BFS的局部冲突图：
        - 从seed出发，按通信半径逐层扩展至 MAX_GRAPH_HOPS 层；
        - 仅纳入“活跃且未到达”的个体（有足够速度历史、低速且不朝目标推进）；
        - 每层限制新增候选数量，整体限制总节点数；
        - 图为无向，边依据 TTC/DMIN 风险阈值建立。
        """
        graph: Dict[int, List[int]] = {}
        if seed_id not in agent_states:
            return graph

        comm_range = float(self.config.get('COMMUNICATION_RANGE', 7.0))
        max_hops = int(self.config.get('MAX_GRAPH_HOPS', 2))
        per_hop_cap = int(self.config.get('MAX_CANDIDATES_PER_HOP', 12))
        max_nodes = int(self.config.get('MAX_GRAPH_NODES', 20))
        min_hist = int(self.velocity_window_size * 0.8)

        def is_arrived(nid: int) -> bool:
            try:
                return self.calculate_distance_to_goal(agent_states.get(nid, {})) <= float(self.goal_tolerance)
            except Exception:
                return False

        def is_active(nid: int) -> bool:
            if is_arrived(nid):
                return False
            if nid not in self.velocity_history or len(self.velocity_history[nid]) < min_hist:
                return False
            avg = self.calculate_average_velocity(nid)
            if not (math.isfinite(avg) and avg < self.small_speed):
                return False
            return self._is_not_progressing_toward_goal(agent_states.get(nid, {}))

        # 多跳BFS：仅通过“活跃”前沿扩展
        visited = set()
        active_nodes: List[int] = []

        # 确保seed进入图（即便seed不满足active，也保留其节点用于连通分量提取）
        active_nodes.append(seed_id)

        frontier = [seed_id]
        hops = 0
        while frontier and hops < max_hops and len(active_nodes) < max_nodes:
            next_candidates: List[int] = []
            for u in frontier:
                if u in visited:
                    continue
                visited.add(u)
                # 邻域内候选
                nbrs = self._get_neighbors_in_range(u, agent_states, comm_range)
                for v in nbrs:
                    if v in visited:
                        continue
                    if v in active_nodes:
                        continue
                    # 仅把“活跃未到达”的加入下一层候选
                    if is_active(v):
                        next_candidates.append(v)
            # 去重并按容量裁剪
            dedup = []
            seen = set()
            for nid in next_candidates:
                if nid not in seen and nid not in active_nodes:
                    dedup.append(nid)
                    seen.add(nid)
            # 控制本层新增与总量上限
            remaining = max(0, max_nodes - len(active_nodes))
            take = min(per_hop_cap, remaining)
            dedup = dedup[:take]
            # 纳入活跃节点集
            active_nodes.extend(dedup)
            # 下一层前沿
            frontier = dedup
            hops += 1

        # 构建无向图：按风险阈值加边（在active_nodes内，两两评估）
        # 若seed并非active，但已被强制纳入，则其与他人也可建立边
        for nid in active_nodes:
            graph[nid] = []

        th_ttc = float(self.ttc_threshold)
        th_dmin = float(self.dmin_threshold)
        for i in range(len(active_nodes)):
            a = active_nodes[i]
            for j in range(i + 1, len(active_nodes)):
                b = active_nodes[j]
                # 跳过已到达者（健壮性双重过滤）
                if is_arrived(a) or is_arrived(b):
                    continue
                m = self._compute_pair_metrics(agent_states.get(a, {}), agent_states.get(b, {}))
                ttc = m.get('ttc', float('inf'))
                dmin = m.get('dmin', float('inf'))
                if (math.isfinite(ttc) and ttc <= th_ttc) or (math.isfinite(dmin) and dmin <= th_dmin):
                    graph[a].append(b)
                    graph[b].append(a)

        return graph

    def _extract_component(self, graph: Dict[int, List[int]], seed_id: int) -> set:
        """提取包含seed的连通分量。若seed不在图中，返回仅含seed的集合。"""
        if seed_id not in graph:
            return set([seed_id])
        seen = set()
        stack = [seed_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in graph.get(cur, []):
                if nxt not in seen:
                    stack.append(nxt)
        return seen

    def _prioritize_component(self, nodes: List[int], agent_states: Dict) -> List[int]:
        """优先级排序并返回列表：seed优先、与seed的TTC小者优先、距离近者优先。"""
        if not isinstance(nodes, list) or len(nodes) == 0:
            return []
        # 确保包含seed时把seed放最前
        def seed_first(nid: int) -> int:
            return 0 if nid == nodes[0] else 1
        seed_id = nodes[0] if len(nodes) > 0 else None
        # 预计算与seed的metrics
        metrics = {}
        for nid in nodes:
            if seed_id is None:
                metrics[nid] = (float('inf'), float('inf'))
                continue
            m = self._compute_pair_metrics(agent_states.get(seed_id, {}), agent_states.get(nid, {}))
            ttc = m.get('ttc', float('inf'))
            dmin = m.get('dmin', float('inf'))
            metrics[nid] = (ttc, dmin)
        # 排序：seed在前；ttc升序；dmin升序
        ordered = sorted(nodes, key=lambda nid: (seed_first(nid), metrics[nid][0], metrics[nid][1]))
        return ordered
    
    def reset_episode(self):
        """Reset velocity history for new episode."""
        self.velocity_history.clear()
        self.last_deadlock_detection.clear()  # Reset cooldown state for new episode
        self.episode_counter += 1
        self.step_counter = 0  # Reset step counter for new episode
        # Reset waypoint-stuck histories
        self._last_wp_index.clear()
        self._last_wp_change_step.clear()
        if self.logger:
            # self.logger.logger.info(f"🔄 Episode {self.episode_counter}: Reset velocity history and step counter for deadlock detection")
            pass
        else:
            # print(f"🔄 Episode {self.episode_counter}: Reset velocity history and step counter for deadlock detection")
            pass
    
    def calculate_current_velocity(self, agent_state: Dict) -> float:
        """
        Calculate current velocity magnitude from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            float: Current velocity magnitude
        """
        if 'velocity' in agent_state:
            velocity = agent_state['velocity']
            if isinstance(velocity, (list, np.ndarray)) and len(velocity) >= 2:
                return math.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Fallback: calculate from position change if available
        if 'position' in agent_state and 'prev_position' in agent_state:
            pos = agent_state['position']
            prev_pos = agent_state['prev_position']
            if isinstance(pos, (list, np.ndarray)) and isinstance(prev_pos, (list, np.ndarray)):
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                return math.sqrt(dx**2 + dy**2)
        
        return 0.0
    
    def update_velocity_history(self, agent_id: int, velocity: float):
        """
        Update velocity history for an agent.
        
        Args:
            agent_id: ID of the agent
            velocity: Current velocity value
        """
        if agent_id not in self.velocity_history:
            self.velocity_history[agent_id] = []
            
        self.velocity_history[agent_id].append(velocity)
        
        # Keep only the last window_size entries
        if len(self.velocity_history[agent_id]) > self.velocity_window_size:
            self.velocity_history[agent_id] = self.velocity_history[agent_id][-self.velocity_window_size:]
    
    def _update_all_velocity_histories(self, agent_id: int, agent_states: Dict, neighbor_states: Dict):
        """
        Update velocity history for current agent and all neighbors.
        
        Args:
            agent_id: ID of the current agent
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states
        """
        # Update current agent velocity history
        if agent_id in agent_states:
            current_velocity = self.calculate_current_velocity(agent_states[agent_id])
            self.update_velocity_history(agent_id, current_velocity)
        
        # Update neighbor velocity histories
        for neighbor_id, neighbor_state in neighbor_states.items():
            current_velocity = self.calculate_current_velocity(neighbor_state)
            self.update_velocity_history(neighbor_id, current_velocity)
    
    def calculate_average_velocity(self, agent_id: int) -> float:
        """
        Calculate average velocity for an agent over the history window.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            float: Average velocity over the history window
        """
        if agent_id not in self.velocity_history or not self.velocity_history[agent_id]:
            return 0.0
        
        velocities = self.velocity_history[agent_id]
        return sum(velocities) / len(velocities)
    
    def calculate_distance_to_goal(self, agent_state: Dict) -> float:
        """
        Calculate distance from agent to its goal.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            float: Distance to goal
        """
        if 'position' in agent_state and 'goal' in agent_state:
            pos = agent_state['position']
            goal = agent_state['goal']
            if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                try:
                    # Support goals like [x, y] or [[x],[y],[theta]]
                    if isinstance(goal, (list, np.ndarray)) and len(goal) >= 2:
                        gx = goal[0][0] if isinstance(goal[0], (list, np.ndarray)) and len(goal[0]) > 0 else goal[0]
                        gy = goal[1][0] if isinstance(goal[1], (list, np.ndarray)) and len(goal[1]) > 0 else goal[1]
                        dx = float(pos[0]) - float(gx)
                        dy = float(pos[1]) - float(gy)
                        return math.sqrt(dx*dx + dy*dy)
                except Exception:
                    pass
        
        return float('inf')
    
    def reset_agent_history(self, agent_id: int):
        """
        Reset velocity history for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.velocity_history:
            self.velocity_history[agent_id].clear()
    
    def reset_all_history(self):
        """Reset velocity history for all agents."""
        self.velocity_history.clear()
    
    
    
    

    @staticmethod
    def _compute_pair_metrics(a_state: Dict, b_state: Dict) -> Dict[str, float]:
        """Compute TTC and minimum distance for two agents based on relative motion."""
        ax, ay = 0.0, 0.0
        avx, avy = 0.0, 0.0
        bx, by = 0.0, 0.0
        bvx, bvy = 0.0, 0.0
        if a_state and 'position' in a_state and isinstance(a_state['position'], (list, np.ndarray)) and len(a_state['position']) >= 2:
            ax, ay = float(a_state['position'][0]), float(a_state['position'][1])
        if a_state and 'velocity' in a_state and isinstance(a_state['velocity'], (list, np.ndarray)) and len(a_state['velocity']) >= 2:
            avx, avy = float(a_state['velocity'][0]), float(a_state['velocity'][1])
        if b_state and 'position' in b_state and isinstance(b_state['position'], (list, np.ndarray)) and len(b_state['position']) >= 2:
            bx, by = float(b_state['position'][0]), float(b_state['position'][1])
        if b_state and 'velocity' in b_state and isinstance(b_state['velocity'], (list, np.ndarray)) and len(b_state['velocity']) >= 2:
            bvx, bvy = float(b_state['velocity'][0]), float(b_state['velocity'][1])
        rx, ry = bx - ax, by - ay
        vx, vy = bvx - avx, bvy - avy
        v2 = vx * vx + vy * vy
        dot = rx * vx + ry * vy
        closing = dot < 0.0
        eps = 1e-6
        if v2 <= eps:
            ttc = float('inf')
            dmin = math.sqrt(rx * rx + ry * ry)
        else:
            t_star = -dot / v2
            if t_star <= 0.0:
                ttc = float('inf')
                dmin = math.sqrt(rx * rx + ry * ry)
            else:
                ttc = t_star
                dx = rx + vx * t_star
                dy = ry + vy * t_star
                dmin = math.sqrt(dx * dx + dy * dy)
        return {'ttc': ttc, 'dmin': dmin, 'closing': 1.0 if closing else 0.0}
