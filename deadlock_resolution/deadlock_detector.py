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
        
        # Unified detection parameters
        self.risk_ttc_threshold = config.get('RISK_TTC_THRESHOLD', 1.0)
        self.risk_dmin_threshold = config.get('RISK_DMIN_THRESHOLD', 0.6)
        self.risk_weights = config.get('RISK_WEIGHTS', {'ttc': 1.0, 'dmin': 0.5})
        self.core_pair_only = config.get('CORE_PAIR_ONLY', True)
        self.consensus_jaccard_threshold = config.get('CONSENSUS_JACCARD_THRESHOLD', 0.5)
        self.max_core_pairs_per_step = config.get('MAX_CORE_PAIRS_PER_STEP', 1)
        
        # Goal tolerance for arrival check
        self.goal_tolerance = config.get('GOAL_TOLERANCE', 0.1)
        
        # Step-cache for per-step computations
        self._cache_step = -1
        self._metrics_cache: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._best_partner_cache: Dict[int, Tuple[int, float]] = {}
        self._group_cache: Dict[int, List[int]] = {}
        # Global velocity history update guard per step
        self._velhist_step = -1
        
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
            
        # Unified trigger path
        if self.trigger_type == 'UNIFIED':
            return self._unified_detect_deadlock(agent_id, agent_states)
        
        if self.trigger_type == 'SPEED_BUFFER':
            result = self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        else:
            # Default to speed buffer trigger
            result = self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        
        # Log detection result and update cooldown
        if result:
            # Update cooldown time for this agent
            self.last_deadlock_detection[agent_id] = self.step_counter
            if self.logger:
                self.logger.log_deadlock_detection(agent_id, self.trigger_type, {
                    'agent_states': {k: v for k, v in agent_states.items() if k == agent_id},
                    'neighbor_count': len(neighbor_states),
                    'trigger_type': self.trigger_type
                })
        else:
            if self.logger:
                self.logger.logger.debug(f"✅ NO DEADLOCK: Agent {agent_id}")
        
        return result
    
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
            # Condition 3: at least one adjacent neighbor cannot progress toward goal
            cond_non_progress = len(non_progress_neighbors) >= 1

            if cond_neighbors_slow and cond_non_progress:
                if self.logger:
                    self.logger.log_deadlock_check(agent_id, avg_velocity, self.small_speed, len(slow_neighbors))
                    self.logger.logger.debug(f"   Slow neighbors: {slow_neighbors}")
                    self.logger.logger.debug(f"   Non-progress neighbors (>=2): {non_progress_neighbors}")
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
                        f"non_progress_neighbors={len(non_progress_neighbors)} -> trigger={cond_neighbors_slow and cond_non_progress}"
                    )
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

        # Unified participant selection path
        if self.trigger_type == 'UNIFIED':
            participants = self._unified_get_participants(agent_id, agent_states)
        else:
            # Fallback redefined: select participants using COMMUNICATION_RANGE, not upper neighbor_states
            participants = [agent_id]
            if agent_id not in agent_states:
                return participants
            # Build candidates within communication range
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
                # Prefer the closest-in-time collision (smallest TTC) if available
                m = self._compute_pair_metrics(agent_states.get(agent_id, {}), agent_states.get(neighbor_id, {}))
                ttc = m.get('ttc', float('inf'))
                score = (1.0 / max(ttc, 1e-6)) if math.isfinite(ttc) else 0.0
                if score > best_score:
                    best_score = score
                    best_neighbor = neighbor_id
            if best_neighbor is not None:
                participants.append(best_neighbor)
            participants = list(dict.fromkeys(participants))
        
        # Lock selected participants for a few steps to avoid oscillation
        if self.participant_lock_steps > 0 and len(participants) > 1:
            expires_at = self.step_counter + self.participant_lock_steps
            for pid in participants:
                self.locked_participants[pid] = (participants.copy(), expires_at)
        
        return participants
    
    def reset_episode(self):
        """Reset velocity history for new episode."""
        self.velocity_history.clear()
        self.last_deadlock_detection.clear()  # Reset cooldown state for new episode
        self.episode_counter += 1
        self.step_counter = 0  # Reset step counter for new episode
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
    
    
    
    

    # ---------------------- Unified detection helpers ----------------------
    def _unified_detect_deadlock(self, agent_id: int, agent_states: Dict) -> bool:
        """Unified deadlock trigger based on TTC/d_min risk for the given agent."""
        if agent_id not in agent_states:
            return False
        self._ensure_step_cache()
        partner_id, risk, metrics = self._get_core_pair_for_agent(agent_id, agent_states)
        if partner_id is not None:
            ttc = metrics.get('ttc', float('inf'))
            dmin = metrics.get('dmin', float('inf'))
            return (ttc < self.risk_ttc_threshold) or (dmin < self.risk_dmin_threshold)
        # No core pair; check if there is a high-risk consensus group
        group = self._build_consensus_group(agent_id, agent_states)
        return len(group) > 1

    def _unified_get_participants(self, agent_id: int, agent_states: Dict) -> List[int]:
        """Return participants relying solely on unified risk logic (idempotent across calls)."""
        self._ensure_step_cache()
        partner_id, risk, metrics = self._get_core_pair_for_agent(agent_id, agent_states)
        if partner_id is not None:
            return [agent_id, partner_id]
        group = self._build_consensus_group(agent_id, agent_states)
        if len(group) > 1:
            return sorted(list(group))
        return [agent_id]

    def _ensure_step_cache(self):
        """Reset per-step caches if step counter advanced."""
        if self._cache_step != self.step_counter:
            self._cache_step = self.step_counter
            self._metrics_cache.clear()
            self._best_partner_cache.clear()
            self._group_cache.clear()

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

    def _get_best_partner_for(self, agent_id: int, agent_states: Dict) -> Tuple[Optional[int], float, Dict[str, float]]:
        """Find the best risk partner for an agent across all other agents."""
        if agent_id in self._best_partner_cache:
            partner_id, risk = self._best_partner_cache[agent_id]
            metrics = {}
            if partner_id is not None:
                key = (min(agent_id, partner_id), max(agent_id, partner_id))
                metrics = self._metrics_cache.get(key, {})
            return partner_id, risk, metrics
        best_partner = None
        best_risk = -1.0
        best_metrics = {}
        weights = self.risk_weights if isinstance(self.risk_weights, dict) else {'ttc': 1.0, 'dmin': 0.5}
        for other_id, other_state in agent_states.items():
            if other_id == agent_id:
                continue
            key = (min(agent_id, other_id), max(agent_id, other_id))
            if key not in self._metrics_cache:
                self._metrics_cache[key] = self._compute_pair_metrics(agent_states[agent_id], other_state)
            m = self._metrics_cache[key]
            if m.get('closing', 0.0) <= 0.0:
                continue
            ttc = m.get('ttc', float('inf'))
            dmin = m.get('dmin', float('inf'))
            inv_ttc = 0.0 if not math.isfinite(ttc) else 1.0 / max(ttc, 1e-6)
            inv_dmin = 0.0 if not math.isfinite(dmin) else 1.0 / max(dmin, 1e-6)
            risk = float(weights.get('ttc', 1.0)) * inv_ttc + float(weights.get('dmin', 0.5)) * inv_dmin
            if risk > best_risk:
                best_risk = risk
                best_partner = other_id
                best_metrics = m
        self._best_partner_cache[agent_id] = (best_partner, best_risk)
        return best_partner, best_risk, best_metrics

    def _get_core_pair_for_agent(self, agent_id: int, agent_states: Dict) -> Tuple[Optional[int], float, Dict[str, float]]:
        """Return reciprocal best partner as core pair if exists; else None."""
        a_partner, a_risk, a_metrics = self._get_best_partner_for(agent_id, agent_states)
        if a_partner is None:
            return None, -1.0, {}
        b_partner, b_risk, _ = self._get_best_partner_for(a_partner, agent_states)
        if b_partner == agent_id:
            return a_partner, min(a_risk, b_risk), a_metrics
        return None, -1.0, {}

    def _build_consensus_group(self, agent_id: int, agent_states: Dict) -> List[int]:
        """Build a small consensus group around the agent using mutual-best closure."""
        if agent_id in self._group_cache:
            return self._group_cache[agent_id]
        group = set([agent_id])
        # seed with agent's best partner if above thresholds
        partner_id, risk, metrics = self._get_core_pair_for_agent(agent_id, agent_states)
        if partner_id is not None:
            ttc = metrics.get('ttc', float('inf'))
            dmin = metrics.get('dmin', float('inf'))
            if (ttc < self.risk_ttc_threshold) or (dmin < self.risk_dmin_threshold):
                group.add(partner_id)
        # attempt to include high-risk neighbors whose best partner lies within current group
        expanded = True
        while expanded:
            expanded = False
            for other_id in agent_states.keys():
                if other_id in group:
                    continue
                best, risk_o, metrics_o = self._get_best_partner_for(other_id, agent_states)
                if best in group:
                    ttc = metrics_o.get('ttc', float('inf'))
                    dmin = metrics_o.get('dmin', float('inf'))
                    if (ttc < self.risk_ttc_threshold) or (dmin < self.risk_dmin_threshold):
                        group.add(other_id)
                        expanded = True
        result = sorted(list(group))
        self._group_cache[agent_id] = result
        return result
