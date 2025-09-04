"""
Deadlock Detector Module

This module provides deadlock detection functionality for RL_RVO navigation system.
It implements two trigger mechanisms: speed buffer trigger and common point trigger.
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
        self.sight_radius = config.get('SIGHT_RADIUS', 2.0)
        self.velocity_window_size = config.get('VELOCITY_WINDOW_SIZE', 5)
        
        # Hybrid detection parameters
        self.hybrid_mode = config.get('HYBRID_MODE', 'AND')
        self.speed_buffer_weight = config.get('SPEED_BUFFER_WEIGHT', 0.6)
        self.common_point_weight = config.get('COMMON_POINT_WEIGHT', 0.4)
        
        # Enhanced speed buffer parameters
        self.speed_buffer_avg_threshold = config.get('SPEED_BUFFER_AVG_THRESHOLD', 0.1)
        self.speed_buffer_max_threshold = config.get('SPEED_BUFFER_MAX_THRESHOLD', 0.2)
        self.speed_buffer_min_history_ratio = config.get('SPEED_BUFFER_MIN_HISTORY_RATIO', 0.8)
        
        # Velocity history for each agent
        self.velocity_history = defaultdict(list)
        
        # Deadlock detection state
        self.deadlock_detection_enabled = config.get('DEADLOCK_DETECTION_ENABLED', True)
        
        # Episode counter for debugging
        self.episode_counter = 0
        
        # Episode start delay for deadlock detection
        self.episode_start_delay = config.get('EPISODE_START_DELAY', 50)
        self.step_counter = 0
        
        # Cooldown mechanism for deadlock detection
        self.deadlock_detection_cooldown = config.get('DEADLOCK_DETECTION_COOLDOWN', 50)
        self.last_deadlock_detection = {}  # Track last detection time for each agent
        
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
        
        # Increment step counter
        self.step_counter += 1
        
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
        
        # Update velocity history for all agents (current agent and neighbors)
        self._update_all_velocity_histories(agent_id, agent_states, neighbor_states)
            
        if self.trigger_type == 'SPEED_BUFFER':
            result = self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        elif self.trigger_type == 'COMMON_POINT':
            result = self.check_common_point_trigger(agent_id, agent_states, neighbor_states)
        elif self.trigger_type == 'HYBRID':
            result = self.check_hybrid_trigger(agent_id, agent_states, neighbor_states)
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
        
        # Calculate average velocity
        avg_velocity = self.calculate_average_velocity(agent_id)
        
        # Check if agent velocity is below threshold
        if avg_velocity < self.small_speed:
            # Check neighbor velocities - only consider neighbors with sufficient history
            neighbor_low_velocity_count = 0
            low_velocity_neighbors = []
            neighbors_with_sufficient_history = 0
            
            for neighbor_id, neighbor_state in neighbor_states.items():
                # Check if neighbor has sufficient velocity history
                if neighbor_id in self.velocity_history and len(self.velocity_history[neighbor_id]) >= min_history_required:
                    neighbors_with_sufficient_history += 1
                    neighbor_avg_velocity = self.calculate_average_velocity(neighbor_id)
                    if neighbor_avg_velocity < self.small_speed:
                        neighbor_low_velocity_count += 1
                        low_velocity_neighbors.append(neighbor_id)
            
            # Only trigger if we have at least one neighbor with sufficient history and low velocity
            if neighbors_with_sufficient_history > 0 and neighbor_low_velocity_count > 0:
                if self.logger:
                    self.logger.log_deadlock_check(agent_id, avg_velocity, self.small_speed, neighbor_low_velocity_count)
                    self.logger.logger.debug(f"   Low velocity neighbors: {low_velocity_neighbors}")
                    self.logger.logger.debug(f"   Neighbors with sufficient history: {neighbors_with_sufficient_history}")
                else:
                    # print(f"🔍 DEADLOCK CHECK: Agent {agent_id} velocity={avg_velocity:.3f} < {self.small_speed}, {neighbor_low_velocity_count} neighbors also slow: {low_velocity_neighbors}")
                    pass
                return True
            else:
                if self.logger:
                    self.logger.logger.debug(f"🔍 INSUFFICIENT NEIGHBOR HISTORY: Agent {agent_id}, neighbors with history: {neighbors_with_sufficient_history}, low velocity neighbors: {neighbor_low_velocity_count}")
        else:
            if self.logger:
                self.logger.logger.debug(f"🔍 VELOCITY CHECK: Agent {agent_id} velocity={avg_velocity:.3f} >= {self.small_speed}")
        
        return False
    
    def check_common_point_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using common point trigger mechanism.
        
        Based on the orca_mapf paper implementation:
        - Check if number of neighbors >= MAPFNum
        - Check if distance to goal < sightRadius
        - Also apply episode start delay and velocity history requirements for stability
        
        Args:
            agent_id: ID of the agent to check
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if deadlock is detected based on common point
        """
        # Get current agent state
        if agent_id not in agent_states:
            return False
            
        agent_state = agent_states[agent_id]
        
        # Check if we have enough velocity history data (at least 80% of window size for stability)
        min_history_required = int(self.velocity_window_size * 0.8)
        if len(self.velocity_history[agent_id]) < min_history_required:
            if self.logger:
                self.logger.logger.debug(f"🔍 COMMON POINT - INSUFFICIENT HISTORY: Agent {agent_id} has {len(self.velocity_history[agent_id])} velocity samples, need at least {min_history_required}")
            return False
        
        # Check if number of neighbors >= MAPFNum
        neighbor_count = len(neighbor_states)
        if neighbor_count < self.mapf_num:
            if self.logger:
                self.logger.logger.debug(f"🔍 COMMON POINT CHECK: Agent {agent_id} has {neighbor_count} neighbors (need >= {self.mapf_num})")
            return False
        
        # Check if distance to goal < sightRadius
        distance_to_goal = self.calculate_distance_to_goal(agent_state)
        if distance_to_goal >= self.sight_radius:
            if self.logger:
                self.logger.logger.debug(f"🔍 COMMON POINT CHECK: Agent {agent_id} distance to goal {distance_to_goal:.3f} >= {self.sight_radius}")
            return False
        
        # Both conditions met: trigger deadlock detection
        if self.logger:
            self.logger.logger.debug(f"🔍 COMMON POINT CHECK: Agent {agent_id} triggered - {neighbor_count} neighbors, distance to goal {distance_to_goal:.3f} < {self.sight_radius}")
        else:
            # print(f"🔍 COMMON POINT CHECK: Agent {agent_id} triggered - {neighbor_count} neighbors, distance to goal {distance_to_goal:.3f} < {self.sight_radius}")
            pass
        
        return True
    
    def get_deadlock_participants(self, agent_id: int, neighbor_states: Dict) -> List[int]:
        """
        Get list of agents that should participate in deadlock resolution.
        
        Args:
            agent_id: ID of the agent that detected deadlock
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            List[int]: List of agent IDs that should participate in PAR
        """
        participants = [agent_id]  # Always include the detecting agent
        
        # Add direct neighbors
        for neighbor_id in neighbor_states.keys():
            participants.append(neighbor_id)
        
        # Add neighbors of neighbors (N(N(i)))
        for neighbor_id in neighbor_states.keys():
            # This would require access to neighbor's neighbors
            # For now, we'll include direct neighbors only
            # TODO: Implement recursive neighbor collection if needed
            pass
        
        return list(set(participants))  # Remove duplicates
    
    def reset_episode(self):
        """Reset velocity history for new episode."""
        self.velocity_history.clear()
        self.episode_counter += 1
        self.step_counter = 0  # Reset step counter for new episode
        if self.logger:
            self.logger.logger.info(f"🔄 Episode {self.episode_counter}: Reset velocity history and step counter for deadlock detection")
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
            
            if isinstance(pos, (list, np.ndarray)) and isinstance(goal, (list, np.ndarray)):
                dx = pos[0] - goal[0]
                dy = pos[1] - goal[1]
                return math.sqrt(dx**2 + dy**2)
        
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
    
    def check_hybrid_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using hybrid trigger mechanism.
        
        Combines speed buffer and common point triggers based on configuration:
        - AND mode: Both conditions must be met
        - OR mode: Either condition can trigger
        
        Args:
            agent_id: ID of the agent to check
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if deadlock is detected based on hybrid criteria
        """
        if self.logger:
            self.logger.logger.debug(f"🔍 HYBRID TRIGGER: Agent {agent_id}, Mode: {self.hybrid_mode}")
        
        # Check speed buffer trigger
        speed_buffer_result = self.check_enhanced_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        
        # Check common point trigger
        common_point_result = self.check_common_point_trigger(agent_id, agent_states, neighbor_states)
        
        # Combine results based on hybrid mode
        if self.hybrid_mode == 'AND':
            result = speed_buffer_result and common_point_result
            if self.logger:
                self.logger.logger.debug(f"🔍 HYBRID AND: Speed buffer: {speed_buffer_result}, Common point: {common_point_result}, Result: {result}")
        else:  # OR mode
            result = speed_buffer_result or common_point_result
            if self.logger:
                self.logger.logger.debug(f"🔍 HYBRID OR: Speed buffer: {speed_buffer_result}, Common point: {common_point_result}, Result: {result}")
        
        return result
    
    def check_enhanced_speed_buffer_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using enhanced speed buffer trigger mechanism.
        
        Enhanced version that considers both average and maximum velocity over time window:
        - Average velocity must be below SPEED_BUFFER_AVG_THRESHOLD
        - Maximum velocity must be below SPEED_BUFFER_MAX_THRESHOLD
        - Both current agent and neighbors must meet criteria
        
        Args:
            agent_id: ID of the agent to check
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if deadlock is detected based on enhanced speed criteria
        """
        # Get current agent state
        if agent_id not in agent_states:
            return False
            
        agent_state = agent_states[agent_id]
        
        # Check if we have enough velocity history data
        min_history_required = int(self.velocity_window_size * self.speed_buffer_min_history_ratio)
        if len(self.velocity_history[agent_id]) < min_history_required:
            if self.logger:
                self.logger.logger.debug(f"🔍 ENHANCED SPEED BUFFER - INSUFFICIENT HISTORY: Agent {agent_id} has {len(self.velocity_history[agent_id])} velocity samples, need at least {min_history_required}")
            return False
        
        # Calculate enhanced velocity metrics
        avg_velocity = self.calculate_average_velocity(agent_id)
        max_velocity = max(self.velocity_history[agent_id])
        
        # Check if agent meets enhanced velocity criteria
        if avg_velocity >= self.speed_buffer_avg_threshold or max_velocity >= self.speed_buffer_max_threshold:
            if self.logger:
                self.logger.logger.debug(f"🔍 ENHANCED SPEED BUFFER - AGENT OK: Agent {agent_id} avg_vel={avg_velocity:.3f} (>= {self.speed_buffer_avg_threshold}), max_vel={max_velocity:.3f} (>= {self.speed_buffer_max_threshold})")
            return False
        
        # Check neighbor velocities - only consider neighbors with sufficient history
        neighbor_low_velocity_count = 0
        low_velocity_neighbors = []
        neighbors_with_sufficient_history = 0
        
        for neighbor_id, neighbor_state in neighbor_states.items():
            # Check if neighbor has sufficient velocity history
            if neighbor_id in self.velocity_history and len(self.velocity_history[neighbor_id]) >= min_history_required:
                neighbors_with_sufficient_history += 1
                neighbor_avg_velocity = self.calculate_average_velocity(neighbor_id)
                neighbor_max_velocity = max(self.velocity_history[neighbor_id])
                
                # Check if neighbor meets enhanced velocity criteria
                if neighbor_avg_velocity < self.speed_buffer_avg_threshold and neighbor_max_velocity < self.speed_buffer_max_threshold:
                    neighbor_low_velocity_count += 1
                    low_velocity_neighbors.append(neighbor_id)
        
        # Only trigger if we have at least one neighbor with sufficient history and low velocity
        if neighbors_with_sufficient_history > 0 and neighbor_low_velocity_count > 0:
            if self.logger:
                self.logger.log_deadlock_check(agent_id, avg_velocity, self.speed_buffer_avg_threshold, neighbor_low_velocity_count)
                self.logger.logger.debug(f"   Enhanced speed buffer triggered - Agent {agent_id}: avg_vel={avg_velocity:.3f}, max_vel={max_velocity:.3f}")
                self.logger.logger.debug(f"   Low velocity neighbors: {low_velocity_neighbors}")
                self.logger.logger.debug(f"   Neighbors with sufficient history: {neighbors_with_sufficient_history}")
            else:
                # print(f"🔍 ENHANCED SPEED BUFFER TRIGGERED: Agent {agent_id} avg_vel={avg_velocity:.3f}, max_vel={max_velocity:.3f}, {neighbor_low_velocity_count} neighbors also slow: {low_velocity_neighbors}")
                pass
            return True
        else:
            if self.logger:
                self.logger.logger.debug(f"🔍 ENHANCED SPEED BUFFER - INSUFFICIENT NEIGHBOR HISTORY: Agent {agent_id}, neighbors with history: {neighbors_with_sufficient_history}, low velocity neighbors: {neighbor_low_velocity_count}")
        
        return False
