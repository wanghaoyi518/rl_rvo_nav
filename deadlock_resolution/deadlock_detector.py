"""
Deadlock Detector Module

This module provides deadlock detection functionality for RL_RVO navigation system.
It implements two trigger mechanisms: speed buffer trigger and common point trigger.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math


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
        
        # Velocity history for each agent
        self.velocity_history = defaultdict(list)
        
        # Deadlock detection state
        self.deadlock_detection_enabled = config.get('DEADLOCK_DETECTION_ENABLED', True)
        
        # Episode counter for debugging
        self.episode_counter = 0
    
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
            
        if self.trigger_type == 'SPEED_BUFFER':
            return self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
        elif self.trigger_type == 'COMMON_POINT':
            return self.check_common_point_trigger(agent_id, agent_states, neighbor_states)
        else:
            # Default to speed buffer trigger
            return self.check_speed_buffer_trigger(agent_id, agent_states, neighbor_states)
    
    def check_speed_buffer_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using speed buffer trigger mechanism.
        
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
        
        # Calculate current velocity
        current_velocity = self.calculate_current_velocity(agent_state)
        
        # Update velocity history
        self.update_velocity_history(agent_id, current_velocity)
        
        # Calculate average velocity
        avg_velocity = self.calculate_average_velocity(agent_id)
        
        # Check if agent velocity is below threshold
        if avg_velocity < self.small_speed:
            # Check neighbor velocities
            neighbor_low_velocity_count = 0
            low_velocity_neighbors = []
            for neighbor_id, neighbor_state in neighbor_states.items():
                neighbor_avg_velocity = self.calculate_average_velocity(neighbor_id)
                if neighbor_avg_velocity < self.small_speed:
                    neighbor_low_velocity_count += 1
                    low_velocity_neighbors.append(neighbor_id)
            
            # If at least one neighbor also has low velocity, consider it deadlock
            if neighbor_low_velocity_count > 0:
                print(f"🔍 DEADLOCK CHECK: Agent {agent_id} velocity={avg_velocity:.3f} < {self.small_speed}, {neighbor_low_velocity_count} neighbors also slow: {low_velocity_neighbors}")
                return True
        
        return False
    
    def check_common_point_trigger(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check deadlock using common point trigger mechanism.
        
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
        
        # Count agents near the goal (including current agent)
        agents_near_goal = 1  # Start with current agent
        
        # Check neighbors
        for neighbor_id, neighbor_state in neighbor_states.items():
            distance_to_goal = self.calculate_distance_to_goal(neighbor_state)
            if distance_to_goal < self.sight_radius:
                agents_near_goal += 1
        
        # If enough agents are near the goal, trigger deadlock detection
        if agents_near_goal >= self.mapf_num:
            print(f"🔍 COMMON POINT CHECK: Agent {agent_id} detected {agents_near_goal} agents near goal (threshold: {self.mapf_num})")
            return True
        
        return False
    
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
        # print(f"🔄 Episode {self.episode_counter}: Reset velocity history for deadlock detection")
    
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
        self.velocity_history[agent_id].append(velocity)
        
        # Keep only the last window_size entries
        if len(self.velocity_history[agent_id]) > self.velocity_window_size:
            self.velocity_history[agent_id] = self.velocity_history[agent_id][-self.velocity_window_size:]
    
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
