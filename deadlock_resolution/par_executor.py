"""
PAR Executor Module

This module provides execution functionality for Push and Rotate (PAR) algorithm.
It handles the execution of PAR steps, including moving to start positions and following PAR paths.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class PARExecutor:
    """
    Executor for Push and Rotate (PAR) algorithm execution.
    
    This class handles the execution of PAR algorithm steps, including moving
    to start positions and following the computed PAR paths.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PAR executor.
        
        Args:
            config: Configuration dictionary containing execution parameters
        """
        self.config = config
        self.position_tolerance = config.get('POSITION_TOLERANCE', 0.1)
        self.velocity_scale = config.get('VELOCITY_SCALE', 1.0)
        self.max_velocity = config.get('MAX_VELOCITY', 1.5)
        
        # Execution state tracking
        self.agent_paths = {}  # Dictionary mapping agent_id to current path index
        self.agent_start_positions = {}  # Dictionary mapping agent_id to start position
        self.agent_goal_positions = {}  # Dictionary mapping agent_id to goal position
    
    def execute_par_step(self, agent_id: int, agent_states: Dict) -> Dict:
        """
        Execute a single PAR step for the given agent.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution containing paths and moves
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Action dictionary containing velocity and mode information
        """
        # Check if agent is moving to start position
        if self.is_moving_to_start(agent_id):
            return self.move_to_par_start(agent_id, agent_states)
        
        # Check if agent is following PAR path
        elif self.is_following_par_path(agent_id):
            # Get PAR solution from state manager
            par_solution = self.get_par_solution(agent_id)
            if par_solution is None:
                return {'action': np.array([0.0, 0.0]), 'mode': 'no_solution', 'target': None}
            return self.follow_par_path(agent_id, par_solution, agent_states)
        
        # Default: no action
        return {
            'action': np.array([0.0, 0.0]),
            'mode': 'idle',
            'target': None
        }
    
    def move_to_par_start(self, agent_id: int, agent_states: Dict) -> Dict:
        """
        Move agent to PAR start position using RL_RVO navigation.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Action dictionary for moving to start position
        """
        if agent_id not in agent_states:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        agent_state = agent_states[agent_id]
        current_position = self.get_agent_position(agent_state)
        start_position = self.agent_start_positions.get(agent_id)
        
        if current_position is None or start_position is None:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        # Check if already at start position
        if self.is_at_start_position(agent_id, start_position):
            return {
                'action': np.array([0.0, 0.0]),
                'mode': 'at_start',
                'target': start_position
            }
        
        # Calculate velocity to move towards start position
        velocity = self.compute_velocity_to_target(current_position, start_position)
        
        return {
            'action': velocity,
            'mode': 'move_to_start',
            'target': start_position
        }
    
    def follow_par_path(self, agent_id: int, par_solution, agent_states: Dict) -> Dict:
        """
        Follow the PAR path for the given agent.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution containing paths
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Action dictionary for following PAR path
        """
        if agent_id not in agent_states:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        agent_state = agent_states[agent_id]
        current_position = self.get_agent_position(agent_state)
        
        if current_position is None:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        # Get current path index
        path_index = self.agent_paths.get(agent_id, 0)
        
        # Get agent path from PAR solution
        agent_path = self.get_agent_path_from_solution(agent_id, par_solution)
        
        if not agent_path or path_index >= len(agent_path):
            return {
                'action': np.array([0.0, 0.0]),
                'mode': 'path_complete',
                'target': None
            }
        
        # Get next target position
        next_target = agent_path[path_index]
        
        # Check if reached current target
        if self.is_at_position(current_position, next_target):
            # Move to next path point
            self.agent_paths[agent_id] = path_index + 1
            path_index += 1
            
            if path_index >= len(agent_path):
                return {
                    'action': np.array([0.0, 0.0]),
                    'mode': 'path_complete',
                    'target': None
                }
            
            next_target = agent_path[path_index]
        
        # Calculate velocity to next target
        velocity = self.compute_velocity_to_target(current_position, next_target)
        
        return {
            'action': velocity,
            'mode': 'follow_path',
            'target': next_target
        }
    
    def is_moving_to_start(self, agent_id: int) -> bool:
        """
        Check if agent is in the moving to start phase.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is moving to start position
        """
        # This should be checked against the state manager
        # For now, we'll assume it's true if we have a start position but haven't reached it
        return agent_id in self.agent_start_positions
    
    def get_par_solution(self, agent_id: int):
        """
        Get PAR solution for the given agent from state manager.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            PAR solution or None if not available
        """
        # This should be implemented to get solution from state manager
        # For now, return None to indicate no solution
        return None
    
    def is_following_par_path(self, agent_id: int) -> bool:
        """
        Check if agent is following PAR path.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is following PAR path
        """
        # This should be checked against the state manager
        # For now, we'll assume it's true if we have a path index
        return agent_id in self.agent_paths
    
    def is_at_start_position(self, agent_id: int, start_position: Tuple[float, float]) -> bool:
        """
        Check if agent has reached its start position.
        
        Args:
            agent_id: ID of the agent
            start_position: Start position to check against
            
        Returns:
            bool: True if agent is at start position
        """
        # This should get the current position from agent_states
        # For now, we'll use a placeholder
        return False  # Placeholder
    
    def is_at_position(self, current_position: Tuple[float, float], target_position: Tuple[float, float]) -> bool:
        """
        Check if agent is at the target position.
        
        Args:
            current_position: Current agent position
            target_position: Target position to check against
            
        Returns:
            bool: True if agent is at target position
        """
        if current_position is None or target_position is None:
            return False
        
        distance = math.sqrt(
            (current_position[0] - target_position[0])**2 + 
            (current_position[1] - target_position[1])**2
        )
        
        return distance <= self.position_tolerance
    
    def compute_velocity_to_target(self, current_position: Tuple[float, float], target_position: Tuple[float, float]) -> np.ndarray:
        """
        Compute velocity to move towards target position.
        
        Args:
            current_position: Current agent position
            target_position: Target position to move towards
            
        Returns:
            np.ndarray: Velocity vector [vx, vy]
        """
        if current_position is None or target_position is None:
            return np.array([0.0, 0.0])
        
        # Calculate direction vector
        direction = np.array([
            target_position[0] - current_position[0],
            target_position[1] - current_position[1]
        ])
        
        # Calculate distance
        distance = np.linalg.norm(direction)
        
        if distance < self.position_tolerance:
            return np.array([0.0, 0.0])
        
        # Normalize direction and scale by velocity
        normalized_direction = direction / distance
        velocity = normalized_direction * self.velocity_scale
        
        # Limit maximum velocity
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.max_velocity:
            velocity = velocity * (self.max_velocity / velocity_magnitude)
        
        return velocity
    
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
    
    def get_agent_path_from_solution(self, agent_id: int, par_solution) -> List[Tuple[float, float]]:
        """
        Get agent path from PAR solution.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution object
            
        Returns:
            List[Tuple[float, float]]: List of positions in the agent's path
        """
        if par_solution is None or not hasattr(par_solution, 'agents_moves'):
            return []
        
        # Extract moves for this agent
        agent_moves = []
        for move in par_solution.agents_moves:
            if move.id == agent_id:
                agent_moves.append(move)
        
        # Convert moves to path
        path = []
        current_pos = (0, 0)  # Start from origin
        
        for move in agent_moves:
            next_pos = (current_pos[0] + move.di, current_pos[1] + move.dj)
            path.append(next_pos)
            current_pos = next_pos
        
        return path
    
    def set_agent_start_position(self, agent_id: int, start_position: Tuple[float, float]):
        """
        Set the start position for an agent.
        
        Args:
            agent_id: ID of the agent
            start_position: Start position for the agent
        """
        self.agent_start_positions[agent_id] = start_position
    
    def set_agent_goal_position(self, agent_id: int, goal_position: Tuple[float, float]):
        """
        Set the goal position for an agent.
        
        Args:
            agent_id: ID of the agent
            goal_position: Goal position for the agent
        """
        self.agent_goal_positions[agent_id] = goal_position
    
    def set_agent_path(self, agent_id: int, path: List[Tuple[float, float]]):
        """
        Set the path for an agent.
        
        Args:
            agent_id: ID of the agent
            path: List of positions in the agent's path
        """
        self.agent_paths[agent_id] = 0  # Start at the beginning of the path
    
    def is_par_complete(self, agent_id: int) -> bool:
        """
        Check if PAR execution is complete for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if PAR is complete for the agent
        """
        # Check if agent has completed its path
        if agent_id not in self.agent_paths:
            return False
        
        # This should check against the actual path length
        # For now, we'll use a placeholder
        return False  # Placeholder
    
    def reset_agent(self, agent_id: int):
        """
        Reset the execution state for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.agent_paths:
            del self.agent_paths[agent_id]
        if agent_id in self.agent_start_positions:
            del self.agent_start_positions[agent_id]
        if agent_id in self.agent_goal_positions:
            del self.agent_goal_positions[agent_id]
    
    def reset_all(self):
        """Reset execution state for all agents."""
        self.agent_paths.clear()
        self.agent_start_positions.clear()
        self.agent_goal_positions.clear()
