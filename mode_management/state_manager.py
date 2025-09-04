"""
State Manager Module

This module provides state management functionality for RL_RVO navigation system.
It handles state storage, updates, and queries for agent modes and PAR status.
"""

from typing import Dict, List, Optional, Any
from deadlock_resolution.par_coordinator import PARCoordinator


class StateManager:
    """
    State manager that handles agent state storage and updates.
    
    This class is responsible for storing and managing agent states,
    including mode information and PAR-related status.
    """
    
    def __init__(self):
        """
        Initialize the state manager.
        """
        # Agent states storage
        self.agent_states = {}  # Dictionary mapping agent_id to state dictionary
        
        # Default state template
        self.default_state = {
            'mode': 'rl_rvo',
            'par_status': {
                'in_par_mode': False,
                'move_to_par_pos': False,
                'par_exec': False,
                'wait_for_finish': False
            },
            'par_solution': None,
            'par_start_position': None,
            'par_goal_position': None,
            'par_path': None,
            'par_path_index': 0,
            'mode_switch_time': 0,
            'last_position': None,
            'last_velocity': None
        }
    
    def set_par_mode(self, agent_id: int, par_solution=None, start_position=None, goal_position=None):
        """
        Set an agent to PAR mode.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution object
            start_position: Start position for PAR execution
            goal_position: Goal position for PAR execution
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id].update({
            'mode': 'par',
            'par_status': {
                'in_par_mode': True,
                'move_to_par_pos': True,  # Start by moving to PAR position
                'par_exec': False,
                'wait_for_finish': False
            },
            'par_solution': par_solution,
            'par_start_position': start_position,
            'par_goal_position': goal_position,
            'par_path': None,
            'par_path_index': 0
        })
    
    def set_rl_rvo_mode(self, agent_id: int):
        """
        Set an agent to RL_RVO mode.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id].update({
            'mode': 'rl_rvo',
            'par_status': {
                'in_par_mode': False,
                'move_to_par_pos': False,
                'par_exec': False,
                'wait_for_finish': False
            },
            'par_solution': None,
            'par_start_position': None,
            'par_goal_position': None,
            'par_path': None,
            'par_path_index': 0
        })
    
    def get_agent_state(self, agent_id: int) -> Dict:
        """
        Get the complete state for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dict: Complete agent state dictionary
        """
        if agent_id not in self.agent_states:
            # Return default state if agent not found
            return self.default_state.copy()
        
        return self.agent_states[agent_id].copy()
    
    def get_agent_mode(self, agent_id: int) -> str:
        """
        Get the current mode of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            str: Current mode ('rl_rvo' or 'par')
        """
        if agent_id not in self.agent_states:
            return 'rl_rvo'  # Default mode
        
        return self.agent_states[agent_id]['mode']
    
    def get_par_status(self, agent_id: int) -> Dict:
        """
        Get the PAR status for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dict: PAR status dictionary
        """
        if agent_id not in self.agent_states:
            return self.default_state['par_status'].copy()
        
        return self.agent_states[agent_id]['par_status'].copy()
    
    def update_par_status(self, agent_id: int, status_updates: Dict):
        """
        Update PAR status for an agent.
        
        Args:
            agent_id: ID of the agent
            status_updates: Dictionary containing status updates
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        # Update PAR status
        self.agent_states[agent_id]['par_status'].update(status_updates)
    
    def set_par_executing(self, agent_id: int):
        """
        Set agent to PAR executing state.
        
        Args:
            agent_id: ID of the agent
        """
        self.update_par_status(agent_id, {
            'move_to_par_pos': False,
            'par_exec': True
        })
    
    def set_par_waiting(self, agent_id: int):
        """
        Set agent to PAR waiting state.
        
        Args:
            agent_id: ID of the agent
        """
        self.update_par_status(agent_id, {
            'par_exec': False,
            'wait_for_finish': True
        })
    
    def is_moving_to_start(self, agent_id: int) -> bool:
        """
        Check if agent is moving to PAR start position.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is moving to start position
        """
        par_status = self.get_par_status(agent_id)
        return par_status['move_to_par_pos']
    
    def is_par_executing(self, agent_id: int) -> bool:
        """
        Check if agent is executing PAR.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is executing PAR
        """
        par_status = self.get_par_status(agent_id)
        return par_status['par_exec']
    
    def is_par_waiting(self, agent_id: int) -> bool:
        """
        Check if agent is waiting for PAR completion.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is waiting for PAR completion
        """
        par_status = self.get_par_status(agent_id)
        return par_status['wait_for_finish']
    
    def is_in_par_mode(self, agent_id: int) -> bool:
        """
        Check if agent is in PAR mode.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is in PAR mode
        """
        par_status = self.get_par_status(agent_id)
        return par_status['in_par_mode']
    
    def get_par_start_position(self, agent_id: int) -> Optional[tuple]:
        """
        Get the PAR start position for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[tuple]: Start position (x, y) or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['par_start_position']
    
    def get_par_goal_position(self, agent_id: int) -> Optional[tuple]:
        """
        Get the PAR goal position for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[tuple]: Goal position (x, y) or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['par_goal_position']
    
    def get_par_solution(self, agent_id: int) -> Any:
        """
        Get the PAR solution for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Any: PAR solution object or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['par_solution']
    
    def set_par_path(self, agent_id: int, path: List[tuple]):
        """
        Set the PAR path for an agent.
        
        Args:
            agent_id: ID of the agent
            path: List of positions in the PAR path
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id]['par_path'] = path
        self.agent_states[agent_id]['par_path_index'] = 0
    
    def get_par_path(self, agent_id: int) -> Optional[List[tuple]]:
        """
        Get the PAR path for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[List[tuple]]: PAR path or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['par_path']
    
    def get_par_path_index(self, agent_id: int) -> int:
        """
        Get the current PAR path index for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            int: Current path index
        """
        if agent_id not in self.agent_states:
            return 0
        
        return self.agent_states[agent_id]['par_path_index']
    
    def increment_par_path_index(self, agent_id: int):
        """
        Increment the PAR path index for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.agent_states:
            self.agent_states[agent_id]['par_path_index'] += 1
    
    def update_agent_position(self, agent_id: int, position: tuple):
        """
        Update the last known position of an agent.
        
        Args:
            agent_id: ID of the agent
            position: Current position (x, y)
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id]['last_position'] = position
    
    def update_agent_velocity(self, agent_id: int, velocity: tuple):
        """
        Update the last known velocity of an agent.
        
        Args:
            agent_id: ID of the agent
            velocity: Current velocity (vx, vy)
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id]['last_velocity'] = velocity
    
    def get_last_position(self, agent_id: int) -> Optional[tuple]:
        """
        Get the last known position of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[tuple]: Last position (x, y) or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['last_position']
    
    def get_last_velocity(self, agent_id: int) -> Optional[tuple]:
        """
        Get the last known velocity of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[tuple]: Last velocity (vx, vy) or None if not set
        """
        if agent_id not in self.agent_states:
            return None
        
        return self.agent_states[agent_id]['last_velocity']
    
    def set_mode_switch_time(self, agent_id: int, switch_time: int):
        """
        Set the mode switch time for an agent.
        
        Args:
            agent_id: ID of the agent
            switch_time: Time when mode was switched
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = self.default_state.copy()
        
        self.agent_states[agent_id]['mode_switch_time'] = switch_time
    
    def get_mode_switch_time(self, agent_id: int) -> int:
        """
        Get the mode switch time for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            int: Mode switch time
        """
        if agent_id not in self.agent_states:
            return 0
        
        return self.agent_states[agent_id]['mode_switch_time']
    
    def reset_agent_states(self, agent_ids: List[int]):
        """
        Reset states for specific agents.
        
        Args:
            agent_ids: List of agent IDs to reset
        """
        for agent_id in agent_ids:
            if agent_id in self.agent_states:
                del self.agent_states[agent_id]
    
    def reset_all_states(self):
        """Reset states for all agents."""
        self.agent_states.clear()
    
    def get_all_agent_ids(self) -> List[int]:
        """
        Get list of all agent IDs with states.
        
        Returns:
            List[int]: List of agent IDs
        """
        return list(self.agent_states.keys())
    
    def get_agents_in_mode(self, mode: str) -> List[int]:
        """
        Get list of agent IDs that are in a specific mode.
        
        Args:
            mode: Mode to filter by ('rl_rvo' or 'par')
            
        Returns:
            List[int]: List of agent IDs in the specified mode
        """
        agent_ids = []
        for agent_id, state in self.agent_states.items():
            if state['mode'] == mode:
                agent_ids.append(agent_id)
        
        return agent_ids
    
    def get_agents_in_par_mode(self) -> List[int]:
        """
        Get list of agent IDs that are in PAR mode.
        
        Returns:
            List[int]: List of agent IDs in PAR mode
        """
        return self.get_agents_in_mode('par')
    
    def get_agents_in_rl_rvo_mode(self) -> List[int]:
        """
        Get list of agent IDs that are in RL_RVO mode.
        
        Returns:
            List[int]: List of agent IDs in RL_RVO mode
        """
        return self.get_agents_in_mode('rl_rvo')
    
    def get_state_summary(self) -> Dict:
        """
        Get a summary of all agent states.
        
        Returns:
            Dict: Summary of agent states
        """
        summary = {
            'total_agents': len(self.agent_states),
            'rl_rvo_agents': len(self.get_agents_in_rl_rvo_mode()),
            'par_agents': len(self.get_agents_in_par_mode()),
            'agent_modes': {}
        }
        
        for agent_id in self.agent_states:
            summary['agent_modes'][agent_id] = self.get_agent_mode(agent_id)
        
        return summary
    
    def reset_episode(self):
        """
        Reset all agent states for a new episode.
        """
        self.agent_states.clear()
        print("🔄 StateManager: Reset all agent states for new episode")