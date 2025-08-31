"""
Mode Controller Module

This module provides mode control functionality for RL_RVO navigation system.
It handles mode switching decisions between RL_RVO and PAR modes.
"""

from typing import Dict, List, Optional
from deadlock_resolution.deadlock_detector import DeadlockDetector
from deadlock_resolution.par_coordinator import PARCoordinator


class ModeController:
    """
    Mode controller that manages transitions between RL_RVO and PAR modes.
    
    This class is responsible for making decisions about when to switch modes
    and coordinating mode transitions across multiple agents.
    """
    
    def __init__(self, deadlock_detector: DeadlockDetector, par_coordinator: PARCoordinator, config: Dict):
        """
        Initialize the mode controller.
        
        Args:
            deadlock_detector: Instance of deadlock detector
            par_coordinator: Instance of PAR coordinator
            config: Configuration dictionary
        """
        self.deadlock_detector = deadlock_detector
        self.par_coordinator = par_coordinator
        self.config = config
        
        # Mode transition parameters
        self.par_completion_threshold = config.get('PAR_COMPLETION_THRESHOLD', 0.9)
        self.mode_switch_delay = config.get('MODE_SWITCH_DELAY', 10)  # time steps
        
        # Mode transition state tracking
        self.mode_switch_counters = {}  # Dictionary mapping agent_id to switch counter
        self.last_mode_switch_time = {}  # Dictionary mapping agent_id to last switch time
    
    def update_agent_mode(self, agent_id: int, current_mode: str, agent_states: Dict, 
                         neighbor_states: Dict, current_time: int = 0) -> str:
        """
        Update the mode for a specific agent based on current conditions.
        
        Args:
            agent_id: ID of the agent
            current_mode: Current mode of the agent ('rl_rvo' or 'par')
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            current_time: Current simulation time step
            
        Returns:
            str: New mode for the agent
        """
        # Check if enough time has passed since last mode switch
        if not self.can_switch_mode(agent_id, current_time):
            return current_mode
        
        # Determine if mode should be switched
        if current_mode == 'rl_rvo':
            if self.should_switch_to_par(agent_id, agent_states, neighbor_states):
                return 'par'
        elif current_mode == 'par':
            if self.should_switch_to_rl_rvo(agent_id, agent_states):
                return 'rl_rvo'
        
        return current_mode
    
    def should_switch_to_par(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Determine if an agent should switch from RL_RVO to PAR mode.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if agent should switch to PAR mode
        """
        # Check if deadlock is detected
        if self.deadlock_detector.detect_deadlock(agent_id, agent_states, neighbor_states):
            return True
        
        # Additional conditions for switching to PAR mode
        # For example, if multiple agents are stuck in a narrow corridor
        if self.check_narrow_corridor_condition(agent_id, agent_states, neighbor_states):
            return True
        
        return False
    
    def should_switch_to_rl_rvo(self, agent_id: int, agent_states: Dict) -> bool:
        """
        Determine if an agent should switch from PAR to RL_RVO mode.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            
        Returns:
            bool: True if agent should switch to RL_RVO mode
        """
        # Check if PAR execution is complete
        if self.par_coordinator.is_par_complete(agent_id):
            return True
        
        # Check if agent has reached its goal through PAR
        if self.has_reached_goal_through_par(agent_id, agent_states):
            return True
        
        # Check if PAR execution has failed or timed out
        if self.is_par_execution_failed(agent_id):
            return True
        
        return False
    
    def can_switch_mode(self, agent_id: int, current_time: int) -> bool:
        """
        Check if enough time has passed to allow mode switching.
        
        Args:
            agent_id: ID of the agent
            current_time: Current simulation time step
            
        Returns:
            bool: True if mode switching is allowed
        """
        if agent_id not in self.last_mode_switch_time:
            return True
        
        time_since_last_switch = current_time - self.last_mode_switch_time[agent_id]
        return time_since_last_switch >= self.mode_switch_delay
    
    def check_narrow_corridor_condition(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check if agents are in a narrow corridor situation that might benefit from PAR.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if narrow corridor condition is detected
        """
        # Count nearby agents
        nearby_agent_count = len(neighbor_states)
        
        # If there are many nearby agents, it might indicate a bottleneck
        if nearby_agent_count >= self.config.get('NARROW_CORRIDOR_THRESHOLD', 4):
            return True
        
        # Check if agents are moving slowly in a confined area
        if self.check_confined_area_movement(agent_id, agent_states, neighbor_states):
            return True
        
        return False
    
    def check_confined_area_movement(self, agent_id: int, agent_states: Dict, neighbor_states: Dict) -> bool:
        """
        Check if agents are moving slowly in a confined area.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            neighbor_states: Dictionary of neighbor states for the agent
            
        Returns:
            bool: True if confined area movement is detected
        """
        # Calculate average velocity of nearby agents
        total_velocity = 0.0
        agent_count = 0
        
        # Include current agent
        if agent_id in agent_states:
            current_velocity = self.calculate_agent_velocity(agent_states[agent_id])
            total_velocity += current_velocity
            agent_count += 1
        
        # Include neighbors
        for neighbor_id, neighbor_state in neighbor_states.items():
            neighbor_velocity = self.calculate_agent_velocity(neighbor_state)
            total_velocity += neighbor_velocity
            agent_count += 1
        
        if agent_count > 0:
            avg_velocity = total_velocity / agent_count
            # If average velocity is very low, consider it confined area movement
            return avg_velocity < self.config.get('CONFINED_AREA_VELOCITY_THRESHOLD', 0.05)
        
        return False
    
    def calculate_agent_velocity(self, agent_state: Dict) -> float:
        """
        Calculate the velocity magnitude of an agent.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            float: Velocity magnitude
        """
        if 'velocity' in agent_state:
            velocity = agent_state['velocity']
            if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                return (velocity[0]**2 + velocity[1]**2)**0.5
        
        return 0.0
    
    def has_reached_goal_through_par(self, agent_id: int, agent_states: Dict) -> bool:
        """
        Check if agent has reached its goal through PAR execution.
        
        Args:
            agent_id: ID of the agent
            agent_states: Dictionary of all agent states
            
        Returns:
            bool: True if agent has reached goal through PAR
        """
        if agent_id not in agent_states:
            return False
        
        agent_state = agent_states[agent_id]
        
        # Check if agent has a goal
        if 'goal' not in agent_state:
            return False
        
        # Get current position and goal
        current_pos = self.get_agent_position(agent_state)
        goal_pos = agent_state['goal']
        
        if current_pos is None or goal_pos is None:
            return False
        
        # Calculate distance to goal
        distance = ((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2)**0.5
        
        # Check if close enough to goal
        goal_tolerance = self.config.get('GOAL_TOLERANCE', 0.5)
        return distance <= goal_tolerance
    
    def is_par_execution_failed(self, agent_id: int) -> bool:
        """
        Check if PAR execution has failed for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if PAR execution has failed
        """
        # This could check for various failure conditions:
        # - PAR solution is empty or invalid
        # - Agent is stuck and not making progress
        # - PAR execution has exceeded time limit
        
        # For now, we'll use a simple check
        # In a real implementation, you might track PAR execution time and progress
        
        return False
    
    def get_agent_position(self, agent_state: Dict) -> Optional[tuple]:
        """
        Extract agent position from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Optional[tuple]: Agent position (x, y) or None if not available
        """
        if 'position' in agent_state:
            position = agent_state['position']
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                return (float(position[0]), float(position[1]))
        
        return None
    
    def coordinate_multi_agent_transition(self, agent_ids: List[int], target_mode: str, 
                                        agent_states: Dict, current_time: int) -> Dict[int, str]:
        """
        Coordinate mode transitions for multiple agents.
        
        Args:
            agent_ids: List of agent IDs to coordinate
            target_mode: Target mode for transition
            agent_states: Dictionary of all agent states
            current_time: Current simulation time step
            
        Returns:
            Dict[int, str]: Dictionary mapping agent IDs to their new modes
        """
        transitions = {}
        
        for agent_id in agent_ids:
            # Check if agent can switch mode
            if self.can_switch_mode(agent_id, current_time):
                transitions[agent_id] = target_mode
                # Update switch time
                self.last_mode_switch_time[agent_id] = current_time
            else:
                # Keep current mode
                transitions[agent_id] = 'rl_rvo'  # Default mode
        
        return transitions
    
    def get_mode_transition_action(self, agent_id: int, current_mode: str, target_mode: str) -> Dict:
        """
        Get the action required for mode transition.
        
        Args:
            agent_id: ID of the agent
            current_mode: Current mode
            target_mode: Target mode
            
        Returns:
            Dict: Action dictionary for mode transition
        """
        if current_mode == target_mode:
            return {'action': 'none', 'mode': current_mode}
        
        if current_mode == 'rl_rvo' and target_mode == 'par':
            return {
                'action': 'switch_to_par',
                'mode': 'par',
                'description': 'Switching to PAR mode for deadlock resolution'
            }
        
        elif current_mode == 'par' and target_mode == 'rl_rvo':
            return {
                'action': 'switch_to_rl_rvo',
                'mode': 'rl_rvo',
                'description': 'Switching back to RL_RVO mode'
            }
        
        return {'action': 'none', 'mode': current_mode}
    
    def reset_agent_mode_state(self, agent_id: int):
        """
        Reset mode state for an agent.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.mode_switch_counters:
            del self.mode_switch_counters[agent_id]
        if agent_id in self.last_mode_switch_time:
            del self.last_mode_switch_time[agent_id]
    
    def reset_all_mode_states(self):
        """Reset mode state for all agents."""
        self.mode_switch_counters.clear()
        self.last_mode_switch_time.clear()
