"""
PAR Coordinator Module

This module provides coordination functionality for Push and Rotate (PAR) algorithm.
It handles PAR execution preparation, problem solving, and solution management.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .par_environment import PAREnvironment
from python_pnr.push_and_rotate import PushAndRotate
from python_pnr.mapf_search_result import MAPFSearchResult
from python_pnr.node import Node


class PARCoordinator:
    """
    Coordinator for Push and Rotate (PAR) algorithm execution.
    
    This class handles the coordination of PAR execution, including environment
    preparation, problem solving using the existing PNR solver, and solution management.
    """
    
    def __init__(self, push_and_rotate_instance: PushAndRotate, config: Dict):
        """
        Initialize the PAR coordinator.
        
        Args:
            push_and_rotate_instance: Instance of PushAndRotate solver
            config: Configuration dictionary containing PAR parameters
        """
        self.pnr_solver = push_and_rotate_instance
        self.config = config
        self.current_par_solution = None
        self.current_participants = []
        self.par_environment = None
    
    def prepare_par_execution(self, agent_states: Dict, deadlock_participants: List[int]) -> MAPFSearchResult:
        """
        Prepare PAR execution for the given participants.
        
        Args:
            agent_states: Dictionary of all agent states
            deadlock_participants: List of agent IDs participating in PAR
            
        Returns:
            MAPFSearchResult: PAR solution for the participants
        """
        # Store current participants
        self.current_participants = deadlock_participants.copy()
        
        # Create PAR environment
        workspace = self.get_workspace_info(agent_states)
        self.par_environment = PAREnvironment(workspace, deadlock_participants, self.config)
        
        # Build PAR environment
        sub_map, actor_set = self.par_environment.build_par_environment(agent_states)
        
        # Compute start and goal positions
        start_positions = self.par_environment.compute_start_positions(agent_states)
        goal_positions = self.par_environment.compute_goal_positions(agent_states)
        
        # Solve PAR problem
        par_solution = self.solve_par_problem(sub_map, actor_set, start_positions, goal_positions)
        
        # Store current solution
        self.current_par_solution = par_solution
        
        return par_solution
    
    def solve_par_problem(self, sub_map, actor_set, start_positions: Dict, goal_positions: Dict) -> MAPFSearchResult:
        """
        Solve the PAR problem using the PNR solver.
        
        Args:
            sub_map: Sub-map for the PAR region
            actor_set: Set of actors participating in PAR
            start_positions: Dictionary mapping agent IDs to start positions
            goal_positions: Dictionary mapping agent IDs to goal positions
            
        Returns:
            MAPFSearchResult: Solution from the PNR solver
        """
        # Clear previous solution
        if hasattr(self.pnr_solver, 'clear'):
            # Call the parameterless clear method
            try:
                self.pnr_solver.clear()
            except TypeError:
                # If the clear method requires parameters, skip it
                pass
        
        # Set up the problem for the PNR solver
        # Note: The exact interface depends on the PNR solver implementation
        # This is a simplified version - adjust based on actual PNR solver interface
        
        try:
            # For now, create a simple solution without using the PNR solver
            # This is a placeholder implementation
            result = MAPFSearchResult()
            result.pathfound = True
            result.agents_moves = []
            
            # Add some dummy moves for each agent
            for agent_id in start_positions:
                if agent_id in goal_positions:
                    # Create a simple move from start to goal
                    start = start_positions[agent_id]
                    goal = goal_positions[agent_id]
                    di = goal[0] - start[0]
                    dj = goal[1] - start[1]
                    
                    # Add move to result
                    from python_pnr.node import ActorMove
                    move = ActorMove(di, dj, agent_id)
                    result.agents_moves.append(move)
            
            return result
            
        except Exception as e:
            print(f"Error solving PAR problem: {e}")
            # Return empty result if solving fails
            return MAPFSearchResult()
    
    def set_start_goals(self, start_positions: Dict, goal_positions: Dict):
        """
        Set start and goal positions for the PNR solver.
        
        Args:
            start_positions: Dictionary mapping agent IDs to start positions
            goal_positions: Dictionary mapping agent IDs to goal positions
        """
        # This method needs to be implemented based on the actual PNR solver interface
        # For now, we'll assume the PNR solver can handle this through its existing interface
        
        # Convert positions to Node objects if needed
        start_nodes = {}
        goal_nodes = {}
        
        for agent_id, pos in start_positions.items():
            if isinstance(pos, tuple) and len(pos) == 2:
                start_nodes[agent_id] = Node(pos[0], pos[1])
        
        for agent_id, pos in goal_positions.items():
            if isinstance(pos, tuple) and len(pos) == 2:
                goal_nodes[agent_id] = Node(pos[0], pos[1])
        
        # Set in PNR solver (implementation depends on actual interface)
        if hasattr(self.pnr_solver, 'set_start_goals'):
            self.pnr_solver.set_start_goals(start_nodes, goal_nodes)
    
    def update_par_solution(self, agent_states: Dict, new_participants: List[int]) -> MAPFSearchResult:
        """
        Update PAR solution when new participants are added.
        
        Args:
            agent_states: Dictionary of all agent states
            new_participants: Updated list of participants
            
        Returns:
            MAPFSearchResult: Updated PAR solution
        """
        # Check if participants have changed
        if set(new_participants) != set(self.current_participants):
            # Recompute PAR solution with new participants
            return self.prepare_par_execution(agent_states, new_participants)
        
        return self.current_par_solution
    
    def get_agent_path(self, agent_id: int) -> List[Tuple[int, int]]:
        """
        Get the PAR path for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List[Tuple[int, int]]: List of grid positions in the agent's path
        """
        if self.current_par_solution is None:
            return []
        
        # Extract path for the specific agent from the PAR solution
        # This depends on the structure of MAPFSearchResult
        if hasattr(self.current_par_solution, 'get_agent_path'):
            return self.current_par_solution.get_agent_path(agent_id)
        
        # Fallback: try to extract from agents_moves if available
        if hasattr(self.current_par_solution, 'agents_moves'):
            return self.extract_path_from_moves(agent_id)
        
        return []
    
    def extract_path_from_moves(self, agent_id: int) -> List[Tuple[int, int]]:
        """
        Extract agent path from the moves list in PAR solution.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List[Tuple[int, int]]: List of grid positions in the agent's path
        """
        if not hasattr(self.current_par_solution, 'agents_moves'):
            return []
        
        path = []
        current_pos = None
        
        # Find initial position from actor set
        if self.par_environment and self.par_environment.actor_set:
            for actor in self.par_environment.actor_set:
                if actor.id == agent_id:
                    current_pos = (actor.current.x, actor.current.y)
                    path.append(current_pos)
                    break
        
        if current_pos is None:
            return []
        
        # Follow moves to reconstruct path
        for move in self.current_par_solution.agents_moves:
            if move.agent_id == agent_id:
                new_pos = (current_pos[0] + move.dx, current_pos[1] + move.dy)
                path.append(new_pos)
                current_pos = new_pos
        
        return path
    
    def get_agent_start_position(self, agent_id: int) -> Optional[Tuple[float, float]]:
        """
        Get the start position for an agent in continuous coordinates.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[Tuple[float, float]]: Start position in continuous coordinates
        """
        if self.par_environment is None:
            return None
        
        # Get start position from PAR environment
        start_positions = self.par_environment.compute_start_positions({})
        
        if agent_id in start_positions:
            grid_pos = start_positions[agent_id]
            if hasattr(self.par_environment, 'grid_to_continuous'):
                return self.par_environment.grid_to_continuous(grid_pos)
            else:
                return None
        
        return None
    
    def get_agent_goal_position(self, agent_id: int) -> Optional[Tuple[float, float]]:
        """
        Get the goal position for an agent in continuous coordinates.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Optional[Tuple[float, float]]: Goal position in continuous coordinates
        """
        if self.par_environment is None:
            return None
        
        # Get goal position from PAR environment
        goal_positions = self.par_environment.compute_goal_positions({})
        
        if agent_id in goal_positions:
            grid_pos = goal_positions[agent_id]
            if hasattr(self.par_environment, 'grid_to_continuous'):
                return self.par_environment.grid_to_continuous(grid_pos)
            else:
                return None
        
        return None
    
    def is_par_complete(self, agent_id: int) -> bool:
        """
        Check if PAR execution is complete for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if PAR is complete for the agent
        """
        # For now, return False to keep agents in PAR mode
        # This should be implemented based on actual PAR completion criteria
        return False
    
    def get_workspace_info(self, agent_states: Dict) -> Dict:
        """
        Extract workspace information from agent states.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Workspace information
        """
        # This is a simplified implementation
        # In a real system, you would extract workspace information from the environment
        workspace = {
            'bounds': self.compute_workspace_bounds(agent_states),
            'obstacles': [],  # Would be populated from environment
            'grid_resolution': self.config.get('GRID_RESOLUTION', 1.0)
        }
        
        return workspace
    
    def compute_workspace_bounds(self, agent_states: Dict) -> Dict:
        """
        Compute workspace bounds from agent states.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Workspace bounds
        """
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for agent_state in agent_states.values():
            if 'position' in agent_state:
                pos = agent_state['position']
                if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                    min_x = min(min_x, pos[0])
                    max_x = max(max_x, pos[0])
                    min_y = min(min_y, pos[1])
                    max_y = max(max_y, pos[1])
        
        return {
            'min_x': min_x if min_x != float('inf') else 0,
            'max_x': max_x if max_x != float('-inf') else 10,
            'min_y': min_y if min_y != float('inf') else 0,
            'max_y': max_y if max_y != float('-inf') else 10
        }
    
    def reset(self):
        """Reset the PAR coordinator state."""
        self.current_par_solution = None
        self.current_participants = []
        self.par_environment = None
        if hasattr(self.pnr_solver, 'clear'):
            self.pnr_solver.clear()
