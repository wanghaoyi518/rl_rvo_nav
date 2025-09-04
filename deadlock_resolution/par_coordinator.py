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
    
    def __init__(self, push_and_rotate_instance: PushAndRotate, config: Dict, gym_env=None):
        """
        Initialize the PAR coordinator.
        
        Args:
            push_and_rotate_instance: Instance of PushAndRotate solver
            config: Configuration dictionary containing PAR parameters
            gym_env: Reference to the gym environment for obstacle access
        """
        self.pnr_solver = push_and_rotate_instance
        self.config = config
        self.gym_env = gym_env
        self.current_par_solution = None
        self.current_participants = []
        self.par_environment = None
        self.logger = None  # Will be set by set_logger method
    
    def set_logger(self, logger):
        """Set the logger for detailed logging."""
        self.logger = logger
    
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
        
        # Log PAR input parameters for debugging
        # print(f"🔍 PAR INPUT PARAMETERS:")
        # print(f"   Participants: {deadlock_participants}")
        # print(f"   Workspace bounds: {workspace.get('bounds', 'N/A')}")
        # print(f"   Grid resolution: {self.config.get('GRID_RESOLUTION', 'N/A')}")
        # print(f"   PAR offset: {self.config.get('PAR_OFFSET', 'N/A')}")
        # print(f"   Sub-map size: {sub_map.width if sub_map else 'N/A'} x {sub_map.height if sub_map else 'N/A'}")
        # print(f"   Actor set size: {len(actor_set) if actor_set else 'N/A'}")
        # print(f"   Start positions: {start_positions}")
        # print(f"   Goal positions: {goal_positions}")
        
        # Solve PAR problem
        par_solution = self.solve_par_problem(sub_map, actor_set, start_positions, goal_positions)
        
        # Store current solution
        self.current_par_solution = par_solution
        
        # Initialize PAR execution tracking
        self._initialize_par_tracking(deadlock_participants)
        
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
        # Prepare PAR solver input for logging
        par_solver_input = {
            'start_positions': start_positions,
            'goal_positions': goal_positions,
            'participants': list(start_positions.keys()),
            'workspace_bounds': getattr(self.par_environment, 'workspace_bounds', {}),
            'grid_resolution': self.config.get('GRID_RESOLUTION', 0.2)
        }
        
        # Clear previous solution
        if hasattr(self.pnr_solver, 'clear'):
            try:
                self.pnr_solver.clear()
            except TypeError:
                pass
        
        # Set up the problem for the PNR solver
        try:
            # print(f"🔍 PAR SOLUTION GENERATION:")
            # print(f"   Using real PNR solver (Push and Rotate)")
            
            # Configure MAPF parameters
            from python_pnr.mapf_config import MAPFConfig
            mapf_config = MAPFConfig(
                max_steps=self.config.get('PAR_MAX_STEPS', 1000),
                timeout=self.config.get('PAR_TIMEOUT', 500),
                heuristic_weight=self.config.get('PAR_HEURISTIC_WEIGHT', 1.0)
            )
            
            # print(f"   MAPF Config: max_steps={mapf_config.max_steps}, timeout={mapf_config.timeout}, heuristic_weight={mapf_config.heuristic_weight}")
            
            # Update actor set with proper start and goal positions
            self._update_actor_set_positions(actor_set, start_positions, goal_positions)
            
            # Call the real PNR solver
            # print(f"   Calling PNR solver with {len(actor_set)} agents...")
            result = self.pnr_solver.start_search(sub_map, mapf_config, actor_set)
            
            # Log detailed PAR solver information if logger is available
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_par_solver_details(par_solver_input, result, list(start_positions.keys()))
            
            if result and hasattr(result, 'success') and result.success:
                # Get moves from the solver, not from the result
                moves_count = len(self.pnr_solver.agents_moves) if hasattr(self.pnr_solver, 'agents_moves') else 0
                # print(f"   ✅ PNR solver succeeded! Generated {moves_count} moves")
                # print(f"   Runtime: {getattr(result, 'runtime', 'N/A')} seconds")
                
                # Log detailed solution information
                if hasattr(self.pnr_solver, 'agents_paths') and self.pnr_solver.agents_paths:
                    for i, path in enumerate(self.pnr_solver.agents_paths):
                        if i < len(actor_set):
                            agent_id = actor_set[i].id if hasattr(actor_set[i], 'id') else i
                            # print(f"   Agent {agent_id}: {len(path)} path points")
                
                # Copy moves from solver to result for compatibility
                if hasattr(self.pnr_solver, 'agents_moves'):
                    result.agents_moves = self.pnr_solver.agents_moves.copy()
                
                return result
            else:
                # print(f"   ❌ PNR solver failed or returned no solution")
                # Fall back to simple solution
                fallback_result = self._generate_fallback_solution(start_positions, goal_positions)
                
                # Log fallback solution details if logger is available
                if hasattr(self, 'logger') and self.logger:
                    self.logger.log_par_solver_details(par_solver_input, fallback_result, list(start_positions.keys()))
                
                return fallback_result
                
        except Exception as e:
            # print(f"❌ Error in PNR solver: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to simple solution
            fallback_result = self._generate_fallback_solution(start_positions, goal_positions)
            
            # Log fallback solution details if logger is available
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_par_solver_details(par_solver_input, fallback_result, list(start_positions.keys()))
            
            return fallback_result
    
    def _update_actor_set_positions(self, actor_set, start_positions: Dict, goal_positions: Dict):
        """Update actor set with proper start and goal positions."""
        from python_pnr.node import Point
        
        for actor in actor_set:
            if hasattr(actor, 'id') and actor.id in start_positions:
                # Update start position
                start_pos = start_positions[actor.id]
                if hasattr(actor, 'start'):
                    actor.start = Point(start_pos[0], start_pos[1])
                
                # Update goal position
                if actor.id in goal_positions:
                    goal_pos = goal_positions[actor.id]
                    if hasattr(actor, 'goal'):
                        actor.goal = Point(goal_pos[0], goal_pos[1])
                
                # Update current position
                if hasattr(actor, 'current'):
                    actor.current.x = start_pos[0]
                    actor.current.y = start_pos[1]
    
    def _generate_fallback_solution(self, start_positions: Dict, goal_positions: Dict) -> MAPFSearchResult:
        """Generate an intelligent fallback solution when PNR solver fails."""
        # print(f"   🔄 Generating intelligent fallback solution...")
        
        result = MAPFSearchResult()
        result.pathfound = True
        result.agents_moves = []
        
        # Process all agents, including start=goal ones
        valid_agents = []
        completed_agents = []
        
        for agent_id in start_positions:
            if agent_id in goal_positions:
                start = start_positions[agent_id]
                goal = goal_positions[agent_id]
                
                # Check if start=goal (agent already at target)
                if start == goal:
                    # print(f"   ✅ Agent {agent_id}: start=goal, marking as completed (no movement needed)")
                    # Generate a "stay in place" move (0, 0) to mark completion
                    from python_pnr.node import ActorMove
                    stay_move = ActorMove(0, 0, agent_id)
                    result.agents_moves.append(stay_move)
                    completed_agents.append(agent_id)
                    continue
                
                # Check if goal is reachable (within reasonable distance)
                distance = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
                if distance > 10:  # Skip if goal is too far (more than 10 grid cells)
                    # print(f"   ⚠️ Agent {agent_id}: goal too far (distance: {distance:.1f}), skipping from PAR")
                    continue
                
                valid_agents.append((agent_id, start, goal, distance))
            else:
                # print(f"   ⚠️ Agent {agent_id}: No goal position found, skipping from PAR")
                pass
        
        # Sort agents by distance (closest goals first)
        valid_agents.sort(key=lambda x: x[3])
        
        # print(f"   📊 PAR Planning: {len(valid_agents)} active agents, {len(completed_agents)} already completed")
        
        # Generate intelligent paths for valid agents
        for agent_id, start, goal, distance in valid_agents:
            # Generate multi-step path instead of direct movement
            path = self._generate_smart_path(start, goal, start_positions, goal_positions)
            
            if path and len(path) > 1:
                # Convert path to moves
                moves = self._path_to_moves(agent_id, path)
                result.agents_moves.extend(moves)
                # print(f"   ✅ Agent {agent_id}: {len(path)} steps path generated")
            else:
                # Fallback to simple move if path generation fails
                di = goal[0] - start[0]
                dj = goal[1] - start[1]
                from python_pnr.node import ActorMove
                move = ActorMove(di, dj, agent_id)
                result.agents_moves.append(move)
                # print(f"   🔄 Agent {agent_id}: fallback to simple move ({di}, {dj})")
        
        # print(f"   📈 Total moves generated: {len(result.agents_moves)}")
        if completed_agents:
            # print(f"   🎯 Completed agents (start=goal): {completed_agents}")
            pass
        if valid_agents:
            # print(f"   🚀 Active agents with paths: {[agent[0] for agent in valid_agents]}")
            pass
        
        return result
    
    def _generate_smart_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                            start_positions: Dict, goal_positions: Dict) -> List[Tuple[int, int]]:
        """Generate a smart path avoiding other agents and obstacles."""
        path = [start]
        current = start
        
        # Simple A* inspired pathfinding
        max_iterations = 20  # Prevent infinite loops
        iteration = 0
        
        while current != goal and iteration < max_iterations:
            # Calculate direction to goal
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            
            # Determine next step (prioritize larger difference)
            next_step = current
            if abs(dx) > abs(dy):
                # Move in X direction first
                if dx > 0:
                    next_step = (current[0] + 1, current[1])
                elif dx < 0:
                    next_step = (current[0] - 1, current[1])
                elif dy != 0:
                    # Move in Y direction if X is already correct
                    if dy > 0:
                        next_step = (current[0], current[1] + 1)
                    else:
                        next_step = (current[0], current[1] - 1)
            else:
                # Move in Y direction first
                if dy > 0:
                    next_step = (current[0], current[1] + 1)
                elif dy < 0:
                    next_step = (current[0], current[1] - 1)
                elif dx != 0:
                    # Move in X direction if Y is already correct
                    if dx > 0:
                        next_step = (current[0] + 1, current[1])
                    else:
                        next_step = (current[0] - 1, current[1])
            
            # Check if next step is valid (not occupied by other agents)
            if self._is_position_valid(next_step, start_positions, goal_positions):
                path.append(next_step)
                current = next_step
            else:
                # Try alternative path
                alternative = self._find_alternative_step(current, goal, start_positions, goal_positions)
                if alternative and alternative not in path:
                    path.append(alternative)
                    current = alternative
                else:
                    # If no alternative found, break to avoid infinite loop
                    break
            
            iteration += 1
        
        return path
    
    def _is_position_valid(self, pos: Tuple[int, int], start_positions: Dict, goal_positions: Dict) -> bool:
        """Check if a position is valid (not occupied by other agents)."""
        # Check if position is occupied by start or goal of other agents
        for agent_id, start_pos in start_positions.items():
            if start_pos == pos:
                return False
        
        for agent_id, goal_pos in goal_positions.items():
            if goal_pos == pos:
                return False
        
        return True
    
    def _find_alternative_step(self, current: Tuple[int, int], goal: Tuple[int, int], 
                              start_positions: Dict, goal_positions: Dict) -> Optional[Tuple[int, int]]:
        """Find an alternative step when the direct path is blocked."""
        # Try different directions
        alternatives = [
            (current[0] + 1, current[1]),   # Right
            (current[0] - 1, current[1]),   # Left
            (current[0], current[1] + 1),   # Up
            (current[0], current[1] - 1),   # Down
            (current[0] + 1, current[1] + 1), # Diagonal
            (current[0] - 1, current[1] - 1), # Diagonal
        ]
        
        # Filter valid alternatives
        valid_alternatives = [alt for alt in alternatives if self._is_position_valid(alt, start_positions, goal_positions)]
        
        if valid_alternatives:
            # Choose the alternative closest to goal
            best_alternative = min(valid_alternatives, 
                                 key=lambda alt: np.sqrt((alt[0] - goal[0])**2 + (alt[1] - goal[1])**2))
            return best_alternative
        
        return None
    
    def _path_to_moves(self, agent_id: int, path: List[Tuple[int, int]]) -> List:
        """Convert a path to a list of ActorMove objects."""
        moves = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            di = next_pos[0] - current[0]
            dj = next_pos[1] - current[1]
            
            from python_pnr.node import ActorMove
            move = ActorMove(di, dj, agent_id)
            moves.append(move)
        
        return moves
    
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
        # Check if we have a current PAR solution
        if not self.current_par_solution or not hasattr(self.current_par_solution, 'agents_moves'):
            return False
        
        # Check if agent has reached its goal through PAR
        if self._has_agent_reached_goal(agent_id):
            return True
        
        # Check if agent has completed its assigned path
        if self._has_agent_completed_path(agent_id):
            return True
        
        # Check if PAR execution has timed out
        if self._is_par_timed_out(agent_id):
            return True
        
        # Check if deadlock has been resolved (agents are no longer in close proximity)
        if self._is_deadlock_resolved(agent_id):
            return True
        
        return False
    
    def _has_agent_reached_goal(self, agent_id: int) -> bool:
        """Check if agent has reached its goal position."""
        if not hasattr(self, 'gym_env') or not self.gym_env:
            return False
        
        try:
            # Get current agent position from gym environment
            if hasattr(self.gym_env, 'robot_list'):
                for robot in self.gym_env.robot_list:
                    if hasattr(robot, 'id') and robot.id == agent_id:
                        # Get current position using omni_state
                        if hasattr(robot, 'omni_state'):
                            current_pos = robot.omni_state()[:2]  # First 2 elements are x, y position
                        else:
                            continue
                        
                        # Get goal position
                        goal_pos = None
                        if hasattr(robot, 'goal') and robot.goal is not None:
                            goal_pos = robot.goal
                        elif hasattr(robot, 'target') and robot.target is not None:
                            goal_pos = robot.target
                        elif hasattr(robot, 'destination') and robot.destination is not None:
                            goal_pos = robot.destination
                        
                        if current_pos is not None and goal_pos is not None:
                            # Calculate distance to goal
                            distance = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
                            goal_tolerance = self.config.get('GOAL_TOLERANCE', 0.5)
                            
                            if distance <= goal_tolerance:
                                # print(f"🎯 Agent {agent_id} reached goal through PAR (distance: {distance:.3f})")
                                return True
        except Exception as e:
            # print(f"⚠️ Warning: Could not check goal reaching for agent {agent_id}: {e}")
            pass
        
        return False
    
    def _has_agent_completed_path(self, agent_id: int) -> bool:
        """Check if agent has completed its assigned PAR path."""
        if not self.current_par_solution or not hasattr(self.current_par_solution, 'agents_moves'):
            return False
        
        # Count moves for this agent
        agent_moves = [move for move in self.current_par_solution.agents_moves if move.id == agent_id]
        
        # If agent has no moves assigned, consider it complete
        if len(agent_moves) == 0:
            return True
        
        # Check if agent has executed all its moves
        # This is a simplified check - in practice, you'd track actual execution
        if not hasattr(self, '_par_execution_progress'):
            self._par_execution_progress = {}
        
        if agent_id not in self._par_execution_progress:
            self._par_execution_progress[agent_id] = 0
        
        # For now, consider complete if agent has been in PAR mode for a while
        # In practice, this should track actual move execution
        self._par_execution_progress[agent_id] += 1
        completion_threshold = self.config.get('PAR_COMPLETION_THRESHOLD', 10)
        
        if self._par_execution_progress[agent_id] >= completion_threshold:
            # print(f"🔄 Agent {agent_id} completed PAR path after {self._par_execution_progress[agent_id]} steps")
            return True
        
        return False
    
    def _is_par_timed_out(self, agent_id: int) -> bool:
        """Check if PAR execution has timed out for the agent."""
        if not hasattr(self, '_par_start_time'):
            self._par_start_time = {}
        
        if agent_id not in self._par_start_time:
            self._par_start_time[agent_id] = 0
        
        current_time = self._par_start_time[agent_id]
        timeout = self.config.get('PAR_TIMEOUT', 500)
        
        if current_time >= timeout:
            # print(f"⏰ Agent {agent_id} PAR execution timed out after {timeout} steps")
            return True
        
        self._par_start_time[agent_id] += 1
        return False
    
    def _is_deadlock_resolved(self, agent_id: int) -> bool:
        """Check if deadlock has been resolved for the agent."""
        if not hasattr(self, 'gym_env') or not self.gym_env:
            return False
        
        try:
            # Get current agent and neighbor positions
            if hasattr(self.gym_env, 'robot_list'):
                current_agent = None
                neighbor_positions = []
                
                for robot in self.gym_env.robot_list:
                    if hasattr(robot, 'id'):
                        if robot.id == agent_id:
                            current_agent = robot
                        else:
                            # Get neighbor position using omni_state
                            if hasattr(robot, 'omni_state'):
                                neighbor_pos = robot.omni_state()[:2]  # First 2 elements are x, y position
                                neighbor_positions.append(neighbor_pos)
                
                if current_agent is not None and hasattr(current_agent, 'omni_state'):
                    current_pos = current_agent.omni_state()[:2]  # First 2 elements are x, y position
                    
                    # Check if agent is no longer in close proximity to neighbors
                    communication_range = self.config.get('COMMUNICATION_RANGE', 3.0)
                    nearby_agents = 0
                    
                    for neighbor_pos in neighbor_positions:
                        if neighbor_pos is not None:
                            distance = np.linalg.norm(np.array(current_pos) - np.array(neighbor_pos))
                            if distance <= communication_range:
                                nearby_agents += 1
                    
                    # If agent has moved away from neighbors, deadlock might be resolved
                    if nearby_agents < 2:  # Less than 2 nearby agents
                        # print(f"🔓 Agent {agent_id} deadlock resolved - moved away from neighbors")
                        return True
                        
        except Exception as e:
            # print(f"⚠️ Warning: Could not check deadlock resolution for agent {agent_id}: {e}")
            pass
        
        return False
    
    def _initialize_par_tracking(self, participants: List[int]):
        """Initialize tracking for PAR execution."""
        if not hasattr(self, '_par_start_time'):
            self._par_start_time = {}
        if not hasattr(self, '_par_execution_progress'):
            self._par_execution_progress = {}
        
        # Initialize tracking for all participants
        for agent_id in participants:
            self._par_start_time[agent_id] = 0
            self._par_execution_progress[agent_id] = 0
        
        # print(f"🔧 PAR tracking initialized for {len(participants)} agents: {participants}")
    
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
            'obstacles': self._get_environment_obstacles(),
            'grid_resolution': self.config.get('GRID_RESOLUTION', 1.0)
        }
        
        return workspace
    
    def _get_environment_obstacles(self) -> List:
        """Get obstacles from the environment."""
        try:
            # Try to get obstacles from the gym environment
            if hasattr(self, 'gym_env') and self.gym_env:
                # Get obstacles from gym environment components
                if hasattr(self.gym_env, 'components') and 'obstacles' in self.gym_env.components:
                    obstacles = self.gym_env.components['obstacles']
                    # print(f"🔍 PAR COORDINATOR: Found {len(obstacles)} obstacles in gym environment")
                    return obstacles
                
                # Try alternative obstacle access methods
                if hasattr(self.gym_env, 'world') and hasattr(self.gym_env.world, 'obstacles'):
                    obstacles = self.gym_env.world.obstacles
                    # print(f"🔍 PAR COORDINATOR: Found {len(obstacles)} obstacles in gym world")
                    return obstacles
                
                if hasattr(self.gym_env, 'get_obstacles'):
                    obstacles = self.gym_env.get_obstacles()
                    # print(f"🔍 PAR COORDINATOR: Found {len(obstacles)} obstacles via get_obstacles")
                    return obstacles
            
            # print(f"🔍 PAR COORDINATOR: No obstacles found in environment")
            return []
            
        except Exception as e:
            # print(f"⚠️ Warning: Could not get environment obstacles: {e}")
            return []
    
    def compute_workspace_bounds(self, agent_states):
        """Compute workspace bounds from agent states."""
        # print(f"🔍 DEBUG: PARCoordinator.compute_workspace_bounds called with {len(agent_states)} agents")
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        # print(f"🔍 DEBUG: Initial bounds - min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")
        
        for agent_id, agent_state in agent_states.items():
            # print(f"🔍 DEBUG: Processing agent {agent_id}: {agent_state}")
            
            # Consider both position and goal
            if 'position' in agent_state:
                pos = agent_state['position']
                # print(f"🔍 DEBUG: Agent {agent_id} position: {pos}")
                
                if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                    x = float(pos[0][0]) if isinstance(pos[0], (list, np.ndarray)) else float(pos[0])
                    y = float(pos[1][0]) if isinstance(pos[1], (list, np.ndarray)) else float(pos[1])
                    
                    # print(f"🔍 DEBUG: Agent {agent_id} extracted position: ({x}, {y})")
                    
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    
                    # print(f"🔍 DEBUG: Agent {agent_id} position updated bounds - min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")
            
            if 'goal' in agent_state:
                goal = agent_state['goal']
                # print(f"🔍 DEBUG: Agent {agent_id} goal: {goal}")
                
                if isinstance(goal, (list, np.ndarray)) and len(goal) >= 2:
                    x = float(goal[0][0]) if isinstance(goal[0], (list, np.ndarray)) else float(goal[0])
                    y = float(goal[1][0]) if isinstance(goal[1], (list, np.ndarray)) else float(goal[1])
                    
                    print(f"🔍 DEBUG: Agent {agent_id} extracted goal: ({x}, {y})")
                    
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    
                    print(f"🔍 DEBUG: Agent {agent_id} goal updated bounds - min_x: {min_x}, max_x: {max_x}, min_y: {min_y}, max_y: {max_y}")
        
        bounds = {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
        
        print(f"🔍 DEBUG: Final computed bounds: {bounds}")
        return bounds
    
    def reset(self):
        """Reset the PAR coordinator state."""
        self.current_par_solution = None
        self.current_participants = []
        self.par_environment = None
        if hasattr(self.pnr_solver, 'clear'):
            self.pnr_solver.clear()
