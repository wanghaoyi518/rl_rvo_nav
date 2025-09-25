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
        # Substep execution for each grid edge (e.g., 0.5 grid split into n substeps)
        self.substeps_per_grid = config.get('PAR_SUBSTEPS_PER_GRID', 10)
        self.agent_substep_index = {}
        
        # Execution state tracking
        self.agent_paths = {}  # Dictionary mapping agent_id to current path index
        self.agent_start_positions = {}  # Dictionary mapping agent_id to start position
        self.agent_goal_positions = {}  # Dictionary mapping agent_id to goal position
        self.state_manager = None
        self.par_coordinator = None

    def set_dependencies(self, state_manager, par_coordinator):
        """Inject StateManager and PARCoordinator dependencies."""
        self.state_manager = state_manager
        self.par_coordinator = par_coordinator
    
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
        debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        
        # Check if agent is moving to start position
        if self.is_moving_to_start(agent_id):
            if debug_mode:
                print(f"PAR EXECUTOR: Agent {agent_id} moving to start position")
            result = self.move_to_par_start(agent_id, agent_states)
            # If到达起点则切换到执行阶段
            if result.get('mode') == 'at_start' and self.state_manager is not None:
                self.state_manager.set_par_executing(agent_id)
            return result
        
        # Check if agent is following PAR path
        elif self.is_following_par_path(agent_id):
            if debug_mode:
                print(f"PAR EXECUTOR: Agent {agent_id} following PAR path")
            # Get PAR solution from state manager
            par_solution = self.get_par_solution(agent_id)
            if par_solution is None:
                if debug_mode:
                    print(f"PAR EXECUTOR: No PAR solution for agent {agent_id}")
                return {'action': np.array([0.0, 0.0]), 'mode': 'no_solution', 'target': None}
            return self.follow_par_path(agent_id, par_solution, agent_states)
        
        # Default: no action
        if debug_mode:
            print(f"PAR EXECUTOR: Agent {agent_id} in idle mode")
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
        
        # If a PAR path exists, directly set to the first waypoint and mark as at_start (enter executing next)
        if hasattr(self, '_agent_full_paths') and agent_id in getattr(self, '_agent_full_paths', {}):
            path = self._agent_full_paths.get(agent_id) or []
            if isinstance(path, list) and len(path) > 0:
                first_wp = path[0]
                # Advance index so the next step uses the second waypoint
                self.agent_paths[agent_id] = 1
                # Initialize substep index for this agent
                self.agent_substep_index[agent_id] = 0
                return {
                    'action': np.array([0.0, 0.0]),
                    'mode': 'at_start',
                    'target': first_wp,
                    'set_position': first_wp
                }

        # Check if already at start position using current vs start
        if self.is_at_position(current_position, start_position):
            return {
                'action': np.array([0.0, 0.0]),
                'mode': 'at_start',
                'target': start_position
            }
        
        # Directly set agent position to start position (no velocity needed)
        return {
            'action': np.array([0.0, 0.0]),  # No velocity needed
            'mode': 'move_to_start',
            'target': start_position,
            'set_position': start_position  # Direct position setting
        }
    
    def follow_par_path(self, agent_id: int, par_solution, agent_states: Dict) -> Dict:
        """
        Follow the PAR path for the given agent with step-by-step execution.
        Simplified: directly extract next position from PNR path.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution containing paths
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Action dictionary for following PAR path
        """
        debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        
        if agent_id not in agent_states:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        agent_state = agent_states[agent_id]
        current_position = self.get_agent_position(agent_state)
        
        if current_position is None:
            return {'action': np.array([0.0, 0.0]), 'mode': 'error', 'target': None}
        
        # Get current path index
        path_index = self.agent_paths.get(agent_id, 0)

        # Get PNR path directly from solution
        pnr_path = self.get_pnr_path_direct(agent_id, par_solution)

        if not pnr_path or path_index >= len(pnr_path):
            if debug_mode:
                print(f"PAR EXECUTOR: Agent {agent_id} path complete. Path length: {len(pnr_path) if pnr_path else 0}, Index: {path_index}")
            return {
                'action': np.array([0.0, 0.0]),
                'mode': 'path_complete',
                'target': None
            }

        # Determine previous and next grid points
        # next grid point
        next_point = pnr_path[path_index]
        if hasattr(next_point, 'x') and hasattr(next_point, 'y'):
            next_grid = (next_point.x, next_point.y)
        else:
            next_grid = next_point

        # previous grid point (use current position if at start)
        if path_index > 0:
            prev_point = pnr_path[path_index - 1]
            if hasattr(prev_point, 'x') and hasattr(prev_point, 'y'):
                prev_grid = (prev_point.x, prev_point.y)
            else:
                prev_grid = prev_point
            prev_xy = self.par_coordinator.par_environment.grid_to_continuous(prev_grid)
        else:
            prev_xy = current_position

        # continuous next point
        next_xy = self.par_coordinator.par_environment.grid_to_continuous(next_grid)

        # Substep interpolation between prev_xy -> next_xy
        n = int(self.substeps_per_grid) if hasattr(self, 'substeps_per_grid') else 10
        sub_idx = int(self.agent_substep_index.get(agent_id, 0))
        alpha = float(sub_idx + 1) / float(max(n, 1))
        pos_x = prev_xy[0] + alpha * (next_xy[0] - prev_xy[0])
        pos_y = prev_xy[1] + alpha * (next_xy[1] - prev_xy[1])
        interp_pos = (pos_x, pos_y)

        if debug_mode:
            print(f"PAR EXECUTOR: Agent {agent_id} path_idx={path_index}, sub_idx={sub_idx}/{n-1}, prev={prev_xy}, next={next_xy}, set={interp_pos}")

        # Advance substep or path index
        if sub_idx + 1 < n:
            # Stay on the same path index, advance substep
            self.agent_substep_index[agent_id] = sub_idx + 1
            return {
                'action': np.array([0.0, 0.0]),
                'mode': 'follow_path',
                'target': interp_pos,
                'path_index': path_index,
                'path_length': len(pnr_path),
                'set_position': interp_pos
            }
        else:
            # Finish this edge: reset substep and advance to next waypoint
            self.agent_substep_index[agent_id] = 0
            self.agent_paths[agent_id] = path_index + 1

            # If this was the last waypoint, mark completion
            if self.agent_paths[agent_id] >= len(pnr_path):
                return {
                    'action': np.array([0.0, 0.0]),
                    'mode': 'path_complete',
                    'target': next_xy,
                    'path_index': path_index,
                    'path_length': len(pnr_path),
                    'set_position': next_xy
                }
            else:
                return {
                    'action': np.array([0.0, 0.0]),
                    'mode': 'follow_path',
                    'target': next_xy,
                    'path_index': path_index,
                    'path_length': len(pnr_path),
                    'set_position': next_xy
                }
    
    def get_pnr_path_direct(self, agent_id: int, par_solution) -> List:
        """
        Get PNR path directly from solution without complex conversion.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution containing paths
            
        Returns:
            List: PNR path (list of Point objects or tuples)
        """
        if par_solution is None:
            return []
        
        # Try to get path from paths (PNR original output)
        if hasattr(par_solution, 'paths') and par_solution.paths:
            print(f"PAR EXECUTOR: Available paths keys: {list(par_solution.paths.keys())}")
            print(f"PAR EXECUTOR: Looking for agent {agent_id} path")
            
            # Get the correct mapping from PAR coordinator
            if hasattr(par_solution, 'id_solver_to_real') and par_solution.id_solver_to_real:
                # Use the correct mapping from coordinator
                id_solver_to_real = par_solution.id_solver_to_real
                print(f"PAR EXECUTOR: Using mapping: {id_solver_to_real}")
                
                # Find the solver ID for this real agent ID
                solver_id = None
                for sid, rid in id_solver_to_real.items():
                    if rid == agent_id:
                        solver_id = sid
                        break
                
                if solver_id is not None and solver_id in par_solution.paths:
                    path = par_solution.paths[solver_id]
                    # If paths are expressed in cropped local grid, restore to global grid using grid_offset
                    try:
                        if hasattr(par_solution, 'grid_offset') and par_solution.grid_offset:
                            ox, oy = par_solution.grid_offset
                            adjusted = []
                            for pt in path:
                                if hasattr(pt, 'x') and hasattr(pt, 'y'):
                                    adjusted.append((pt.x + ox, pt.y + oy))
                                else:
                                    adjusted.append((pt[0] + ox, pt[1] + oy))
                            return adjusted
                    except Exception:
                        pass
                    if path:
                        print(f"PAR EXECUTOR: Found path for agent {agent_id} using solver ID {solver_id}, length: {len(path)}")
                        return path
                else:
                    print(f"PAR EXECUTOR: No solver ID found for agent {agent_id} in mapping {id_solver_to_real}")
            else:
                # Fallback: try direct mapping first
                for agent_key in [str(agent_id), agent_id]:
                    if agent_key in par_solution.paths:
                        path = par_solution.paths[agent_key]
                        if path:
                            print(f"PAR EXECUTOR: Found path for agent {agent_id} with key {agent_key}, length: {len(path)}")
                            return path
                
                print(f"PAR EXECUTOR: No mapping available, trying fallback logic")
            
            print(f"PAR EXECUTOR: No path found for agent {agent_id}")
        else:
            print(f"PAR EXECUTOR: No paths attribute or empty paths")
        
        return []
    
    def is_moving_to_start(self, agent_id: int) -> bool:
        """
        Check if agent is in the moving to start phase.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is moving to start position
        """
        if agent_id not in self.agent_start_positions:
            return False
        # If state manager says executing, not moving-to-start anymore
        if self.state_manager is not None and self.state_manager.is_par_executing(agent_id):
            return False
        return True
    
    def get_par_solution(self, agent_id: int):
        """
        Get PAR solution for the given agent from state manager.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            PAR solution or None if not available
        """
        if self.state_manager is None:
            return None
        return self.state_manager.get_par_solution(agent_id)
    
    def is_following_par_path(self, agent_id: int) -> bool:
        """
        Check if agent is following PAR path.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if agent is following PAR path
        """
        if self.state_manager is not None and self.state_manager.is_par_executing(agent_id):
            return True
        return False
    
    def is_at_start_position(self, agent_id: int, start_position: Tuple[float, float]) -> bool:
        """
        Check if agent has reached its start position.
        
        Args:
            agent_id: ID of the agent
            start_position: Start position to check against
            
        Returns:
            bool: True if agent is at start position
        """
        # Check actual distance
        if self.state_manager is None:
            return False
        # Try to get current position from state_manager last_position or agent_states via caller
        # Here we only compare start_position with executor's last known start target using tolerance
        # We rely on compute_velocity_to_target to converge to within tolerance
        # If current position is close enough to start_position, return True
        # Note: current_position should be passed via move_to_par_start using agent_states
        return False
    
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
        Get agent path from PAR solution and convert to continuous coordinates.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution object
            
        Returns:
            List[Tuple[float, float]]: List of continuous positions in the agent's path
        """
        if par_solution is None or not hasattr(par_solution, 'agents_moves'):
            return []
        
        # Extract moves for this agent
        agent_moves = []
        for move in par_solution.agents_moves:
            if move.id == agent_id:
                agent_moves.append(move)
        
        if not agent_moves:
            return []
        
        # Get grid path from coordinator
        grid_path = []
        if self.par_coordinator is not None:
            grid_path = self.par_coordinator.get_agent_path(agent_id)
        
        if not grid_path:
            # Fallback: reconstruct grid path from moves
            origin = (0, 0)
            if self.par_coordinator is not None and hasattr(self.par_coordinator, 'par_environment') and self.par_coordinator.par_environment and hasattr(self.par_coordinator.par_environment, 'actor_set'):
                try:
                    for actor in self.par_coordinator.par_environment.actor_set:
                        if getattr(actor, 'id', None) == agent_id and hasattr(actor, 'current'):
                            origin = (actor.current.x, actor.current.y)
                            break
                except Exception:
                    pass
            
            grid_path = [origin]
            current_pos = origin
            for move in agent_moves:
                # ActorMove uses di/dj naming where di=row_increment, dj=col_increment
                # In RL coordinate system: x=col, y=row, so dj->x, di->y
                next_pos = (current_pos[0] + move.dj, current_pos[1] + move.di)
                grid_path.append(next_pos)
                current_pos = next_pos
        
        # Convert grid path to continuous coordinates
        continuous_path = []
        if self.par_coordinator is not None and hasattr(self.par_coordinator, 'par_environment') and self.par_coordinator.par_environment:
            print(f"PAR EXECUTOR: Converting grid path for agent {agent_id}: {grid_path[:5]}...")
            for grid_pos in grid_path:
                # Handle both Point objects and (x, y) tuples
                if hasattr(grid_pos, 'x') and hasattr(grid_pos, 'y'):
                    # Point object: extract x, y coordinates
                    grid_tuple = (grid_pos.x, grid_pos.y)
                else:
                    # Already a tuple
                    grid_tuple = grid_pos
                
                continuous_pos = self.par_coordinator.par_environment.grid_to_continuous(grid_tuple)
                continuous_path.append(continuous_pos)
                if len(continuous_path) <= 5:  # Only print first 5 conversions
                    print(f"PAR EXECUTOR: Grid {grid_tuple} -> Continuous {continuous_pos}")
        else:
            print(f"PAR EXECUTOR: Using fallback conversion for agent {agent_id}")
            # Fallback: assume grid resolution and bounds
            grid_resolution = self.config.get('GRID_RESOLUTION', 0.5)
            for grid_pos in grid_path:
                # Simple conversion assuming origin at (0,0)
                continuous_pos = (
                    (grid_pos[0] + 0.5) * grid_resolution,
                    (grid_pos[1] + 0.5) * grid_resolution
                )
                continuous_path.append(continuous_pos)
                if len(continuous_path) <= 5:  # Only print first 5 conversions
                    print(f"PAR EXECUTOR: Fallback Grid {grid_pos} -> Continuous {continuous_pos}")
        
        return continuous_path
    
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
        # Save full path and reset index
        if not hasattr(self, '_agent_full_paths'):
            self._agent_full_paths = {}
        self._agent_full_paths[agent_id] = path
        self.agent_paths[agent_id] = 0  # Reset path index
        # Reset substep index
        self.agent_substep_index[agent_id] = 0
        
        # Debug output
        debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        if debug_mode:
            print(f"PAR EXECUTOR: Set path for agent {agent_id} with {len(path)} waypoints")
            if path:
                print(f"  Start: {path[0]}, End: {path[-1]}")
                print(f"  First few waypoints: {path[:min(5, len(path))]}")
    
    def initialize_par_execution(self, agent_id: int, par_solution):
        """
        Initialize PAR execution for an agent by converting grid path to continuous path.
        
        Args:
            agent_id: ID of the agent
            par_solution: PAR solution containing the path
        """
        if par_solution is None:
            return
        
        # Convert grid path to continuous path
        continuous_path = self.get_agent_path_from_solution(agent_id, par_solution)
        
        if continuous_path:
            self.set_agent_path(agent_id, continuous_path)
            
            # Debug output
            debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
            if debug_mode:
                print(f"PAR EXECUTOR: Initialized execution for agent {agent_id}")
                print(f"  Grid path length: {len(continuous_path)}")
                print(f"  Continuous path: {continuous_path[:3]}...{continuous_path[-3:] if len(continuous_path) > 6 else continuous_path[3:]}")
        else:
            debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
            if debug_mode:
                print(f"PAR EXECUTOR: Warning - No path found for agent {agent_id}")
    
    def get_path_progress(self, agent_id: int) -> Dict:
        """
        Get the current progress of an agent along its PAR path.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dict: Progress information including current index, total length, and percentage
        """
        path_index = self.agent_paths.get(agent_id, 0)
        total_length = 0
        
        if hasattr(self, '_agent_full_paths') and agent_id in self._agent_full_paths:
            total_length = len(self._agent_full_paths[agent_id])
        
        progress_percentage = (path_index / total_length * 100) if total_length > 0 else 0
        
        # Debug: print when agent has no path
        if total_length == 0:
            print(f"PAR EXECUTOR DEBUG: Agent {agent_id} has no path (total_length=0)")
            print(f"  _agent_full_paths exists: {hasattr(self, '_agent_full_paths')}")
            if hasattr(self, '_agent_full_paths'):
                print(f"  _agent_full_paths keys: {list(self._agent_full_paths.keys())}")
                print(f"  agent_id in _agent_full_paths: {agent_id in self._agent_full_paths}")
        
        return {
            'agent_id': agent_id,
            'current_index': path_index,
            'total_length': total_length,
            'progress_percentage': progress_percentage,
            'is_complete': path_index >= total_length if total_length > 0 else False
        }
    
    def is_par_complete(self, agent_id: int) -> bool:
        """
        Check if PAR execution is complete for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            bool: True if PAR is complete for the agent
        """
        # Check if agent has a path assigned
        if agent_id not in self.agent_paths:
            return False
        
        # Check if agent has completed its path
        current_path_index = self.agent_paths[agent_id]
        
        # Get the agent's full path length
        total_path_length = 0
        if hasattr(self, '_agent_full_paths') and agent_id in self._agent_full_paths:
            total_path_length = len(self._agent_full_paths[agent_id])
        
        # Complete if we've reached the end of the path
        if total_path_length > 0 and current_path_index >= total_path_length:
            # print(f"🔄 PAR Executor: Agent {agent_id} completed path execution ({current_path_index}/{total_path_length})")
            return True
        
        return False
    
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
