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

        # Ensure workspace bounds are available for logging and downstream tools
        # Use full world bounds instead of local region bounds
        try:
            full_bounds = workspace.get('bounds', {})
            self.par_environment.workspace_bounds = {
                'min_x': full_bounds.get('min_x', 0.0),
                'max_x': full_bounds.get('max_x', 10.0),
                'min_y': full_bounds.get('min_y', 0.0),
                'max_y': full_bounds.get('max_y', 10.0),
                'width': full_bounds.get('max_x', 10.0) - full_bounds.get('min_x', 0.0),
                'height': full_bounds.get('max_y', 10.0) - full_bounds.get('min_y', 0.0),
                'participants': deadlock_participants,
                'grid_resolution': self.config.get('GRID_RESOLUTION', 0.2)
            }
        except Exception:
            pass
        
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
            'grid_resolution': self.config.get('GRID_RESOLUTION', 0.2),
            'grid_offset': getattr(self.par_environment, 'grid_offset', (0, 0))  # Add grid_offset for debugging
        }

        # Add sub-map diagnostics
        try:
            width = sub_map.width if sub_map else 0
            height = sub_map.height if sub_map else 0
            occupancy = 0.0
            if sub_map and hasattr(sub_map, 'grid') and width > 0 and height > 0:
                occ = sum(1 for row in sub_map.grid for cell in row if cell)
                occupancy = occ / float(width * height)
            par_solver_input['sub_map_dims'] = {'width': width, 'height': height}
            par_solver_input['sub_map'] = sub_map
            par_solver_input['obstacle_occupancy'] = occupancy
            par_solver_input['participant_count'] = len(actor_set.actors) if actor_set and hasattr(actor_set, 'actors') else len(start_positions)
            # Per-agent grid diagnostics
            diagnostics = {}
            for aid, sp in start_positions.items():
                gp = goal_positions.get(aid)
                if isinstance(sp, tuple) and gp and isinstance(gp, tuple):
                    dx = gp[0] - sp[0]
                    dy = gp[1] - sp[1]
                    diagnostics[aid] = {
                        'start': sp,
                        'goal': gp,
                        'start_equals_goal': (dx == 0 and dy == 0),
                        'grid_distance': (dx*dx + dy*dy) ** 0.5
                    }
            par_solver_input['diagnostics'] = diagnostics
        except Exception:
            pass

        # Connectivity diagnostic: simple BFS reachability on current sub_map for each agent
        try:
            debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        except Exception:
            debug_mode = False
        if debug_mode and sub_map and hasattr(sub_map, 'grid'):
            def in_bounds(i, j):
                return 0 <= i < sub_map.height and 0 <= j < sub_map.width
            def is_free(i, j):
                # Assumption: 0 = free, 1 = obstacle
                return in_bounds(i, j) and sub_map.grid[i][j] == 0
            from collections import deque
            bfs_connectivity = {}
            for aid, sp in start_positions.items():
                gp = goal_positions.get(aid)
                if sp is None or gp is None:
                    bfs_connectivity[aid] = {'free_start': None, 'free_goal': None, 'reachable': None}
                    continue
                si, sj = sp[1], sp[0]
                gi, gj = gp[1], gp[0]
                reachable = False
                if is_free(si, sj) and is_free(gi, gj):
                    q = deque([(si, sj)])
                    seen = set([(si, sj)])
                    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                    while q:
                        ci, cj = q.popleft()
                        if ci == si and cj == sj:
                            pass
                        if ci == gi and cj == gj:
                            reachable = True
                            break
                        for di, dj in dirs:
                            ni, nj = ci + di, cj + dj
                            if (ni, nj) not in seen and is_free(ni, nj):
                                seen.add((ni, nj))
                                q.append((ni, nj))
                fs = is_free(si, sj)
                fg = is_free(gi, gj)
                bfs_connectivity[aid] = {'free_start': fs, 'free_goal': fg, 'reachable': reachable}
                print(f"PAR DIAG BFS: agent={aid} start={sp} goal={gp} free_start={fs} free_goal={fg} reachable={reachable}")
            # attach to solver input for logging
            par_solver_input['bfs_connectivity'] = bfs_connectivity

            # Connectivity extras: free-cell components and slack
            try:
                # Compute connected components over free cells
                comp_id = [[-1 for _ in range(sub_map.width)] for __ in range(sub_map.height)]
                comp_sizes = {}
                comp_idx = 0
                dirs = [(1,0),(-1,0),(0,1),(0,-1)]
                for i in range(sub_map.height):
                    for j in range(sub_map.width):
                        if is_free(i,j) and comp_id[i][j] == -1:
                            q = deque([(i,j)])
                            comp_id[i][j] = comp_idx
                            size = 0
                            while q:
                                ci,cj = q.popleft()
                                size += 1
                                for di,dj in dirs:
                                    ni,nj = ci+di,cj+dj
                                    if is_free(ni,nj) and comp_id[ni][nj] == -1:
                                        comp_id[ni][nj] = comp_idx
                                        q.append((ni,nj))
                            comp_sizes[comp_idx] = size
                            comp_idx += 1
                # Participant components
                part_comp = {}
                for aid, sp in start_positions.items():
                    if sp is None: continue
                    si,sj = sp[1], sp[0]
                    cid = comp_id[si][sj] if in_bounds(si,sj) else -1
                    part_comp[aid] = cid
                same_component = len(set([c for c in part_comp.values() if c is not None])) == 1 if part_comp else False
                first_cid = next(iter(part_comp.values())) if part_comp else None
                free_cells_total = sum(1 for i in range(sub_map.height) for j in range(sub_map.width) if is_free(i,j))
                free_cells_component = comp_sizes.get(first_cid, 0) if first_cid is not None else 0
                slack_component = free_cells_component - len(start_positions)
                par_solver_input['connectivity_extras'] = {
                    'same_component': same_component,
                    'participant_components': part_comp,
                    'component_sizes': comp_sizes,
                    'free_cells_total': free_cells_total,
                    'free_cells_component': free_cells_component,
                    'slack_component': slack_component,
                }
            except Exception:
                pass
        
        # --- Optional sub-map cropping to minimal bounding box over starts/goals ---
        # Introduce a config switch to disable cropping entirely for debugging
        grid_offset = (0, 0)
        disable_crop = bool(self.config.get('PAR_DISABLE_CROP', True)) if isinstance(self.config, dict) else True
        if not disable_crop:
            try:
                if sub_map and hasattr(sub_map, 'grid') and hasattr(sub_map, 'width') and hasattr(sub_map, 'height'):
                    if start_positions and goal_positions:
                        min_x = min([sp[0] for sp in start_positions.values()] + [gp[0] for gp in goal_positions.values()])
                        min_y = min([sp[1] for sp in start_positions.values()] + [gp[1] for gp in goal_positions.values()])
                        max_x = max([sp[0] for sp in start_positions.values()] + [gp[0] for gp in goal_positions.values()])
                        max_y = max([sp[1] for sp in start_positions.values()] + [gp[1] for gp in goal_positions.values()])

                        pad = int(self.config.get('PAR_CROP_PADDING', 1)) if isinstance(self.config, dict) else 1
                        min_x = max(0, min_x - pad)
                        min_y = max(0, min_y - pad)
                        max_x = min(sub_map.width - 1, max_x + pad)
                        max_y = min(sub_map.height - 1, max_y + pad)

                        # Crop grid (rows = y, cols = x)
                        cropped_grid = [row[min_x:max_x + 1] for row in sub_map.grid[min_y:max_y + 1]]

                        # Apply crop to sub_map
                        sub_map.grid = cropped_grid
                        sub_map.width = max_x - min_x + 1
                        sub_map.height = max_y - min_y + 1

                        # Record local->original grid offset
                        grid_offset = (min_x, min_y)
            except Exception:
                # On any issue, skip cropping and keep full sub_map
                grid_offset = (0, 0)

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

            # Remap real agent ids -> solver-local contiguous ids [0..k-1]
            # Many MAPF implementations assume contiguous ids for internal arrays
            participants_real_ids = list(start_positions.keys())
            id_real_to_solver: Dict[int, int] = {rid: idx for idx, rid in enumerate(participants_real_ids)}
            id_solver_to_real: Dict[int, int] = {v: k for k, v in id_real_to_solver.items()}

            # Build a solver-local ActorSet with contiguous ids
            from python_pnr.actor_set import ActorSet as SolverActorSet
            from python_pnr.actor import Actor as SolverActor
            from python_pnr.node import Point as SolverPoint

            solver_actor_set = SolverActorSet()
            for rid in participants_real_ids:
                sp = start_positions.get(rid)
                gp = goal_positions.get(rid)
                if sp is None or gp is None:
                    # Skip if either missing
                    continue
                sid = id_real_to_solver[rid]
                # Coordinate mapping: our start/goal tuples are (x, y) = (col, row) -> SolverPoint(x, y)
                # Adjust to cropped local coordinates by subtracting grid_offset
                try:
                    ox, oy = grid_offset
                except Exception:
                    ox, oy = 0, 0
                s_point = SolverPoint(int(sp[0] - ox), int(sp[1] - oy))
                g_point = SolverPoint(int(gp[0] - ox), int(gp[1] - oy))
                solver_actor_set.add_actor(SolverActor(sid, s_point, g_point))
            
            # Add solver input information to logging
            par_solver_input['mapf_config'] = {
                'max_steps': mapf_config.max_steps,
                'timeout': mapf_config.timeout,
                'heuristic_weight': mapf_config.heuristic_weight
            }
            
            # Add solver actor set information
            solver_actors_info = []
            for actor in solver_actor_set:
                solver_actors_info.append({
                    'id': actor.id,
                    'start': (actor.current.x, actor.current.y),
                    'goal': (actor.goal.x, actor.goal.y)
                })
            par_solver_input['solver_actor_set'] = solver_actors_info
            
            # Add ID mapping information
            par_solver_input['id_mapping'] = {
                'real_to_solver': id_real_to_solver,
                'solver_to_real': id_solver_to_real
            }
            
            # Call solver with the original sub_map
            chosen_api = 'start_search'
            result = self.pnr_solver.start_search(sub_map, mapf_config, solver_actor_set)
            
            # Store the ID mapping in the result for later use
            result.id_solver_to_real = id_solver_to_real
            result.id_real_to_solver = id_real_to_solver
            # Also store the cropping offset so executor can restore to original grid frame
            try:
                result.grid_offset = grid_offset
            except Exception:
                pass

            # Capture solver meta info if available
            solution_meta = {}
            try:
                # Pre-solver diagnostics
                try:
                    # Count actors passed to solver
                    actor_count = 0
                    try:
                        for _ in solver_actor_set:
                            actor_count += 1
                    except Exception:
                        actor_count = len(getattr(solver_actor_set, 'actors', [])) if hasattr(solver_actor_set, 'actors') else 0
                    solution_meta['solver_actor_count'] = actor_count

                    # Traversability checks (solver view: 0=free)
                    starts_trav = {}
                    goals_trav = {}
                    starts_equal_goals = {}
                    eff_participants = 0
                    if sub_map and hasattr(sub_map, 'grid'):
                        height = sub_map.height
                        width = sub_map.width
                        def in_bounds(i,j):
                            return 0 <= i < height and 0 <= j < width
                        def is_free(i,j):
                            return in_bounds(i,j) and sub_map.grid[i][j] == 0
                        for aid, sp in start_positions.items():
                            gp = goal_positions.get(aid)
                            si, sj = (sp[1], sp[0]) if isinstance(sp, tuple) else (None, None)
                            gi, gj = (gp[1], gp[0]) if isinstance(gp, tuple) else (None, None)
                            st = is_free(si, sj) if si is not None else None
                            gt = is_free(gi, gj) if gi is not None else None
                            eq = (sp == gp) if (sp is not None and gp is not None) else None
                            starts_trav[aid] = st
                            goals_trav[aid] = gt
                            starts_equal_goals[aid] = eq
                            if sp is not None and gp is not None:
                                eff_participants += 1
                    solution_meta['starts_traversable'] = starts_trav
                    solution_meta['goals_traversable'] = goals_trav
                    solution_meta['starts_equal_goals'] = starts_equal_goals
                    solution_meta['effective_participants'] = eff_participants
                    if actor_count < 2:
                        solution_meta.setdefault('failure_reason', 'insufficient_actors')
                except Exception:
                    pass

                if hasattr(result, 'runtime'):
                    solution_meta['runtime'] = result.runtime
                if hasattr(result, 'steps'):
                    solution_meta['steps'] = result.steps
                if hasattr(result, 'stats'):
                    solution_meta['stats'] = result.stats
                # also capture solver attributes if exposed
                for attr in ['expanded_nodes', 'generated_nodes', 'timeouts', 'max_frontier', 'failure_reason']:
                    if hasattr(self.pnr_solver, attr):
                        solution_meta[attr] = getattr(self.pnr_solver, attr)
                # attach into input for logger persistence
                try:
                    par_solver_input['solution_meta'] = solution_meta
                except Exception:
                    pass
            except Exception:
                pass

            # Ensure agents_moves are attached to result BEFORE logging (remap solver ids -> real ids)
            try:
                if hasattr(self.pnr_solver, 'agents_moves') and self.pnr_solver.agents_moves is not None:
                    remapped_moves = []
                    for mv in self.pnr_solver.agents_moves:
                        try:
                            solver_id = getattr(mv, 'id', None)
                            real_id = id_solver_to_real.get(solver_id, solver_id)
                            mv.id = real_id
                            remapped_moves.append(mv)
                        except Exception:
                            remapped_moves.append(mv)
                    result.agents_moves = remapped_moves
                    if debug_mode:
                        print(f"PAR TRAJECTORY DEBUG: Found {len(remapped_moves)} moves, remapped to real IDs")
                        for i, mv in enumerate(remapped_moves[:5]):  # Show first 5 moves
                            print(f"  Move {i}: agent_id={getattr(mv, 'id', 'N/A')}, di={getattr(mv, 'di', 'N/A')}, dj={getattr(mv, 'dj', 'N/A')}")
                else:
                    if debug_mode:
                        print(f"PAR TRAJECTORY DEBUG: No agents_moves found in solver")
            except Exception as e:
                if debug_mode:
                    print(f"PAR TRAJECTORY DEBUG: Error processing moves: {e}")
                pass

            # Minimal solver diagnostics
            try:
                debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
            except Exception:
                debug_mode = False
            if debug_mode:
                moves_cnt = len(getattr(self.pnr_solver, 'agents_moves', []) or [])
                paths_cnt = len(getattr(self.pnr_solver, 'agents_paths', []) or [])
                succ_flag = getattr(result, 'success', None)
                print(f"PAR DIAG RESULT: api={chosen_api} success={succ_flag} moves={moves_cnt} paths={paths_cnt}")
            
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
                
                # agents_moves already attached above before logging; keep as-is
                return result
            else:
                # print(f"   ❌ PNR solver failed or returned no solution")
                # Disable fallback: return empty unsuccessful result for focused PAR debugging
                empty_result = MAPFSearchResult()
                empty_result.success = False
                empty_result.agents_moves = []
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.log_par_solver_details(par_solver_input, empty_result, list(start_positions.keys()))
                
                return empty_result
            
        except Exception as e:
            # print(f"❌ Error in PNR solver: {e}")
            import traceback
            traceback.print_exc()
            # Disable fallback on exception: return empty unsuccessful result
            empty_result = MAPFSearchResult()
            empty_result.success = False
            empty_result.agents_moves = []
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_par_solver_details(par_solver_input, empty_result, list(start_positions.keys()))
            
            return empty_result
    
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
        """Disabled: Fallback solution generation is turned off to force PNR only."""
        # Fallback disabled: return empty unsuccessful result
        result = MAPFSearchResult()
        result.success = False
        result.agents_moves = []
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
        
        # First, try to get path from paths (PNR original output)
        if hasattr(self.current_par_solution, 'paths') and self.current_par_solution.paths:
            # Try both string and integer keys
            for agent_key in [str(agent_id), agent_id]:
                if agent_key in self.current_par_solution.paths:
                    path = self.current_par_solution.paths[agent_key]
                    if path:
                        try:
                            # If solver ran on cropped local grid, restore to global grid using grid_offset
                            if hasattr(self.current_par_solution, 'grid_offset') and self.current_par_solution.grid_offset:
                                ox, oy = self.current_par_solution.grid_offset
                                adjusted = []
                                for pt in path:
                                    if hasattr(pt, 'x') and hasattr(pt, 'y'):
                                        adjusted.append((pt.x + ox, pt.y + oy))
                                    else:
                                        adjusted.append((pt[0] + ox, pt[1] + oy))
                                print(f"PAR COORDINATOR: Using PNR original path for agent {agent_id}: {adjusted[:5]}...")
                                return adjusted
                        except Exception:
                            pass
                        print(f"PAR COORDINATOR: Using PNR original path for agent {agent_id}: {path[:5]}...")
                        return path
        
        # Extract path for the specific agent from the PAR solution
        # This depends on the structure of MAPFSearchResult
        if hasattr(self.current_par_solution, 'get_agent_path'):
            return self.current_par_solution.get_agent_path(agent_id)
        
        # Fallback: try to extract from agents_moves if available
        if hasattr(self.current_par_solution, 'agents_moves'):
            print(f"PAR COORDINATOR: Using reconstructed path from moves for agent {agent_id}")
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
            if getattr(move, 'id', None) == agent_id:
                # ActorMove uses di/dj naming where di=row_increment, dj=col_increment
                # In RL coordinate system: x=col, y=row, so dj->x, di->y
                new_pos = (current_pos[0] + getattr(move, 'dj', 0),  # dj is column increment -> x
                           current_pos[1] + getattr(move, 'di', 0))  # di is row increment -> y
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
        Use YAML config world dimensions directly from env_base.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict: Workspace information with YAML world bounds
        """
        # Get world dimensions from gym_env (loaded from YAML by env_base)
        width = float(self.gym_env._env_base__width)  # world_width from YAML
        height = float(self.gym_env._env_base__height)  # world_height from YAML
        offset_x = float(self.gym_env.offset_x)
        offset_y = float(self.gym_env.offset_y)
        
        bounds = {
            'min_x': offset_x,
            'min_y': offset_y,
            'max_x': offset_x + width,
            'max_y': offset_y + height
        }

        # Enrich workspace with obstacle map if available
        map_matrix = None
        xy_reso = None
        try:
            if hasattr(self.gym_env, 'components') and isinstance(self.gym_env.components, dict):
                map_matrix = self.gym_env.components.get('map_matrix', None)
            if hasattr(self.gym_env, 'xy_reso'):
                xy_reso = float(getattr(self.gym_env, 'xy_reso'))
        except Exception:
            pass
        
        workspace = {
            'bounds': bounds,
            'obstacles': self._get_environment_obstacles(),
            'grid_resolution': self.config.get('GRID_RESOLUTION', 1.0),
            'map_matrix': map_matrix,
            'xy_reso': xy_reso,
            'offset_x': getattr(self.gym_env, 'offset_x', 0.0),
            'offset_y': getattr(self.gym_env, 'offset_y', 0.0)
        }
        
        return workspace
    
    def _get_environment_obstacles(self) -> List:
        """Get obstacles from the environment."""
        try:
            # Try to get obstacles from the gym environment
            if hasattr(self, 'gym_env') and self.gym_env:
                # Prefer extracting from known simulator components
                if hasattr(self.gym_env, 'components') and isinstance(self.gym_env.components, dict):
                    comp = self.gym_env.components
                    obstacles: List = []
                    # Polygons
                    try:
                        if 'obs_polygons' in comp and hasattr(comp['obs_polygons'], 'obs_poly_list'):
                            obstacles.extend(list(comp['obs_polygons'].obs_poly_list))
                    except Exception:
                        pass
                    # Circles
                    try:
                        if 'obs_circles' in comp and hasattr(comp['obs_circles'], 'obs_cir_list'):
                            obstacles.extend(list(comp['obs_circles'].obs_cir_list))
                    except Exception:
                        pass
                    # If an explicit obstacles list exists, include it as well
                    try:
                        if 'obstacles' in comp and isinstance(comp['obstacles'], list):
                            obstacles.extend(comp['obstacles'])
                    except Exception:
                        pass
                    if len(obstacles) > 0:
                        return obstacles
                
                # Try alternative obstacle access methods as fallbacks
                if hasattr(self.gym_env, 'world') and hasattr(self.gym_env.world, 'obstacles'):
                    obstacles = self.gym_env.world.obstacles
                    return obstacles
                
                if hasattr(self.gym_env, 'get_obstacles'):
                    obstacles = self.gym_env.get_obstacles()
                    return obstacles
            
            # print(f"🔍 PAR COORDINATOR: No obstacles found in environment")
            return []
            
        except Exception as e:
            # print(f"⚠️ Warning: Could not get environment obstacles: {e}")
            return []
    
    
    def reset(self):
        """Reset the PAR coordinator state."""
        self.current_par_solution = None
        self.current_participants = []
        self.par_environment = None
        if hasattr(self.pnr_solver, 'clear'):
            self.pnr_solver.clear()
