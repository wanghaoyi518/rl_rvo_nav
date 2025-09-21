"""
PAR Environment Builder Module

This module provides functionality to build the environment for Push and Rotate (PAR) algorithm.
It handles sub-map construction, start/goal position computation, and region expansion.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from python_pnr.sub_map import SubMap
from python_pnr.actor_set import ActorSet
from python_pnr.actor import Actor


class PAREnvironment:
    """
    Environment builder for Push and Rotate (PAR) algorithm.
    
    This class handles the construction of the local environment for PAR execution,
    including sub-map creation, start/goal position computation, and region expansion.
    """
    
    def __init__(self, workspace: Dict, participants: List[int], config: Dict):
        """
        Initialize the PAR environment builder.
        
        Args:
            workspace: Workspace configuration dictionary
            participants: List of agent IDs participating in PAR
            config: Configuration dictionary containing PAR parameters
        """
        self.workspace = workspace
        self.participants = participants
        self.config = config
        self.par_offset = config.get('PAR_OFFSET', 2)
        self.grid_resolution = config.get('GRID_RESOLUTION', 1.0)
        
        # Environment boundaries
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
        
        # Sub-map and actor set
        self.sub_map = None
        self.actor_set = None
        # Global->local grid offset when submap is cropped from full map_matrix
        self.grid_offset: Tuple[int, int] = (0, 0)
        # Cache workspace fields when available
        self._ws_xy_reso: Optional[float] = None
        self._ws_offset_x: float = 0.0
        self._ws_offset_y: float = 0.0
        self._ws_map_matrix = None

    def _read_workspace_params(self):
        """Read and cache workspace parameters used for world<->grid mapping."""
        try:
            if hasattr(self.workspace, 'get') and callable(self.workspace.get):
                self._ws_map_matrix = self.workspace.get('map_matrix', None)
                self._ws_xy_reso = self.workspace.get('xy_reso', None)
                self._ws_offset_x = float(self.workspace.get('offset_x', 0.0))
                self._ws_offset_y = float(self.workspace.get('offset_y', 0.0))
        except Exception:
            pass
        # Ensure grid_resolution equals xy_reso when map_matrix exists
        if self._ws_map_matrix is not None and self._ws_xy_reso is not None:
            try:
                self.grid_resolution = float(self._ws_xy_reso)
            except Exception:
                pass

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous world coordinates to global grid column/row indices.

        Uses workspace xy_reso and offsets. Returns (col, row).
        """
        if self._ws_xy_reso is None:
            # Fallback to previous mapping relative to current region
            col = int((x - self.min_x) / self.grid_resolution)
            row = int((y - self.min_y) / self.grid_resolution)
            return (col, row)
        col = int((float(x) - self._ws_offset_x) / float(self._ws_xy_reso))
        row = int((float(y) - self._ws_offset_y) / float(self._ws_xy_reso))
        return (col, row)

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert global grid column/row to world coordinates (cell center)."""
        if self._ws_xy_reso is None:
            # Fallback to previous mapping relative to current region
            x = self.min_x + (col + 0.5) * self.grid_resolution
            y = self.min_y + (row + 0.5) * self.grid_resolution
            return (x, y)
        x = self._ws_offset_x + (int(col) + 0.5) * float(self._ws_xy_reso)
        y = self._ws_offset_y + (int(row) + 0.5) * float(self._ws_xy_reso)
        return (x, y)
    
    def build_par_environment(self, agent_states: Dict) -> Tuple[SubMap, ActorSet]:
        """
        Build the complete PAR environment including sub-map and actor set.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Tuple[SubMap, ActorSet]: Built sub-map and actor set for PAR
        """
        # print(f"🔍 DEBUG: PAREnvironment.build_par_environment called with {len(agent_states)} agents")
        # print(f"🔍 DEBUG: Participants: {self.participants}")
        # print(f"🔍 DEBUG: Agent states keys: {list(agent_states.keys())}")
        
        # Read workspace params and decide source
        self._read_workspace_params()
        has_map = (self._ws_map_matrix is not None and self._ws_xy_reso is not None)

        if has_map:
            # Full vs cropped submap mode
            try:
                submap_mode = str(self.config.get('PAR_SUBMAP_MODE', 'crop')).lower() if isinstance(self.config, dict) else 'crop'
            except Exception:
                submap_mode = 'crop'

            # If full mode: use entire map_matrix as sub_map
            if submap_mode == 'full':
                import numpy as _np
                m = _np.array(self._ws_map_matrix, dtype=float)
                bin_grid = (m != 0).astype(int)
                sub_map = SubMap(bin_grid.tolist())
                sub_map.height = bin_grid.shape[0]
                sub_map.width = bin_grid.shape[1]
                sub_map.origin_x = float(self._ws_offset_x)
                sub_map.origin_y = float(self._ws_offset_y)
                sub_map.resolution = float(self._ws_xy_reso)
                self.sub_map = sub_map
                self.grid_offset = (0, 0)
                self.min_x = sub_map.origin_x
                self.min_y = sub_map.origin_y
                self.max_x = sub_map.origin_x + sub_map.width * sub_map.resolution
                self.max_y = sub_map.origin_y + sub_map.height * sub_map.resolution
            else:
                # Derive global grid coords for participants' start/goal, compute bbox, slice directly
                starts_global = {}
                goals_global = {}
                for agent_id in self.participants:
                    if agent_id not in agent_states:
                        continue
                    pos = self.get_agent_position(agent_states[agent_id])
                    goal = self.get_agent_goal(agent_states[agent_id])
                    if pos is not None:
                        cg = self.world_to_grid(pos[0], pos[1])
                        starts_global[agent_id] = cg
                    if goal is not None:
                        gg = self.world_to_grid(goal[0], goal[1])
                        goals_global[agent_id] = gg

                # Compute bbox in global grid
                if submap_mode != 'full':
                    cols = []
                    rows = []
                    for d in (starts_global, goals_global):
                        for _, (c, r) in d.items():
                            cols.append(int(c))
                            rows.append(int(r))
                    if len(cols) == 0 or len(rows) == 0:
                        # Fallback to previous region method if no positions
                        self.compute_region_boundaries(agent_states)
                        self.expand_region()
                        self.sub_map = self.build_sub_map()
                    else:
                        import numpy as _np
                        h, w = _np.array(self._ws_map_matrix).shape
                        pad = int(self.config.get('PAR_CROP_PADDING', 1)) if isinstance(self.config, dict) else 1
                        min_col = max(0, min(cols) - pad)
                        min_row = max(0, min(rows) - pad)
                        max_col = min(w - 1, max(cols) + pad)
                        max_row = min(h - 1, max(rows) + pad)

                        # Slice and binarize obstacles: nonzero -> 1, zero -> 0
                        sliced = _np.array(self._ws_map_matrix, dtype=float)[min_row:max_row + 1, min_col:max_col + 1]
                        bin_grid = (sliced != 0).astype(int)

                        # Build SubMap from slice
                        sub_map = SubMap(bin_grid.tolist())
                        sub_map.height = bin_grid.shape[0]
                        sub_map.width = bin_grid.shape[1]
                        sub_map.origin_x = self._ws_offset_x + float(min_col) * float(self._ws_xy_reso)
                        sub_map.origin_y = self._ws_offset_y + float(min_row) * float(self._ws_xy_reso)
                        sub_map.resolution = float(self._ws_xy_reso)

                        self.sub_map = sub_map
                        self.grid_offset = (min_col, min_row)
                        # Also update region bounds for diagnostics/consumers
                        self.min_x = sub_map.origin_x
                        self.min_y = sub_map.origin_y
                        self.max_x = sub_map.origin_x + sub_map.width * sub_map.resolution
                        self.max_y = sub_map.origin_y + sub_map.height * sub_map.resolution
        else:
            # No map_matrix: build full grid from workspace bounds and rasterize analytic obstacles
            try:
                submap_mode = str(self.config.get('PAR_SUBMAP_MODE', 'full')).lower() if isinstance(self.config, dict) else 'full'
            except Exception:
                submap_mode = 'full'

            # Always use workspace bounds for full world dimensions
            bounds = None
            if hasattr(self.workspace, 'get') and callable(self.workspace.get):
                b = self.workspace.get('bounds', None)
                if isinstance(b, dict):
                    bounds = b
            
            # Force use of workspace bounds - no fallback to participant region
            if bounds is None:
                raise ValueError("Workspace bounds not found - cannot build PAR environment")

            bx0 = float(bounds.get('min_x', 0.0))
            by0 = float(bounds.get('min_y', 0.0))
            bx1 = float(bounds.get('max_x', bx0))
            by1 = float(bounds.get('max_y', by0))
            res = float(self._ws_xy_reso) if self._ws_xy_reso is not None else float(self.grid_resolution)
            import math as _math
            full_w = max(1, int(_math.floor((bx1 - bx0) / res)))
            full_h = max(1, int(_math.floor((by1 - by0) / res)))

            # Build full SubMap
            full_map = SubMap([[0 for _ in range(full_w)] for _ in range(full_h)])
            full_map.width = full_w
            full_map.height = full_h
            full_map.origin_x = bx0
            full_map.origin_y = by0
            full_map.resolution = res

            # Ensure obstacle rasterization uses this origin/resolution
            self.min_x = bx0
            self.min_y = by0
            self.max_x = bx1
            self.max_y = by1
            self.grid_resolution = res
            self._populate_obstacles(full_map)

            if submap_mode == 'full':
                self.sub_map = full_map
                self.grid_offset = (0, 0)
            else:
                # Crop around participants (global grid bbox)
                starts_global = {}
                goals_global = {}
                for agent_id in self.participants:
                    if agent_id not in agent_states:
                        continue
                    pos = self.get_agent_position(agent_states[agent_id])
                    goal = self.get_agent_goal(agent_states[agent_id])
                    if pos is not None:
                        cg = self.world_to_grid(pos[0], pos[1])
                        starts_global[agent_id] = cg
                    if goal is not None:
                        gg = self.world_to_grid(goal[0], goal[1])
                        goals_global[agent_id] = gg
                cols, rows = [], []
                for d in (starts_global, goals_global):
                    for _, (c, r) in d.items():
                        cols.append(int(c))
                        rows.append(int(r))
                if len(cols) == 0 or len(rows) == 0:
                    # No valid positions; use full
                    self.sub_map = full_map
                    self.grid_offset = (0, 0)
                else:
                    pad = int(self.config.get('PAR_CROP_PADDING', 1)) if isinstance(self.config, dict) else 1
                    min_col = max(0, min(cols) - pad)
                    min_row = max(0, min(rows) - pad)
                    max_col = min(full_w - 1, max(cols) + pad)
                    max_row = min(full_h - 1, max(rows) + pad)
                    # Slice
                    sliced = [row[min_col:max_col + 1] for row in full_map.grid[min_row:max_row + 1]]
                    sub_map = SubMap(sliced)
                    sub_map.height = len(sliced)
                    sub_map.width = len(sliced[0]) if sub_map.height > 0 else 0
                    sub_map.origin_x = bx0 + float(min_col) * res
                    sub_map.origin_y = by0 + float(min_row) * res
                    sub_map.resolution = res
                    self.sub_map = sub_map
                    self.grid_offset = (min_col, min_row)
            # Update region bounds for diagnostics/consumers
            self.min_x = self.sub_map.origin_x
            self.min_y = self.sub_map.origin_y
            self.max_x = self.sub_map.origin_x + self.sub_map.width * self.sub_map.resolution
            self.max_y = self.sub_map.origin_y + self.sub_map.height * self.sub_map.resolution
        # print(f"🔍 DEBUG: Sub-map built: {self.sub_map.width} x {self.sub_map.height}")
        
        # Build actor set
        self.actor_set = self.build_actor_set(agent_states)
        # print(f"🔍 DEBUG: Actor set built with {len(self.actor_set.actors)} actors")
        
        # print(f"🔍 DEBUG: build_par_environment completed")
        # Minimal diagnostics (printed only if config DEBUG_MODE is True)
        try:
            debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        except Exception:
            debug_mode = False
        if debug_mode and self.sub_map is not None:
            width = getattr(self.sub_map, 'width', 0)
            height = getattr(self.sub_map, 'height', 0)
            occ = 0
            if hasattr(self.sub_map, 'grid') and width > 0 and height > 0:
                occ = sum(1 for row in self.sub_map.grid for cell in row if cell)
            occupancy = (occ / float(width * height)) if width * height > 0 else 0.0
            print(f"PAR DIAG: bounds=({self.min_x:.3f},{self.min_y:.3f})-({self.max_x:.3f},{self.max_y:.3f}), grid={width}x{height}, res={self.grid_resolution}, occ={occupancy:.3f}")
            # Print per-actor grid start/goal
            try:
                starts = self.compute_start_positions(agent_states)
                goals = self.compute_goal_positions(agent_states)
                print(f"PAR DIAG: starts={starts}, goals={goals}")
            except Exception:
                pass
        return self.sub_map, self.actor_set
    
    def compute_region_boundaries(self, agent_states: Dict):
        """
        Compute the boundaries of the region containing all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
        """
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                
                # Consider both position and goal
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                
                if position is not None:
                    x, y = position
                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)
                
                if goal is not None:
                    x, y = goal
                    self.min_x = min(self.min_x, x)
                    self.max_x = max(self.max_x, x)
                    self.min_y = min(self.min_y, y)
                    self.max_y = max(self.max_y, y)
    
    def expand_region(self):
        """Expand the region boundaries by the specified offset."""
        self.min_x -= self.par_offset
        self.max_x += self.par_offset
        self.min_y -= self.par_offset
        self.max_y += self.par_offset
    
    def build_sub_map(self) -> SubMap:
        """
        Build the sub-map for the PAR region.
        
        Returns:
            SubMap: Built sub-map
        """
        # If we already built sub_map from map_matrix slicing, return it
        if self.sub_map is not None and hasattr(self.sub_map, 'grid') and len(getattr(self.sub_map, 'grid', [])) > 0:
            return self.sub_map
        # Legacy fallback: region-based dimensions
        width = int((self.max_x - self.min_x) / self.grid_resolution) + 1
        height = int((self.max_y - self.min_y) / self.grid_resolution) + 1
        sub_map = SubMap([[0 for _ in range(width)] for _ in range(height)])
        sub_map.width = width
        sub_map.height = height
        sub_map.origin_x = self.min_x
        sub_map.origin_y = self.min_y
        sub_map.resolution = self.grid_resolution
        self._populate_obstacles(sub_map)
        return sub_map
    
    def _populate_obstacles(self, sub_map: SubMap):
        """Populate the sub-map with obstacles from the environment."""
        try:
            # Lightweight debug toggle
            try:
                debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
            except Exception:
                debug_mode = False

            # Get obstacles from the workspace
            if hasattr(self.workspace, 'get') and callable(self.workspace.get):
                obstacles = self.workspace.get('obstacles', [])
                # If a rasterized map is provided, rasterize obstacles directly
                map_matrix = self.workspace.get('map_matrix', None)
                xy_reso = self.workspace.get('xy_reso', None)
                offset_x = self.workspace.get('offset_x', 0.0)
                offset_y = self.workspace.get('offset_y', 0.0)
            else:
                obstacles = getattr(self.workspace, 'obstacles', [])
                map_matrix = None
                xy_reso = None
                offset_x = 0.0
                offset_y = 0.0
            
            # Config switches to isolate issues during debugging
            use_analytic = bool(self.config.get('PAR_USE_ANALYTIC_OBSTACLES', True)) if isinstance(self.config, dict) else True
            use_map = bool(self.config.get('PAR_USE_MAP_MATRIX', True)) if isinstance(self.config, dict) else True
            disable_boundary_corrections = bool(self.config.get('PAR_DISABLE_BOUNDARY_CORRECTIONS', False)) if isinstance(self.config, dict) else False

            # print(f"🔍 PAR ENVIRONMENT: Found {len(obstacles)} obstacles in workspace")
            
            # If map_matrix is used to build submap, do not rasterize it again here
            if map_matrix is None and use_analytic and isinstance(obstacles, (list, tuple)) and len(obstacles) > 0:
                for obstacle in obstacles:
                    self._add_obstacle_to_grid(sub_map, obstacle)
            else:
                # No raster map available; analytic obstacles already processed above
                # Debug summary after population
                if debug_mode and hasattr(sub_map, 'grid'):
                    try:
                        total = sub_map.width * sub_map.height
                        occ_sum = sum(1 for row in sub_map.grid for cell in row if cell)
                        ratio = (occ_sum / float(total)) if total > 0 else 0.0
                        sample = []
                        for i in range(sub_map.height):
                            for j in range(sub_map.width):
                                if sub_map.grid[i][j]:
                                    sample.append((j, i))
                                    if len(sample) >= 5:
                                        break
                            if len(sample) >= 5:
                                break
                        print(f"PAR OCC LOG: bounds=({self.min_x:.3f},{self.min_y:.3f})-({self.max_x:.3f},{self.max_y:.3f}) grid={sub_map.width}x{sub_map.height} res={self.grid_resolution} occ_sum={occ_sum} ratio={ratio:.4f} samples={sample}")
                    except Exception:
                        pass
                
        except Exception as e:
            # print(f"⚠️ Warning: Could not populate obstacles: {e}")
            # Continue with free space if obstacle processing fails
            pass
    
    def _add_obstacle_to_grid(self, sub_map: SubMap, obstacle):
        """Add a single obstacle to the grid."""
        try:
            if hasattr(obstacle, 'pos') and hasattr(obstacle, 'radius'):
                # Circular obstacle
                center_x, center_y = obstacle.pos[0], obstacle.pos[1]
                radius = obstacle.radius
                self._add_circular_obstacle(sub_map, center_x, center_y, radius)
                
            elif hasattr(obstacle, 'vertices'):
                # Polygon obstacle
                vertices = obstacle.vertices
                self._add_polygon_obstacle(sub_map, vertices)
            elif hasattr(obstacle, 'vertexes'):
                # Polygon obstacle (ir_sim obs_polygon uses 'vertexes' 2xN)
                try:
                    verts = obstacle.vertexes
                    # Expect ndarray shape (2, N)
                    if hasattr(verts, 'shape') and len(verts.shape) == 2 and verts.shape[0] == 2:
                        vertices = [(float(verts[0, i]), float(verts[1, i])) for i in range(verts.shape[1])]
                    else:
                        # Fallback: attempt to iterate columns
                        vertices = [(float(v[0]), float(v[1])) for v in getattr(obstacle, 'vertexes')]
                    self._add_polygon_obstacle(sub_map, vertices)
                except Exception:
                    pass
                
            elif isinstance(obstacle, (list, tuple)) and len(obstacle) >= 2:
                # Point obstacle
                x, y = obstacle[0], obstacle[1]
                self._add_point_obstacle(sub_map, x, y)
                
        except Exception as e:
            # print(f"⚠️ Warning: Could not process obstacle {obstacle}: {e}")
            pass
    
    def _add_circular_obstacle(self, sub_map: SubMap, center_x: float, center_y: float, radius: float):
        """Add a circular obstacle to the grid."""
        # Convert to grid coordinates
        grid_center_x = int((center_x - self.min_x) / self.grid_resolution)
        grid_center_y = int((center_y - self.min_y) / self.grid_resolution)
        grid_radius = int(radius / self.grid_resolution) + 1
        
        # Mark grid cells within the circle as obstacles
        for i in range(max(0, grid_center_y - grid_radius), min(sub_map.height, grid_center_y + grid_radius + 1)):
            for j in range(max(0, grid_center_x - grid_radius), min(sub_map.width, grid_center_x + grid_radius + 1)):
                # Check if cell is within circle
                if (i - grid_center_y) ** 2 + (j - grid_center_x) ** 2 <= grid_radius ** 2:
                    if 0 <= i < sub_map.height and 0 <= j < sub_map.width:
                        sub_map.grid[i][j] = 1  # Mark as obstacle
    
    def _add_polygon_obstacle(self, sub_map: SubMap, vertices):
        """Add a polygon obstacle to the grid using grid cell overlap method."""
        debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        
        if debug_mode:
            print(f"PAR POLYGON DEBUG: Processing polygon with {len(vertices)} vertices")
            print(f"PAR POLYGON DEBUG: Original vertices: {vertices}")
            print(f"PAR POLYGON DEBUG: Grid bounds: min_x={self.min_x}, min_y={self.min_y}, resolution={self.grid_resolution}")
        
        if len(vertices) < 3:
            if debug_mode:
                print(f"PAR POLYGON DEBUG: Skipping polygon with < 3 vertices")
            return
        
        # Use grid cell overlap method
        self._fill_polygon_grid_cell_overlap(sub_map, vertices)
    
    def _add_point_obstacle(self, sub_map: SubMap, x: float, y: float):
        """Add a point obstacle to the grid."""
        grid_x = int((x - self.min_x) / self.grid_resolution)
        grid_y = int((y - self.min_y) / self.grid_resolution)
        
        if 0 <= grid_y < sub_map.height and 0 <= grid_x < sub_map.width:
            sub_map.grid[grid_y][grid_x] = 1  # Mark as obstacle
    
    def _fill_polygon_grid_cell_overlap(self, sub_map: SubMap, vertices):
        """Fill polygon area using grid cell overlap method."""
        debug_mode = bool(self.config.get('DEBUG_MODE', False)) if isinstance(self.config, dict) else False
        
        # Calculate polygon bounding box in continuous coordinates
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        
        if debug_mode:
            print(f"PAR FILL DEBUG: Polygon bounds: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")
        
        # Calculate grid cell range that might overlap with polygon
        grid_min_x = max(0, int((min_x - self.min_x) / self.grid_resolution))
        grid_max_x = min(sub_map.width, int((max_x - self.min_x) / self.grid_resolution) + 1)
        grid_min_y = max(0, int((min_y - self.min_y) / self.grid_resolution))
        grid_max_y = min(sub_map.height, int((max_y - self.min_y) / self.grid_resolution) + 1)
        
        if debug_mode:
            print(f"PAR FILL DEBUG: Grid search range: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
        
        filled_count = 0
        for i in range(grid_min_y, grid_max_y):
            for j in range(grid_min_x, grid_max_x):
                # Calculate grid cell corners in continuous coordinates
                cell_x = self.min_x + j * self.grid_resolution
                cell_y = self.min_y + i * self.grid_resolution
                cell_corners = [
                    (cell_x, cell_y),                           # bottom-left
                    (cell_x + self.grid_resolution, cell_y),    # bottom-right
                    (cell_x, cell_y + self.grid_resolution),    # top-left
                    (cell_x + self.grid_resolution, cell_y + self.grid_resolution)  # top-right
                ]
                
                # Check if grid cell overlaps with polygon (excluding boundary-only overlap)
                if self._grid_cell_overlaps_polygon(cell_corners, vertices):
                    sub_map.grid[i][j] = 1  # Mark as obstacle
                    filled_count += 1
                    if debug_mode and i <= 2:  # Debug first few rows
                        print(f"PAR FILL DEBUG: Filled cell ({j}, {i}) at continuous ({cell_x:.3f}, {cell_y:.3f})")
        
        if debug_mode:
            print(f"PAR FILL DEBUG: Total cells filled: {filled_count}")
    
    def _grid_cell_overlaps_polygon(self, cell_corners, vertices):
        """Check if a grid cell overlaps with polygon (excluding boundary-only overlap)."""
        # Check if any corner is inside the polygon
        corners_inside = 0
        corners_on_boundary = 0
        
        for corner in cell_corners:
            inside, on_boundary = self._point_in_polygon_with_boundary(corner[0], corner[1], vertices)
            if inside and not on_boundary:
                corners_inside += 1
            elif on_boundary:
                corners_on_boundary += 1
        
        # If any corner is inside (not on boundary), the cell overlaps
        if corners_inside > 0:
            return True
        
        # If all corners are on boundary, the cell doesn't overlap
        if corners_inside == 0 and corners_on_boundary == 4:
            return False
        
        # If some corners are on boundary and some are outside, check if polygon intersects cell interior
        # According to user requirement: boundary-only overlap should NOT mark the cell as obstacle
        # Only mark as obstacle if there are corners inside the polygon (not on boundary)
        return corners_inside > 0
    
    def _point_in_polygon_with_boundary(self, x: float, y: float, vertices):
        """Check if a point is inside a polygon, with boundary detection."""
        n = len(vertices)
        inside = False
        on_boundary = False
        
        # Check if point is exactly on a vertex
        for vx, vy in vertices:
            if abs(x - vx) < 1e-10 and abs(y - vy) < 1e-10:
                return True, True  # on boundary
        
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            
            # Check if point is on the edge
            if self._point_on_line_segment(x, y, p1x, p1y, p2x, p2y):
                return True, True  # on boundary
            
            # Ray casting for inside/outside determination
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = max(p1x, p2x)
                        
                        if abs(x - xinters) < 1e-10:  # Point is on the ray
                            on_boundary = True
                        elif x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside, on_boundary
    
    def _point_on_line_segment(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float):
        """Check if point (px, py) is on line segment from (x1, y1) to (x2, y2)."""
        # Check if point is within the bounding box of the line segment
        if not (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)):
            return False
        
        # Check if point is on the line (using cross product)
        cross_product = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        return abs(cross_product) < 1e-10
    
    def _point_in_polygon(self, x: int, y: int, vertices) -> bool:
        """Check if a point is inside a polygon using improved ray casting."""
        n = len(vertices)
        inside = False
        
        # Handle edge case: if point is exactly on a vertex, consider it inside
        for vx, vy in vertices:
            if x == vx and y == vy:
                return True
        
        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            
            # Improved boundary handling for ray casting
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            # Handle horizontal edges
                            xinters = max(p1x, p2x)
                        
                        # Special handling for boundary points
                        if p1x == p2x:  # Vertical edge
                            if x == p1x:  # Point is on vertical edge
                                return True
                        elif p1y == p2y:  # Horizontal edge
                            if y == p1y and min(p1x, p2x) <= x <= max(p1x, p2x):  # Point is on horizontal edge
                                return True
                        
                        if x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def build_actor_set(self, agent_states: Dict) -> ActorSet:
        """
        Build the actor set for participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            ActorSet: Built actor set
        """
        # print(f"🔍 DEBUG: build_actor_set called with {len(agent_states)} agents")
        # print(f"🔍 DEBUG: Participants: {self.participants}")
        
        # Ensure boundaries are computed and expanded before building actors
        if self.min_x == float('inf'):
            # print(f"🔍 DEBUG: Bounds not computed yet in build_actor_set, calling compute_region_boundaries and expand_region")
            self.compute_region_boundaries(agent_states)
            self.expand_region()
            # print(f"🔍 DEBUG: Updated bounds in build_actor_set - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        actor_set = ActorSet()
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                if position is None:
                    continue
                # Map to global grid then to local by subtracting grid_offset
                g_start = self.world_to_grid(position[0], position[1])
                l_start = (g_start[0] - self.grid_offset[0], g_start[1] - self.grid_offset[1])
                from python_pnr.node import Point
                start_point = Point(int(l_start[0]), int(l_start[1]))
                if goal is not None:
                    g_goal = self.world_to_grid(goal[0], goal[1])
                    l_goal = (g_goal[0] - self.grid_offset[0], g_goal[1] - self.grid_offset[1])
                    # Clamp to sub_map bounds if available
                    gx = int(max(0, min(l_goal[0], self.sub_map.width - 1))) if self.sub_map else int(l_goal[0])
                    gy = int(max(0, min(l_goal[1], self.sub_map.height - 1))) if self.sub_map else int(l_goal[1])
                    goal_point = Point(gx, gy)
                else:
                    goal_point = Point(int(l_start[0]), int(l_start[1]))
                    actor = Actor(agent_id, start_point, goal_point)
                    actor_set.add_actor(actor)
        
        # print(f"🔍 DEBUG: build_actor_set completed with {len(actor_set.actors)} actors")
        return actor_set
    
    def compute_start_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute start positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to start positions
        """
        # print(f"🔍 DEBUG: compute_start_positions called with {len(agent_states)} agents")
        
        # Use world->global grid->local mapping when map_matrix exists
        self._read_workspace_params()
        has_map = (self._ws_map_matrix is not None and self._ws_xy_reso is not None)
        
        # print(f"🔍 DEBUG: Current bounds - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        start_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                if position is not None:
                    if has_map:
                        g = self.world_to_grid(position[0], position[1])
                        start_positions[agent_id] = (int(g[0] - self.grid_offset[0]), int(g[1] - self.grid_offset[1]))
                    else:
                        x, y = position
                        grid_x = int((x - self.min_x) / self.grid_resolution)
                        grid_y = int((y - self.min_y) / self.grid_resolution)
                        start_positions[agent_id] = (grid_x, grid_y)
        
        # print(f"🔍 DEBUG: Final start_positions: {start_positions}")
        return start_positions
    
    def compute_goal_positions(self, agent_states: Dict) -> Dict[int, Tuple[int, int]]:
        """
        Compute goal positions for all participating agents.
        
        Args:
            agent_states: Dictionary of all agent states
            
        Returns:
            Dict[int, Tuple[int, int]]: Dictionary mapping agent IDs to goal positions
        """
        # print(f"🔍 DEBUG: compute_goal_positions called with {len(agent_states)} agents")
        
        # Use world->global grid->local mapping when map_matrix exists
        self._read_workspace_params()
        has_map = (self._ws_map_matrix is not None and self._ws_xy_reso is not None)
        
        # print(f"🔍 DEBUG: Current bounds - min_x: {self.min_x}, max_x: {self.max_x}, min_y: {self.min_y}, max_y: {self.max_y}")
        
        goal_positions = {}
        
        for agent_id in self.participants:
            if agent_id in agent_states:
                agent_state = agent_states[agent_id]
                position = self.get_agent_position(agent_state)
                goal = self.get_agent_goal(agent_state)
                if position is not None and goal is not None:
                    pos_x, pos_y = position
                    goal_x, goal_y = goal
                    continuous_distance = np.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
                    if continuous_distance < self.grid_resolution/2:
                        continue
                if goal is not None:
                    if has_map:
                        g = self.world_to_grid(goal[0], goal[1])
                        gx = int(g[0] - self.grid_offset[0])
                        gy = int(g[1] - self.grid_offset[1])
                        if self.sub_map:
                            gx = max(0, min(gx, self.sub_map.width - 1))
                            gy = max(0, min(gy, self.sub_map.height - 1))
                        goal_positions[agent_id] = (gx, gy)
                    else:
                        x, y = goal
                        grid_x = int((x - self.min_x) / self.grid_resolution)
                        grid_y = int((y - self.min_y) / self.grid_resolution)
                        grid_x = max(0, min(grid_x, self.sub_map.width - 1)) if self.sub_map else grid_x
                        grid_y = max(0, min(grid_y, self.sub_map.height - 1)) if self.sub_map else grid_y
                        goal_positions[agent_id] = (grid_x, grid_y)
        
        return goal_positions
    
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
    
    def get_agent_goal(self, agent_state: Dict) -> Optional[Tuple[float, float]]:
        """
        Extract agent goal from agent state.
        
        Args:
            agent_state: Agent state dictionary
            
        Returns:
            Optional[Tuple[float, float]]: Agent goal (x, y) or None if not available
        """
        if 'goal' in agent_state:
            goal = agent_state['goal']
            if isinstance(goal, (list, np.ndarray)) and len(goal) >= 2:
                return (float(goal[0]), float(goal[1]))
        
        # Try alternative goal fields
        for field in ['target', 'destination', 'end_pos']:
            if field in agent_state:
                goal_data = agent_state[field]
                if isinstance(goal_data, (list, np.ndarray)) and len(goal_data) >= 2:
                    return (float(goal_data[0]), float(goal_data[1]))
        
        return None
    
    def grid_to_continuous(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert grid coordinates to continuous coordinates.
        
        Args:
            grid_pos: Grid position (x, y)
            
        Returns:
            Tuple[float, float]: Continuous position (x, y)
        """
        grid_x, grid_y = grid_pos
        # Prefer precise origin/resolution from sub_map if available
        if self.sub_map is not None and hasattr(self.sub_map, 'origin_x'):
            x = float(self.sub_map.origin_x) + (int(grid_x) + 0.5) * float(self.sub_map.resolution)
            y = float(self.sub_map.origin_y) + (int(grid_y) + 0.5) * float(self.sub_map.resolution)
        return (x, y)
        # Fallback to world mapping helpers
        return self.grid_to_world(int(grid_x) + self.grid_offset[0], int(grid_y) + self.grid_offset[1])
    
    def continuous_to_grid(self, continuous_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert continuous coordinates to grid coordinates.
        
        Args:
            continuous_pos: Continuous position (x, y)
            
        Returns:
            Tuple[int, int]: Grid position (x, y)
        """
        x, y = continuous_pos
        if self.sub_map is not None and hasattr(self.sub_map, 'origin_x'):
            gx = int((float(x) - float(self.sub_map.origin_x)) / float(self.sub_map.resolution))
            gy = int((float(y) - float(self.sub_map.origin_y)) / float(self.sub_map.resolution))
            return (gx, gy)
        # Fallback to global mapping then subtract grid_offset to get local
        gcol, grow = self.world_to_grid(float(x), float(y))
        return (int(gcol - self.grid_offset[0]), int(grow - self.grid_offset[1]))
    
    def is_position_in_bounds(self, position: Tuple[float, float]) -> bool:
        """
        Check if a position is within the PAR region bounds.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            bool: True if position is within bounds
        """
        x, y = position
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y)
    
    def get_region_info(self) -> Dict:
        """
        Get information about the PAR region.
        
        Returns:
            Dict: Region information including boundaries and dimensions
        """
        return {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y,
            'width': self.max_x - self.min_x,
            'height': self.max_y - self.min_y,
            'participants': self.participants,
            'grid_resolution': self.grid_resolution
        }
