import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from datetime import datetime
import gym
import gym_env


class TestVisualizer:
    """
    Visualizer for test case execution with agent positions and modes
    """
    
    def __init__(self, test_type="test", base_vis_dir="/home/haoyiwang/Desktop/RL_RVO/vis"):
        """
        Initialize the test visualizer
        
        Args:
            test_type: Type of test ("test" or "test_with_deadlock")
            base_vis_dir: Base directory for visualizations
        """
        self.test_type = test_type
        self.base_vis_dir = Path(base_vis_dir)
        self.base_vis_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_vis_dir / f"{timestamp}_{test_type}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Visualization settings
        self.agent_radius = 0.2
        self.obstacle_color = 'black'
        self.rl_agent_color = 'blue'
        self.par_agent_color = 'red'
        self.goal_color = 'green'
        self.frame_duration = 200  # milliseconds (0.2 seconds)
        
        # Agent colors for waypoint visualization (avoiding system colors: blue=RL, red=PAR, green=start/goal)
        self.agent_colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'darkorange']
        
        # Waypoint visualization settings
        self.waypoint_radius = 0.15
        self.waypoint_line_width = 2.0
        self.waypoint_alpha = 0.7
        
        # Environment reference for map information
        self.env = None
        self.map_bounds = None
        self.obstacles = []
        
        print(f"Test visualizer initialized. Session directory: {self.session_dir}")
    
    def set_environment(self, env):
        """
        Set the environment reference for map information
        
        Args:
            env: Gym environment instance
        """
        self.env = env
        self._extract_map_bounds()
    
    def _extract_map_bounds(self):
        """Extract map bounds from environment"""
        if self.env is None:
            return
        
        try:
            # Prepare fallback from world_plot if available (avoids hardcoded 10x10)
            fallback_bounds = None
            try:
                if hasattr(self.env, 'ir_gym') and hasattr(self.env.ir_gym, 'world_plot'):
                    wp = self.env.ir_gym.world_plot
                    width = float(getattr(wp, 'width', 0.0)) if hasattr(wp, 'width') else 0.0
                    height = float(getattr(wp, 'height', 0.0)) if hasattr(wp, 'height') else 0.0
                    offset_x = float(getattr(wp, 'offset_x', 0.0)) if hasattr(wp, 'offset_x') else float(getattr(self.env.ir_gym, 'offset_x', 0.0))
                    offset_y = float(getattr(wp, 'offset_y', 0.0)) if hasattr(wp, 'offset_y') else float(getattr(self.env.ir_gym, 'offset_y', 0.0))
                    if width > 0.0 and height > 0.0:
                        fallback_bounds = [offset_x, offset_y, offset_x + width, offset_y + height]
            except Exception:
                fallback_bounds = None

            # Get map bounds from environment
            if hasattr(self.env.ir_gym, 'components'):
                # Extract workspace bounds
                if 'workspace' in self.env.ir_gym.components:
                    workspace = self.env.ir_gym.components['workspace']
                    if hasattr(workspace, 'square'):
                        self.map_bounds = workspace.square  # [x_min, y_min, x_max, y_max]
                    elif hasattr(workspace, 'circular'):
                        circular = workspace.circular
                        center_x, center_y, radius = circular
                        self.map_bounds = [center_x - radius, center_y - radius, 
                                         center_x + radius, center_y + radius]
                    else:
                        self.map_bounds = fallback_bounds if fallback_bounds else [0, 0, 10, 10]
                else:
                    self.map_bounds = fallback_bounds if fallback_bounds else [0, 0, 10, 10]
            else:
                self.map_bounds = fallback_bounds if fallback_bounds else [0, 0, 10, 10]

        except Exception as e:
            print(f"Warning: Could not extract map bounds: {e}")
            # Set default bounds
            self.map_bounds = [0, 0, 10, 10]

    def _get_current_obstacles(self):
        """
        Extract current obstacle information from environment.
        This should be called per episode to get dynamically generated obstacles.
        """
        obstacles = []
        if self.env is None:
            return obstacles

        try:
            if hasattr(self.env.ir_gym, 'components'):
                # Circular obstacles
                if 'obs_circles' in self.env.ir_gym.components:
                    obs_circles = self.env.ir_gym.components['obs_circles']
                    if hasattr(obs_circles, 'obs_cir_list'):
                        for obs in obs_circles.obs_cir_list:
                            if hasattr(obs, 'state') and hasattr(obs, 'radius'):
                                pos = obs.state[0:2].flatten()
                                obstacles.append({
                                    'type': 'circle',
                                    'position': pos,
                                    'radius': obs.radius
                                })
                
                # Line obstacles
                if 'obs_lines' in self.env.ir_gym.components:
                    obs_lines = self.env.ir_gym.components['obs_lines']
                    if hasattr(obs_lines, 'obs_line_list'):
                        for obs in obs_lines.obs_line_list:
                            if hasattr(obs, 'line'):
                                obstacles.append({
                                    'type': 'line',
                                    'line': obs.line
                                })
                
                # Polygon obstacles (static and random)
                if 'obs_polygons' in self.env.ir_gym.components:
                    obs_polygons = self.env.ir_gym.components['obs_polygons']
                    if hasattr(obs_polygons, 'obs_poly_list'):
                        for obs in obs_polygons.obs_poly_list:
                            # Check for both 'vertices' and 'vertexes' attributes
                            if hasattr(obs, 'vertices'):
                                obstacles.append({
                                    'type': 'polygon',
                                    'vertices': obs.vertices
                                })
                            elif hasattr(obs, 'vertexes'):
                                # Convert vertexes (2xN array) to vertices (list of [x,y] pairs)
                                vertexes = obs.vertexes
                                if vertexes is not None and vertexes.shape[0] == 2:
                                    vertices = [[vertexes[0, i], vertexes[1, i]] for i in range(vertexes.shape[1])]
                                    obstacles.append({
                                        'type': 'polygon',
                                        'vertices': vertices
                                    })
                
                # Random polygon obstacles (Mode 7) - This is the key for dynamic obstacles
                if 'obs_polygons_random' in self.env.ir_gym.components:
                    obs_polygons_random = self.env.ir_gym.components['obs_polygons_random']
                    if hasattr(obs_polygons_random, 'obs_poly_list'):
                        for obs in obs_polygons_random.obs_poly_list:
                            # Check for both 'vertices' and 'vertexes' attributes
                            if hasattr(obs, 'vertices'):
                                obstacles.append({
                                    'type': 'polygon',
                                    'vertices': obs.vertices
                                })
                            elif hasattr(obs, 'vertexes'):
                                # Convert vertexes (2xN array) to vertices (list of [x,y] pairs)
                                vertexes = obs.vertexes
                                if vertexes is not None and vertexes.shape[0] == 2:
                                    vertices = [[vertexes[0, i], vertexes[1, i]] for i in range(vertexes.shape[1])]
                                    obstacles.append({
                                        'type': 'polygon',
                                        'vertices': vertices
                                    })
                
                # Alternative: try to get obstacles from total_states
                if not obstacles:
                    try:
                        ts = self.env.ir_gym.components['robots'].total_states()
                        if len(ts) >= 4:
                            # ts[2] should be obs_circular_list, ts[3] should be obs_line_list
                            obs_cir_list = ts[2] if len(ts) > 2 else []
                            obs_line_list = ts[3] if len(ts) > 3 else []
                            
                            # Process circular obstacles
                            for obs in obs_cir_list:
                                if hasattr(obs, 'state') and hasattr(obs, 'radius'):
                                    pos = obs.state[0:2].flatten()
                                    obstacles.append({
                                        'type': 'circle',
                                        'position': pos,
                                        'radius': obs.radius
                                    })
                            
                            # Process line obstacles
                            for obs in obs_line_list:
                                if hasattr(obs, 'line'):
                                    obstacles.append({
                                        'type': 'line',
                                        'line': obs.line
                                    })
                    except Exception as e:
                        print(f"Warning: Could not extract obstacles from total_states: {e}")

        except Exception as e:
            print(f"Warning: Could not extract current obstacles: {e}")
        
        return obstacles
    
    def create_episode_visualization(self, episode_data, episode_id):
        """
        Create visualization for a single episode
        
        Args:
            episode_data: Episode data from logs
            episode_id: Episode ID
        """
        if not episode_data or 'steps' not in episode_data:
            print(f"Warning: No valid episode data for episode {episode_id}")
            return
        
        steps = episode_data['steps']
        robot_number = episode_data.get('robot_number', len(steps[0]['robot_positions']) if steps else 0)
        
        # Get obstacles for this episode
        self.obstacles = self._get_current_obstacles()

        # Debug: print obstacle information
        # print(f"Extracted {len(self.obstacles)} obstacles for episode {episode_id}:")
        # for i, obs in enumerate(self.obstacles):
        #     print(f"  Obstacle {i}: {obs['type']}")
        #     if obs['type'] == 'circle':
        #         print(f"    Position: {obs['position']}, Radius: {obs['radius']}")
        #     elif obs['type'] == 'line':
        #         print(f"    Line: {obs['line']}")
        #     elif obs['type'] == 'polygon':
        #         print(f"    Vertices: {len(obs['vertices'])} points")
        
        # Set up the figure and axis with extra space for legend
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(right=0.75)  # Leave space for legend on the right
        
        # Set map bounds without extra margins (edges coincide with bounds)
        if self.map_bounds:
            ax.set_xlim(self.map_bounds[0], self.map_bounds[2])
            ax.set_ylim(self.map_bounds[1], self.map_bounds[3])
        else:
            # Auto-scale based on agent positions (no margins)
            all_positions = []
            for step in steps:
                all_positions.extend(step['robot_positions'])
            if all_positions:
                positions = np.array(all_positions)
                ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
                ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
        
        # Get start and goal positions from episode data
        start_positions = episode_data.get('start_positions', [])
        goal_positions = episode_data.get('goal_positions', [])
        
        # Get waypoint data for long-range navigation
        waypoint_data = episode_data.get('waypoint_data', {})
        
        # If goal_positions are missing, fall back to final_goal from waypoint_data
        if (not goal_positions) and waypoint_data:
            try:
                goal_positions = [v.get('final_goal') for v in waypoint_data.values() if isinstance(v, dict) and 'final_goal' in v]
                goal_positions = [g for g in goal_positions if g is not None]
            except Exception:
                pass
        
        # Draw static map elements (obstacles, start positions, goal positions)
        self._draw_map_elements(ax, start_positions, goal_positions)
        
        # Draw waypoint paths if available (for long-range navigation)
        if waypoint_data:
            self._draw_waypoints(ax, waypoint_data)
        
        # Initialize agent circles
        agent_circles = []
        agent_texts = []
        for i in range(robot_number):
            circle = patches.Circle((0, 0), self.agent_radius, 
                                  color=self.rl_agent_color, alpha=0.8)
            ax.add_patch(circle)
            agent_circles.append(circle)
            
            # Add agent ID text
            text = ax.text(0, 0, str(i), ha='center', va='center', 
                          fontsize=8, fontweight='bold', color='white')
            agent_texts.append(text)
        
        # Add step counter
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.rl_agent_color, label='RL Agent'),
            patches.Patch(color=self.par_agent_color, label='PAR Agent'),
            patches.Patch(color=self.obstacle_color, label='Obstacle'),
            patches.Patch(color='green', label='Start Position'),
            patches.Patch(color='green', label='Goal Position')
        ]
        
        # Add waypoint legend if waypoint data exists
        if waypoint_data:
            legend_elements.extend([
                patches.Patch(color='gray', label='Waypoint Path'),
                patches.Patch(color='gray', label='Waypoint (S=Start, E=End)')
            ])
        
        # Place legend outside the plot area to avoid blocking map content
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Episode {episode_id} - {self.test_type.replace("_", " ").title()}')
        
        def animate(frame):
            if frame >= len(steps):
                return
            
            step = steps[frame]
            positions = step['robot_positions']
            agent_modes = step.get('additional_info', {}).get('agent_modes', 
                                                           ['rl_rvo'] * len(positions))
            
            # Update agent positions and colors
            for i, (pos, mode) in enumerate(zip(positions, agent_modes)):
                if i < len(agent_circles):
                    agent_circles[i].center = (pos[0], pos[1])
                    agent_texts[i].set_position((pos[0], pos[1]))
                    
                    # Set color based on mode
                    if mode == 'par':
                        agent_circles[i].set_color(self.par_agent_color)
                    else:
                        agent_circles[i].set_color(self.rl_agent_color)
            
            # Update step counter
            step_text.set_text(f'Step: {frame + 1}/{len(steps)}')
            
            return agent_circles + agent_texts + [step_text]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(steps), 
                           interval=self.frame_duration, blit=False, repeat=True)
        
        # Save as GIF
        gif_path = self.session_dir / f"episode_{episode_id:03d}.gif"
        writer = PillowWriter(fps=1000/self.frame_duration)
        anim.save(gif_path, writer=writer)
        
        plt.close(fig)
        # print(f"Episode {episode_id} visualization saved to {gif_path}")
    
    def _draw_map_elements(self, ax, start_positions=None, goal_positions=None):
        """Draw static map elements (obstacles, start positions, goal positions)"""
        # Draw obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                circle = patches.Circle(obstacle['position'], obstacle['radius'],
                                      color=self.obstacle_color, alpha=0.8)
                ax.add_patch(circle)
            
            elif obstacle['type'] == 'line':
                line = obstacle['line']
                ax.plot([line[0], line[2]], [line[1], line[3]], 
                       color=self.obstacle_color, linewidth=3)
            
            elif obstacle['type'] == 'polygon':
                vertices = obstacle['vertices']
                if len(vertices) >= 3:
                    polygon = patches.Polygon(vertices, color=self.obstacle_color, alpha=0.8)
                    ax.add_patch(polygon)
        
        # Draw start positions (green squares)
        if start_positions:
            for i, (x, y) in enumerate(start_positions):
                square = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2,
                                         color='green', alpha=0.8, linewidth=2)
                ax.add_patch(square)
                # Add start label
                ax.text(x, y - 0.3, f'S{i}', ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='green')
        
        # Draw goal positions (green stars)
        if goal_positions:
            for i, (x, y) in enumerate(goal_positions):
                # Create a 5-pointed star
                star = patches.RegularPolygon((x, y), 5, radius=0.15,
                                            orientation=0, color='green', alpha=0.8)
                ax.add_patch(star)
                # Add goal label
                ax.text(x, y + 0.3, f'G{i}', ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='green')
    
    def _draw_waypoints(self, ax, waypoint_data):
        """Draw waypoint paths for long-range navigation"""
        if not waypoint_data:
            return
        
        for agent_id_str, agent_data in waypoint_data.items():
            try:
                agent_id = int(agent_id_str)
                color = self.agent_colors[agent_id % len(self.agent_colors)]
                
                waypoints = agent_data.get('waypoints', [])
                if len(waypoints) < 1:
                    continue
                
                if len(waypoints) == 1:
                    # Single waypoint: draw as end point (E)
                    x, y = waypoints[0][0], waypoints[0][1]
                    star = patches.RegularPolygon((x, y), 5, radius=self.waypoint_radius * 1.2,
                                                orientation=0, color=color, alpha=0.9, zorder=2)
                    ax.add_patch(star)
                    ax.text(x, y, 'E', ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white', zorder=3)
                else:
                    # Draw waypoint lines
                    waypoint_x = [wp[0] for wp in waypoints]
                    waypoint_y = [wp[1] for wp in waypoints]
                    
                    # Draw connecting lines between waypoints
                    ax.plot(waypoint_x, waypoint_y, color=color, linewidth=self.waypoint_line_width,
                           alpha=self.waypoint_alpha, linestyle='-', zorder=1)
                    
                    # Draw waypoint points
                    for i, (x, y) in enumerate(waypoints):
                        if i == 0:  # Start waypoint
                            circle = patches.Circle((x, y), self.waypoint_radius * 1.2, 
                                                  color=color, alpha=0.9, zorder=2)
                            ax.add_patch(circle)
                            ax.text(x, y, 'S', ha='center', va='center', 
                                   fontsize=8, fontweight='bold', color='white', zorder=3)
                        elif i == len(waypoints) - 1:  # End waypoint
                            star = patches.RegularPolygon((x, y), 5, radius=self.waypoint_radius * 1.2,
                                                        orientation=0, color=color, alpha=0.9, zorder=2)
                            ax.add_patch(star)
                            ax.text(x, y, 'E', ha='center', va='center', 
                                   fontsize=8, fontweight='bold', color='white', zorder=3)
                        else:  # Intermediate waypoints
                            circle = patches.Circle((x, y), self.waypoint_radius, 
                                                  color=color, alpha=0.8, zorder=2)
                            ax.add_patch(circle)
                            ax.text(x, y, str(i), ha='center', va='center', 
                                   fontsize=6, fontweight='bold', color='white', zorder=3)
                    
                    # Add direction arrows on the path
                    self._draw_path_arrows(ax, waypoints, color)
                
            except (ValueError, KeyError, IndexError) as e:
                print(f"Warning: Could not draw waypoints for agent {agent_id_str}: {e}")
                continue
    
    def _draw_path_arrows(self, ax, waypoints, color):
        """Draw arrows on the path to show direction"""
        if len(waypoints) < 2:
            return
        
        # Add arrows at regular intervals along the path
        arrow_interval = max(1, len(waypoints) // 4)  # Add 3-4 arrows along the path
        
        for i in range(arrow_interval, len(waypoints), arrow_interval):
            if i < len(waypoints):
                # Calculate direction vector
                if i > 0:
                    dx = waypoints[i][0] - waypoints[i-1][0]
                    dy = waypoints[i][1] - waypoints[i-1][1]
                    
                    # Normalize direction vector
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx /= length
                        dy /= length
                        
                        # Draw arrow
                        ax.annotate('', xy=(waypoints[i][0], waypoints[i][1]),
                                  xytext=(waypoints[i][0] - dx*0.3, waypoints[i][1] - dy*0.3),
                                  arrowprops=dict(arrowstyle='->', color=color, 
                                                lw=1.5, alpha=0.8), zorder=2)
    
    def create_session_visualizations(self, logs_dir):
        """
        Create visualizations for all episodes in a session
        
        Args:
            logs_dir: Path to the logs directory containing episode files
        """
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            print(f"Error: Logs directory {logs_dir} does not exist")
            return
        
        # Find all episode files
        episode_files = sorted(logs_path.glob("episode_*.json"))
        
        if not episode_files:
            print(f"No episode files found in {logs_dir}")
            return
        
        print(f"Found {len(episode_files)} episode files")
        
        # Create environment for map information
        if self.env is None:
            self._create_environment()
        
        # Process each episode
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    episode_data = json.load(f)
                
                episode_id = episode_data.get('episode_id', 0)
                self.create_episode_visualization(episode_data, episode_id)
                
            except Exception as e:
                print(f"Error processing {episode_file}: {e}")
        
        print(f"Visualization session completed. Files saved to {self.session_dir}")
    
    def _create_environment(self):
        """Create environment instance for map information"""
        try:
            # Create a minimal environment to get map information
            # This is a simplified approach - in practice, you might want to pass the actual env
            self.env = gym.make('mrnav-v1', world_name='mode7_stage3_complex.yaml', 
                              robot_number=8, env_train=False)
            self._extract_map_info()
        except Exception as e:
            print(f"Warning: Could not create environment for map info: {e}")
            self.map_bounds = [0, 0, 10, 10]  # Default bounds
