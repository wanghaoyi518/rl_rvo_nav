class LongRangeConfig:
    """Centralized configuration for long-range waypoint navigation.

    This class holds default parameters for global planning, waypoint management,
    scenario scale, and testing options. Values can be overridden by caller code
    via kwargs or by constructing an alternative config object.
    """

    # Global path planning parameters
    grid_resolution: float = 0.5
    waypoint_min_spacing: float = 5.0
    path_simplification_epsilon: float = 0.5
    obstacle_inflation_radius: float = 0.5

    # Waypoint manager parameters
    reach_threshold: float = 0.2
    goal_switch_smoothing: bool = True

    # Scenario parameters (meters)
    map_size = (100.0, 100.0)
    num_static_obstacles: int = 50
    obstacle_size_range = (1.0, 5.0)

    # Test parameters
    max_episode_steps: int = 5000
    render_waypoints: bool = True
    save_trajectories: bool = True

    def as_dict(self):
        return {
            "grid_resolution": self.grid_resolution,
            "waypoint_min_spacing": self.waypoint_min_spacing,
            "path_simplification_epsilon": self.path_simplification_epsilon,
            "obstacle_inflation_radius": self.obstacle_inflation_radius,
            "reach_threshold": self.reach_threshold,
            "goal_switch_smoothing": self.goal_switch_smoothing,
            "map_size": self.map_size,
            "num_static_obstacles": self.num_static_obstacles,
            "obstacle_size_range": self.obstacle_size_range,
            "max_episode_steps": self.max_episode_steps,
            "render_waypoints": self.render_waypoints,
            "save_trajectories": self.save_trajectories,
        }


