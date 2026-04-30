# Long-Range Navigation Contract

## Scope and goals

The `LongRangeNavi` module provides **global path planning** (A* on a grid) and **per-agent waypoint progression** for long-range navigation. It is used by the gym environment when `enable_long_range_nav=True`. It does **not** run the simulation; it produces waypoint lists and updates the current goal index based on agent position.

**Out of scope**: RVO, deadlock, or reward computation are in gym_env and other modules.

## Public interfaces

### LongRangeConfig

- **Attributes** (class or instance): `grid_resolution` (0.5), `waypoint_min_spacing` (5.0), `path_simplification_epsilon` (0.5), `obstacle_inflation_radius` (0.5), `reach_threshold` (0.2), `goal_switch_smoothing` (True), `map_size`, `num_static_obstacles`, `obstacle_size_range`, `max_episode_steps`, `render_waypoints`, `save_trajectories`.
- **as_dict()**: Returns a dict of the above for serialization. Caller may pass a dict with the same keys instead of an instance.

### GlobalPathPlanner

- **Constructor**: `GlobalPathPlanner(grid: List[List[int]], resolution: float, waypoint_spacing: float)`
  - **grid**: 2D occupancy grid (0=free, non-zero=obstacle); row = y, col = x in world.
  - **resolution**: Cell size in world units.
  - **waypoint_spacing**: Min distance between consecutive waypoints for sparsification.
- **plan_path(start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[Tuple[float, float]]**: Returns list of waypoints in world coordinates (x, y) from start to goal; uses A* (ISearch) then sparsifies by waypoint_spacing. If no path found, returns `[goal_xy]`.
- **separate_waypoints(waypoint_lists, min_distance)**: (Optional) Separates waypoints across agents to avoid overlap; may be disabled in env. Returns list of waypoint lists per agent.

### WaypointManager

- **Constructor**: `WaypointManager(agent_id: int, waypoint_list: List[Tuple[float, float]], reach_threshold: float = 1.0, force_switch_enabled: bool = False, force_switch_steps: int = 0)`
  - **waypoint_list**: Ordered list of (x, y) in world coordinates.
  - **reach_threshold**: Distance to waypoint to count as reached.
  - **force_switch_enabled / force_switch_steps**: If set, advance to next waypoint after force_switch_steps without reaching (non-final only).
- **get_current_goal() -> Optional[Tuple[float, float]]**: Returns current waypoint or None if all consumed.
- **update(agent_position: Tuple[float, float]) -> (bool, bool)**: Updates progress; returns (goal_reached_this_step, final_goal_reached). If within reach_threshold of current waypoint, advances index; if index past end, returns (True, True).
- **get_progress_info()**: Returns dict with agent_id, current_index, remaining, total (for logging).

## Input/output and errors

- **grid**: Must be consistent with env world (same resolution and origin as used in env). Built by env from obstacles and bounds.
- **start_xy / goal_xy**: Continuous world coordinates (e.g. from robot.state and robot.goal).
- **WaypointManager.update**: Idempotent per step; caller should call once per step per agent. Reaching the last waypoint sets final_goal_reached=True; env may set done=True for that agent.

## Performance

- plan_path: O(grid cells) for A*; waypoint count is bounded by path length / waypoint_spacing.
- WaypointManager.update: O(1) per call.

## Versioning

- Adding optional parameters to LongRangeConfig or WaypointManager is backward compatible.
- Changing plan_path return format (e.g. including headings) would be breaking if env assumes list of (x,y).

## Example usage (from env)

```python
# On reset:
grid, resolution, w, h = self._build_occupancy_grid_for_long_range()
planner = GlobalPathPlanner(grid, resolution, waypoint_spacing)
for aid, robot in enumerate(robots):
    start = (robot.state[0,0], robot.state[1,0])
    goal = (robot.goal[0,0], robot.goal[1,0])
    waypoints = planner.plan_path(start, goal)
    mgr = WaypointManager(aid, waypoints, reach_threshold=reach_thr)
    self._waypoint_managers[aid] = mgr
# Each step:
pos = (robot.state[0,0], robot.state[1,0])
reached, final = self._waypoint_managers[aid].update(pos)
cur_goal = self._waypoint_managers[aid].get_current_goal()
if cur_goal is not None:
    robot.goal = np.array([[cur_goal[0]], [cur_goal[1]]])
```

## Consumers

- **gym_env (ir_gym)**: Builds occupancy grid, creates GlobalPathPlanner and per-agent WaypointManager on reset; each step updates waypoints and sets robot.goal. See [gym_env/docs/integrations/long_range_navigation.md](../gym_env/docs/integrations/long_range_navigation.md).
