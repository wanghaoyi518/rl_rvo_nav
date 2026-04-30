# Integration: gym_env → LongRangeNavi

## Caller and callee

- **Caller**: `gym_env` (ir_gym in `gym_env/envs/ir_gym.py`).
- **Callee**: `LongRangeNavi` (LongRangeConfig, GlobalPathPlanner, WaypointManager).

## Where and how they are used

- **When**: Only when `enable_long_range_nav=True`. On env reset (`env_reset`), ir_gym builds an occupancy grid via `_build_occupancy_grid_for_long_range()` (world bounds and obstacles, resolution from long_range_config), then creates one `GlobalPathPlanner` and one `WaypointManager` per robot. For each robot, it calls `planner.plan_path(start_xy, goal_xy)` with the robot’s initial state and final goal, then constructs `WaypointManager(agent_id, waypoints, reach_threshold=...)`. Optionally waypoints are separated with `planner.separate_waypoints` (currently may be disabled in code).
- **Each step**: Before observation/reward, for each agent (in both pure RL and deadlock step paths), ir_gym gets current position from robot.state, calls `waypoint_managers[aid].update(pos)`, then `get_current_goal()` and sets `robot.goal` to that waypoint so RVO and reward use the current subgoal. When `update` returns final_goal_reached=True, the env sets done=True for that agent. When long-range is used with deadlock, waypoint progress is also reported to DeadlockDetector via `update_waypoint_history(agent_id, waypoint_index)` for the waypoint-stuck trigger.

## Dependencies and assumptions

- **Grid**: Env builds the grid once per reset; resolution and bounds must match what LongRangeNavi expects (same coordinate system as robot state and goals). Grid is 0=free, non-zero=obstacle; row/col correspond to y/x.
- **WaypointManager**: Env calls update exactly once per agent per step; get_current_goal is used to set robot.goal. Reaching the last waypoint is the completion condition for the episode for that agent.
- **PAR interaction**: When PAR is triggered for a participant, env may replace that agent’s WaypointManager with a temporary one filled with PAR path waypoints; on PAR completion, the original long-range WaypointManager is restored.

## Configuration and injection

- **Config**: Passed as `long_range_config` dict or LongRangeConfig instance to `gym.make(..., long_range_config={...})`. Keys: grid_resolution, waypoint_min_spacing, reach_threshold, waypoint_separation_manhattan (if separation is enabled).
- **Mock/test**: Tests can disable long-range or pass a small grid and two waypoints to force deterministic behavior.

## Error handling and fallback

- If plan_path returns only [goal_xy] (no path found), the agent effectively has a single waypoint (final goal); behavior remains well-defined.
- If grid build fails, long-range init is skipped and _waypoint_managers may be left empty; env should handle missing waypoint managers (e.g. skip update or use robot.goal as-is).

## Conversion and validation

- start_xy and goal_xy are taken from robot.state and robot.goal in world coordinates; no conversion. Waypoints returned by plan_path are in world coordinates; robot.goal is set from get_current_goal() directly.
