# Deadlock Resolution Design — Internal

## Purpose

This document describes internal design of the deadlock_resolution module for maintainers. Callers rely on [deadlock_resolution.contract.md](deadlock_resolution.contract.md).

## File layout

- **deadlock_detector.py**: DeadlockDetector — velocity history, speed-buffer trigger, waypoint-stuck trigger, single-agent fallback, participant selection (local conflict graph, BFS component, prioritization).
- **par_coordinator.py**: PARCoordinator — workspace from gym_env, PAREnvironment build, PNR solver call (python_pnr), ID remap, grid_offset for cropped sub-maps, solution storage.
- **par_executor.py**: PARExecutor — move-to-start, follow PAR path with substeps, path from solution paths or agents_moves, set_position for env.
- **par_environment.py**: PAREnvironment — build SubMap and actor set from workspace and agent states, grid resolution alignment with long-range. Static obstacle dilation is applied here using `DEADLOCK_OBSTACLE_MARGIN_CELLS` when present in the config; legacy `ENABLE_OBSTACLE_DILATION` / `OBSTACLE_DILATION_CELLS` are only used when `DEADLOCK_OBSTACLE_MARGIN_CELLS` is absent.
- **cbs_coordinator.py**: CBSCoordinator — CBS-based alternative when enabled. Depends on external package `cbs-mapf` (and its low-level `space-time-astar`). If import fails, `prepare_cbs_execution` returns None and the env keeps agents in RL_RVO. **Coordinate convention**: space-time-astar uses (x, y) for grid positions; we use grid[row][col] with world (x,y) -> (col, row), so we pass (col, row) as (x, y) to the Planner and interpret plan() result as (col, row). **Grid extent**: the library infers grid bounds from the bounding box of `static_obstacles`; we add four boundary points at (-1,-1), (width,-1), (-1,height), (width,height) so the grid covers the full world and start/goal are inside it. **Obstacle margin**: the environment builds a deadlock occupancy grid (e.g. via `_build_occupancy_grid_for_long_range`) and dilates it using `DEADLOCK_OBSTACLE_MARGIN_CELLS` from deadlock config (when set), so all deadlock solvers see the same inflated static obstacles; CBS can add solver-internal safety margin via `CBS_ROBOT_RADIUS_CELLS` (default 0) without changing the underlying grid.
- **rule_based_coordinator.py**: RuleBasedSequentialCoordinator — rule-based sequential solver. No external MAPF package; uses only agent positions and goals. **Sorting**: participants ordered by `(x, y)` ascending (left-to-right, bottom-to-top). **Path construction**: relay style — agent i moves from its position to the next agent's position (last agent moves to its goal). **Wait segment**: because the env injects all waypoints at once, “only one moves at a time” is encoded by prefixing each non-first agent's path with repeated current-position waypoints; the number of repeats equals the previous agent's path length so that under typical step advancement the previous agent finishes before the next starts. Paths are continuous; straight-line interpolation by step size (e.g. GRID_RESOLUTION or RULE_BASED_WAIT_WAYPOINTS). First version does not avoid obstacles (straight segments only); obstacle-aware segments can be added later using the same grid as long-range if needed.
- **Planner-vs-simulator obstacle overlays**: The obstacle-collision debug overlay does not build a second planner grid. It reuses the same long-range / deadlock occupancy grid snapshot that gym_env already uses for planning (`_build_occupancy_grid_for_long_range()` plus the active dilation settings). For a reproduced `robot_obstacle` collision, the logger compares the simulator collision point against that grid and classifies it as planner-occupied, planner-free, or planner-out-of-bounds. This keeps Issue 7 evidence anchored to the existing deadlock workspace semantics rather than to a parallel debug-only rasterization.

## Detection triggers

- **SPEED_BUFFER**: Low average velocity over window, plus at least one slow neighbor not progressing toward goal, plus risk (TTC/dmin) and optional single-agent fallback after delay.
- **WAYPOINT_STUCK**: Same waypoint index for >= WAYPOINT_STUCK_STEPS while not at goal; independent of speed buffer.
- **SINGLE_AGENT**: When only one agent is unfinished, relaxed trigger (e.g. not progressing and slow or after SINGLE_AGENT_TIME_THRESHOLD).

Participant selection uses a local conflict graph (communication range, active nodes, TTC/dmin edges), then connected component containing seed, then prioritization and max participants; fallback to pairwise best neighbor or closest neighbor on timeout. A `MIN_PAR_PARTICIPANTS` (default 4) hard constraint is applied at the end: if the final participant set has fewer than `MIN_PAR_PARTICIPANTS` agents, an empty list is returned and the env does not start MAPF for that detection. As a result, runs may observe groups of size 4, 5, … up to `MAX_PAR_PARTICIPANTS`, but never 1–3 agents entering MAPF together under the normal multi-agent trigger.

## PAR and python_pnr

- PARCoordinator builds PAREnvironment (SubMap, actor set in grid), then calls `pnr_solver.start_search(sub_map, mapf_config, solver_actor_set)`. Solver returns MAPFSearchResult; coordinator remaps solver IDs to real agent IDs and attaches grid_offset for cropped grids. Executor uses `par_solution.paths` or reconstructs from agents_moves; grid_to_continuous uses PAREnvironment.

## Rule-based sequential solver (internal)

- **Sorting rule**: Order participants by key `(x, y)` lexicographic ascending. So: left has higher priority than right; if x equal, lower y has higher priority (bottom before top). Resulting order is `p0, p1, ..., p_{n-1}` with p0 highest priority.
- **Path semantics**: Relay style. For `i < n-1`, agent `order[i]` moves from `pos(order[i])` to `pos(order[i+1])`. For the last agent `order[n-1]`, path is from `pos(order[n-1])` to `goal(order[n-1])`. So “when current arrives at next’s position, next goes” is naturally satisfied by the segment endpoints.
- **Wait segment**: The env injects one waypoint list per agent at trigger time; it does not re-inject when “previous agent arrives”. So to approximate “only one moves at a time”, each path (except the first) is prefixed with `N_prev` copies of the agent’s current position, where `N_prev` is the number of waypoints in the previous agent’s path. Under typical waypoint advancement (reach current then advance), the previous agent consumes its list first, then the next agent’s list starts moving. Optional config `RULE_BASED_WAIT_WAYPOINTS` can override the wait length if needed.
- **Segment interpolation**: Each segment (start → end) is turned into a list of continuous waypoints by linear interpolation. Step size can use `GRID_RESOLUTION` or a fixed value so that segment length is bounded and non-zero. No grid or A* in the first version; segments are straight lines.

## Consistency with contract

- Detector does not run simulation; env is responsible for step_counter and update_waypoint_history.
- Coordinator returns a single result per prepare_par_execution; executor holds path state per agent until reset.
- Reset: detector.reset_episode(), state_manager and executor reset by env (or coordinator.reset()).

## Known limitations

- Fallback solution generation is disabled; solver failure yields empty moves.
- PAR_DISABLE_CROP is True by default (no sub-map cropping).
- Some debug prints remain; consider DEBUG_MODE gating.
