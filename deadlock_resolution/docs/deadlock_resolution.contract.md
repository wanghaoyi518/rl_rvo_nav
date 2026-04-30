# Deadlock Resolution Contract

## Scope and goals

The `deadlock_resolution` module provides **deadlock detection** and **resolution via Push and Rotate (PAR)** (and optionally CBS, or a rule-based sequential solver). It is used by the gym environment when `enable_deadlock_resolution=True`. The module does **not** run the simulation; it only detects deadlocks, selects participants, computes PAR (or CBS, or rule-based) plans, and exposes execution state so the env can apply actions.

**Out of scope**: Policy training, reward design, or world loading are outside this module.

## Public interfaces

### DeadlockDetector

- **Constructor**: `DeadlockDetector(config: Dict)`
  - **config**: Dict with keys (defaults in parentheses): `TRIGGER_TYPE` ('SPEED_BUFFER'), `SMALL_SPEED` (0.2), `VELOCITY_WINDOW_SIZE` (5), `EPISODE_START_DELAY` (5), `DEADLOCK_DETECTION_COOLDOWN` (5), `GOAL_TOLERANCE` (0.1), `COMMUNICATION_RANGE` (3.0/7.0), `TTC_THRESHOLD`, `DMIN_THRESHOLD`, `REQUIRED_NON_PROGRESS_NEIGHBORS`, `ENABLE_WAYPOINT_STUCK_TRIGGER`, `WAYPOINT_STUCK_STEPS`, `SINGLE_AGENT_TRIGGER_ENABLED`, `SINGLE_AGENT_TIME_THRESHOLD`, `MAX_PAR_PARTICIPANTS`, `MIN_PAR_PARTICIPANTS` (4), `MAX_GRAPH_HOPS`, `PARTICIPANT_LOCK_STEPS`, etc.
  - **detect_deadlock(agent_id, agent_states, neighbor_states) -> bool**: Returns True if the agent is considered in deadlock. Uses velocity history (updated internally) and optional waypoint-stuck trigger. Caller must increment `step_counter` each step and call `update_waypoint_history(agent_id, waypoint_index)` when long-range waypoints are used.
  - **get_deadlock_participants(agent_id, agent_states, neighbor_states) -> List[int]**: Returns list of agent IDs to include in PAR. Selection uses a local conflict graph (communication range, active nodes, TTC/dmin edges), extracts the connected component containing the seed, prioritizes and clips it by `MAX_PAR_PARTICIPANTS`, and then applies a **hard lower bound**: the final participant set is returned only if its size is at least `MIN_PAR_PARTICIPANTS`; otherwise an empty list is returned and MAPF is not triggered for that detection.
- **reset_episode()**: Clears velocity history and cooldown state; call on env reset.
- **update_waypoint_history(agent_id, waypoint_index)**: Call once per step per agent when using long-range waypoints (for waypoint-stuck trigger).
- **set_logger(logger)**: Optional; sets logger for debug output.

**Agent state format**: Each `agent_states[id]` and `neighbor_states[id]` should provide `position`, `velocity`, and `goal` (goal can be `[x,y]` or `[[x],[y],[theta]]`). Detector uses these for distance-to-goal, velocity magnitude, and TTC/dmin.

### PARCoordinator

- **Constructor**: `PARCoordinator(push_and_rotate_instance: PushAndRotate, config: Dict, gym_env=None)`
  - **push_and_rotate_instance**: From `python_pnr.push_and_rotate.PushAndRotate`.
  - **config**: Dict with `GRID_RESOLUTION`, `PAR_OFFSET`, `GOAL_TOLERANCE`, `PAR_TIMEOUT`, `COMMUNICATION_RANGE`, `PAR_DISABLE_CROP`, `DEBUG_MODE`, `DISABLE_ID_REMAP`, etc.
  - **gym_env**: Reference to env for workspace bounds and obstacles.
- **prepare_par_execution(agent_states, deadlock_participants) -> MAPFSearchResult**: Builds PAR environment (sub-map, actor set), computes start/goal positions, calls PNR solver, returns result with `success`, `agents_moves`, optional `paths`, `id_solver_to_real`, `grid_offset`. If solver fails, returns an unsuccessful result with empty `agents_moves`.
- **get_agent_path(agent_id) -> List[Tuple]**: Returns grid or continuous path for the agent from current solution (or empty list).
- **is_par_complete(agent_id) -> bool**: True if agent reached goal or completed path or timed out.
- **get_workspace_info(agent_states) -> Dict**: Returns workspace bounds and obstacles for PAR env build.
- **reset()**: Clears current solution and participants.

### PARExecutor

- **Constructor**: `PARExecutor(config: Dict)`  
  Config: `POSITION_TOLERANCE`, `VELOCITY_SCALE`, `MAX_VELOCITY`, `PAR_SUBSTEPS_PER_GRID`, `DEBUG_MODE`.
- **set_dependencies(state_manager, par_coordinator)**: Injects StateManager and PARCoordinator.
- **execute_par_step(agent_id, agent_states) -> Dict**: Returns action dict with keys `action` (velocity or zero), `mode` ('move_to_start'|'at_start'|'follow_path'|'path_complete'|'idle'|'no_solution'|'error'), `target`, optional `set_position` (for env to teleport agent).
- **initialize_par_execution(agent_id, par_solution)**: Converts PAR solution path to continuous path and sets internal path for the agent.
- **set_agent_start_position(agent_id, start)**, **set_agent_goal_position(agent_id, goal)**: Set start/goal for move-to-start phase.
- **is_par_complete(agent_id) -> bool**: True when path index >= path length.
- **reset_agent(agent_id)**, **reset_all()**: Clear execution state.

### PAREnvironment

- Used internally by PARCoordinator to build SubMap and actor set from workspace and agent states. Not required to be used directly by env; the env uses PARCoordinator and PARExecutor only.

### CBSCoordinator

- Optional; provides CBS-based coordination when used instead of (or in addition to) PAR. Contract similar in spirit (prepare execution, get paths).
- **prepare_cbs_execution(agent_states, deadlock_participants) -> self | None**: Same inputs as PAR. Returns `self` on success (paths stored; use `get_agent_path(agent_id)` for continuous waypoints); returns `None` on failure (e.g. `cbs_mapf` import error, occupancy grid build failure, or solver finds no solution). Caller must not assume MAPF mode is set when None is returned.
- **Dependency**: Requires the `cbs-mapf` package (`pip install cbs-mapf`). If the package is missing, `prepare_cbs_execution` returns None and the env keeps agents in RL_RVO.

### RuleBasedSequentialCoordinator

- Optional; provides rule-based sequential coordination: participants are ordered by position (left-to-right, bottom-to-top); only one agent moves at a time; when the current-priority agent reaches the next agent's position, the next agent starts. Same input/output contract as PAR and CBS.
- **Constructor**: `RuleBasedSequentialCoordinator(config: Dict, gym_env=None)` — same as CBSCoordinator. `gym_env` is optional (used only if future extensions need grid/obstacles).
- **prepare_rule_based_execution(agent_states, deadlock_participants) -> self | None**: Same inputs as PAR/CBS. Returns `self` on success (paths stored; use `get_agent_path(agent_id)` for continuous waypoints); returns `None` on failure (e.g. missing position/goal for a participant, or empty participants). Paths are in **continuous** coordinates; no grid-to-continuous conversion is needed by the caller.
- **get_agent_path(agent_id) -> List[Tuple[float, float]]**: Returns continuous waypoints for the agent, or empty list if no path. Same role as CBSCoordinator.get_agent_path.
- **reset()**: Clears current solution and participants (optional, for consistency with other coordinators).
- **Config**: Solver selection is via `DEADLOCK_SOLVER`: `'par' | 'cbs' | 'rule_based'`. If `USE_CBS_INSTEAD_OF_PAR` is True and `DEADLOCK_SOLVER` is not set, behavior is treated as `DEADLOCK_SOLVER='cbs'`. Optional keys for tuning: `RULE_BASED_WAIT_WAYPOINTS` (override wait length), `GRID_RESOLUTION` (for interpolation step size).
- **Consumers**: Same as PAR/CBS — gym_env (ir_gym) chooses solver via config and injects paths from `get_agent_path`; no separate executor (waypoint following only). See [gym_env/docs/integrations/deadlock_resolution.md](../../gym_env/docs/integrations/deadlock_resolution.md).

- **agent_states**: Dict mapping agent_id to state dict with at least `position`, `velocity`, `goal` (and optionally `prev_position`). Positions and goals in continuous world coordinates.
- **neighbor_states**: Dict mapping neighbor_id to same-shaped state dict.
- **MAPFSearchResult**: Has `success: bool`, `agents_moves: list`, optional `paths`, `id_solver_to_real`, `grid_offset`. Moves may use solver IDs; coordinator remaps to real IDs when attaching to result.
- Exceptions: PNR solver or PAR env build may raise; coordinator catches and returns unsuccessful result with empty moves. Caller should check `result.success` and handle empty path. For CBS, failures (import, grid, no solution) yield `None`; caller should not set MAPF mode and may log for diagnostics.

## Performance and constraints

- Detection is O(agents × neighbors) per step; velocity history is bounded by VELOCITY_WINDOW_SIZE.
- PAR solve time depends on participant count and map size; no hard timeout in contract (solver may have internal timeout).
- Grid resolution should match long-range config when both are used (coordinator reads gym_env.long_range_config for GRID_RESOLUTION).
- **Deadlock workspace and obstacle margin (single SoT)**:
  - The deadlock workspace (occupancy grid / sub-map) is built in the environment layer (e.g. `ir_gym._build_occupancy_grid_for_long_range` and `PAREnvironment.build_par_environment`) and is the **only place** where static obstacles are dilated.
  - Deadlock config key `DEADLOCK_OBSTACLE_MARGIN_CELLS` (optional) controls this dilation uniformly for all deadlock solvers: the environment and PAR share the same cell-level dilation semantics around static obstacles.
  - PAR may additionally use legacy keys `ENABLE_OBSTACLE_DILATION` / `OBSTACLE_DILATION_CELLS` for debugging; when `DEADLOCK_OBSTACLE_MARGIN_CELLS` is present it takes precedence and these legacy keys are ignored.
  - CBS reads the already-dilated grid from the environment and does **not** apply extra grid dilation; `CBS_ROBOT_RADIUS_CELLS` is reserved for solver-internal safety margin and does not change the underlying occupancy grid.
- **Logging**: Deadlock config key `DEBUG_MODE` (default False) gates per-step and per-agent debug prints (e.g. "DEBUG: Agent …", "PAR INIT/CBS INIT" path length, PAR EXECUTor path lookups). When False, only minimal one-line summaries per MAPF trigger (e.g. "MAPF: solver=cbs waypoints injected for N agents") and errors are printed. Set `DEBUG_MODE: true` in the deadlock config for verbose diagnostics.
- **Structured participant-selection traces**: When `DEBUG_MODE` is True, the deadlock episode JSON also includes `deadlock_check_traces`, one record per RL-agent deadlock check. Each record contains the per-step `loop_index` / `current_mode`, cooldown and min-history state, `average_velocity`, communication-range neighbors (`communication_range_neighbors`, `slow_neighbors`, `non_progress_neighbors`, `risk_neighbors`), participant-selection graph fields (`graph_adjacency`, `component_nodes`, `filtered_nodes`, `prioritized_nodes`, `clipped_nodes`), and the final gating / solver fields (`returned_participants`, `empty_return_reason`, `solver_cache_key`, `solver_invoked`, `solver_skipped_reason`). This structured trace is the debug contract for falsifying participant-selection hypotheses without attaching a debugger.
- **Experimental final-row direction-correction keys**: Deadlock config also accepts optional debug-stage keys `FINAL_ROW_DIRECTION_CORRECTION_ENABLED`, `FINAL_ROW_DIRECTION_CORRECTION_DEBUG_ONLY`, `FINAL_ROW_DIRECTION_CORRECTION_MIN_NON_PROGRESS_STEPS`, and `FINAL_ROW_DIRECTION_CORRECTION_WEAK_SPEED_RATIO`. These are env-side experimentation knobs, not accepted solver semantics. They gate a bounded runtime experiment in gym_env only; callers should treat them as temporary debug controls rather than stable deadlock-resolution behavior.

## Versioning

- New config keys are backward compatible if given defaults.
- Changing the semantics of `detect_deadlock` or the shape of `agents_moves`/paths is breaking for the env integration.

## Example usage (from env)

```python
# In _step_with_deadlock_resolution:
self.deadlock_detector.step_counter = self.step_count  # or similar
if self.deadlock_detector.detect_deadlock(agent_id, agent_states, neighbor_states):
    participants = self.deadlock_detector.get_deadlock_participants(agent_id, agent_states, neighbor_states)
    result = self.par_coordinator.prepare_par_execution(agent_states, participants)
    if result.success:
        for pid in participants:
            self.par_executor.initialize_par_execution(pid, result)
            self.state_manager.set_par_mode(pid, par_solution=result, ...)
# For agents in MAPF mode:
action_dict = self.par_executor.execute_par_step(agent_id, agent_states)
# Apply action_dict['set_position'] or action_dict['action'] as needed.
```

## Consumers

- **gym_env (ir_gym)**: Uses DeadlockDetector, PARCoordinator, PARExecutor, StateManager, ModeController; wires them in `_step_with_deadlock_resolution`. See [gym_env/docs/integrations/deadlock_resolution.md](../gym_env/docs/integrations/deadlock_resolution.md).
- **mode_management**: ModeController uses DeadlockDetector and PARCoordinator; StateManager stores PAR state. See same integration doc.
