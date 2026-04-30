# Mode Management Contract

## Scope and goals

The `mode_management` module provides **mode state** and **mode transition logic** for each agent: `rl_rvo` (normal RL+RVO) vs `mapf` (following PAR/CBS waypoints). It is used by the gym environment when deadlock resolution is enabled. It does **not** run the simulation; it stores per-agent mode and PAR status and decides when to switch modes given the deadlock detector and PAR coordinator.

**Out of scope**: Observation/reward computation and world loading are in gym_env.

## Public interfaces

### StateManager

- **Constructor**: `StateManager()` — no config.
- **set_par_mode(agent_id, par_solution=None, start_position=None, goal_position=None)**: Sets agent to `mapf` mode; PAR status: in_par_mode=True, move_to_par_pos=True, par_exec=False, wait_for_finish=False; stores par_solution, start_position, goal_position; resets par_path and par_path_index.
- **set_rl_rvo_mode(agent_id)**: Sets agent to `rl_rvo` mode; clears PAR-related state.
- **get_agent_mode(agent_id) -> str**: Returns `'rl_rvo'` or `'mapf'`.
- **get_par_status(agent_id) -> Dict**: Returns dict with keys in_par_mode, move_to_par_pos, par_exec, wait_for_finish.
- **update_par_status(agent_id, status_updates)**: Merges status_updates into par_status.
- **set_par_executing(agent_id)**, **set_par_waiting(agent_id)**: Set par_exec True/False and wait_for_finish.
- **get_par_solution(agent_id)**, **get_par_start_position(agent_id)**, **get_par_goal_position(agent_id)**, **get_par_path(agent_id)**, **get_par_path_index(agent_id)**: Accessors for stored PAR data.
- **set_par_path(agent_id, path)**, **increment_par_path_index(agent_id)**: Set path and advance index.
- **is_in_par_mode(agent_id)**, **is_moving_to_start(agent_id)**, **is_par_executing(agent_id)**, **is_par_waiting(agent_id)**: Booleans for current phase.
- **reset_episode()**, **force_reset_all_agents_to_rl_rvo(num_agents)**: Clear all state or init all agents to rl_rvo.

### ModeController

- **Constructor**: `ModeController(deadlock_detector: DeadlockDetector, par_coordinator: PARCoordinator, config: Dict)`
  - **config**: Dict with PAR_COMPLETION_THRESHOLD, MODE_SWITCH_DELAY, NARROW_CORRIDOR_THRESHOLD, CONFINED_AREA_VELOCITY_THRESHOLD, GOAL_TOLERANCE.
- **update_agent_mode(agent_id, current_mode, agent_states, neighbor_states, current_time=0) -> str**: Returns new mode ('rl_rvo' or 'mapf') based on can_switch_mode, should_switch_to_par, should_switch_to_rl_rvo.
- **should_switch_to_par(agent_id, agent_states, neighbor_states) -> bool**: True when deadlock_detector.detect_deadlock is True or narrow-corridor condition holds.
- **should_switch_to_rl_rvo(agent_id, agent_states) -> bool**: True when par_coordinator.is_par_complete(agent_id) or has_reached_goal_through_par or is_par_execution_failed.
- **can_switch_mode(agent_id, current_time) -> bool**: True if enough steps since last_mode_switch_time (MODE_SWITCH_DELAY).
- **get_mode_transition_action(agent_id, current_mode, target_mode) -> Dict**: Returns action description for logging (e.g. switch_to_mapf / switch_to_rl_rvo).
- **reset_agent_mode_state(agent_id)**, **reset_all_mode_states()**: Clear switch counters and last switch time.

## Input/output and assumptions

- **agent_states / neighbor_states**: Same format as in deadlock_resolution contract (position, velocity, goal).
- **current_time**: Typically env step index; used for MODE_SWITCH_DELAY.
- StateManager is the single store for mode and PAR state; the env reads it to decide whether to run RL policy or PAR executor for each agent.

## Consumers

- **gym_env (ir_gym)**: Creates StateManager and ModeController when deadlock resolution is enabled; uses StateManager for mode and PAR state, ModeController for transition decisions (or equivalent logic in _step_with_deadlock_resolution). See [gym_env/docs/integrations/deadlock_resolution.md](../gym_env/docs/integrations/deadlock_resolution.md).
