# Design Documents vs Current Implementation

This document compares the desired features from the design/requirement notes under `RL_RVO/` with the current codebase in `rl_rvo_nav/`. Each section states whether the design has been **implemented**, **partially implemented**, or **not implemented**, with brief evidence.

---

## 1. Deadlock trigger and participant setup (about_deadlock_trigger_and_setup.md)

### 1.1 Trigger: TTC/DMIN risk gating (SPEED_BUFFER)

- **Desired**: In addition to low average velocity, require at least one neighbor with “not progressing toward goal” and pair metrics (TTC ≤ threshold, dmin ≤ threshold) to avoid false triggers when agents are merely slow but far apart.
- **Current**: `deadlock_detector.check_speed_buffer_trigger` uses `_compute_pair_metrics`, `ttc_threshold`, `dmin_threshold`, `required_non_progress_neighbors`, and `risk_neighbors`; trigger requires `cond_neighbors_slow and cond_non_progress and cond_risk`.
- **Status**: **Implemented.**

### 1.2 Participants: local conflict graph + connected component

- **Desired**: Replace “at most two agents (best pair)” with building a local conflict graph (edges by TTC/dmin + not progressing), take connected component containing seed, prioritize and clip by `MAX_PAR_PARTICIPANTS`; fallback to best neighbor pair if component too small.
- **Current**: `_build_local_conflict_graph`, `_extract_component`, `_prioritize_component` exist; `get_deadlock_participants` uses component and falls back to best neighbor / closest neighbor on timeout.
- **Status**: **Implemented.**

### 1.3 PAR execution: exclude PAR neighbors from RVO

- **Desired**: When in PAR mode, do not use other PAR participants as RVO neighbors (or reduce weight) so PAR agents are not mutually suppressed.
- **Current**: In `ir_gym.observation_reward` and `observation`, when `current_mode == 'mapf'` and `EXCLUDE_PAR_NEIGHBORS_IN_RVO` is True, `nei_used` is set to `[]` so PAR agents do not see each other in RVO. Config key `EXCLUDE_PAR_NEIGHBORS_IN_RVO` exists in `deadlock_config.py`.
- **Status**: **Implemented.**

### 1.4 Non-participants: speed cap near PAR zone (let PAR pass)

- **Desired**: Non-participants inside or near the PAR region have their action capped (e.g. `PAR_PROTECTED_ZONE_SPEED_CAP`) to yield to PAR execution.
- **Current**: No `PAR_PROTECTED_ZONE_SPEED_CAP` or similar logic found; no speed cap for non-participants near PAR zone.
- **Status**: **Not implemented.**

### 1.5 Exit: per-agent completion + timeout

- **Desired**: Each PAR participant exits when its own waypoint manager is final (or on timeout), not only when all participants complete; config `PAR_AGENT_TIMEOUT_STEPS`.
- **Current**: In `_step_with_deadlock_resolution`, per-agent exit is implemented: for each `pid` in `current_par_agents`, completion is checked via `_waypoint_managers[pid].get_current_goal() is None`, and timeout via `PAR_AGENT_MAX_STEPS` and `state_manager.get_mode_switch_time(pid)`. `_force_par_agent_exit` exists. Config key is `PAR_AGENT_MAX_STEPS` (same intent as PAR_AGENT_TIMEOUT_STEPS).
- **Status**: **Implemented.**

---

## 2. Metrics (add_metrics.md)

### 2.1 Navigation: makespan, flowtime, path length, average speed

- **Desired**: Per-episode makespan, flowtime, average travel time, path length per agent, average speed; aggregate in session and write to a single metrics file.
- **Current**: `test_logger.py` computes `travel_times`, `path_lengths`, `makespan`, `flowtime`, `avg_travel_time`, `avg_path_length`, `avg_speed` per episode and session; writes `metrics_summary` and `metrics_summary.json`.
- **Status**: **Implemented.**

### 2.2 Deadlock: rate, resolution success, trigger frequency, participant distribution

- **Desired**: Deadlock occurrence rate, resolution success rate, average resolution time, trigger count/frequency per episode/agent, participant count distribution.
- **Current**: `deadlock_logger.py` records deadlock events, PAR/CBS executions, and mode switches; session metrics can be aggregated. Exact presence of “resolution success rate” and “average resolution time” and “participant distribution” in the final metrics summary should be confirmed in `deadlock_logger` and any code that builds `metrics_summary`.
- **Status**: **Partially implemented** (events and execution counts exist; full aggregation into a single metrics summary may need verification).

### 2.3 Runtime: step timing, MAPF solve time, episode wall-clock

- **Desired**: Per-step control cycle time (RL + deadlock check + MAPF), MAPF solve time distribution, episode wall-clock time; baseline vs method comparison.
- **Current**: `ir_gym` has `_last_step_timing` with `deadlock_check_time` and `mapf_solve_time`; `par_coordinator` logs solver details. TestLogger session metadata and metrics_summary can include timing if wired; episode wall-clock is typical in loggers.
- **Status**: **Partially implemented** (step/MAPF timing present; full pipeline and baseline comparison script may need verification).

---

## 3. PAR execution as waypoint tracking (agent_execute_reconstruct.md)

### 3.1 Remove set_position queue and position overwrite

- **Desired**: Remove PAR “set_position” queue and any direct position overwrite before dynamics; PAR agents advance only via the same dynamics as RL, with goals set from waypoints.
- **Current**: PAR path is injected into `WaypointManager`; agents in MAPF mode follow waypoints through the same step path (waypoint update → goal → RVO → dynamics). No separate “apply PAR positions before _step_pure_rl” block; PAR execution uses waypoint manager, not `execute_par_step`-driven set_position in the main dynamics.
- **Status**: **Implemented.**

### 3.2 PAR and RL share same pipeline (waypoint → RVO → dynamics)

- **Desired**: PAR agents use the same RL pipeline: waypoint progression, then RVO, then dynamics; no separate PAR execution branch that overwrites position.
- **Current**: Same step flow for all agents; MAPF agents’ `robot.goal` is set from their (PAR-injected) waypoint manager; RVO and dynamics use that goal. PAR-specific “execute_par_step” and set_position application have been removed from the main step in favor of waypoint injection.
- **Status**: **Implemented.**

### 3.3 Exit by waypoint manager final (per-agent)

- **Desired**: Exit PAR when the agent’s waypoint manager reports final (current goal None), with optional per-agent timeout.
- **Current**: Exit condition is `cur_goal2 is None` (manager consumed all waypoints) or timeout (`PAR_AGENT_MAX_STEPS`) or stay_fail; then `set_rl_rvo_mode(pid)` and restore `_saved_lr_managers`.
- **Status**: **Implemented.**

### 3.4 _inject_par_waypoints and restore LR manager

- **Desired**: Helper to inject PAR path into WaypointManager and save/restore long-range manager on exit.
- **Current**: PAR path is converted to continuous and used to create a WaypointManager (or replace existing); `_saved_lr_managers` is used to restore the original long-range manager on per-agent exit.
- **Status**: **Implemented.**

---

## 4. Deadlock debug and state (deadlock debug note.md)

### 4.1 State manager reset and force rl_rvo

- **Desired**: On episode reset, ensure all agents start in `rl_rvo`; avoid stale `par` mode so deadlock detection runs.
- **Current**: `StateManager.force_reset_all_agents_to_rl_rvo(num_agents)` exists and is used on reset; `get_agent_mode` returns `'rl_rvo'` when agent not in state.
- **Status**: **Implemented.**

### 4.2 Participants ≥ 2 for PAR

- **Desired**: PAR should run only when at least two participants; otherwise do not switch to PAR.
- **Current**: In `_step_with_deadlock_resolution`, when `len(deadlock_participants) > 1` (and valid path check) PAR/CBS is prepared and waypoints injected; single-agent case does not start PAR.
- **Status**: **Implemented.**

### 4.3 neighbor_states for participant selection

- **Desired**: `get_deadlock_participants` should receive meaningful neighbor state (e.g. positions) so that multiple participants can be chosen.
- **Current**: `agent_states` and `neighbor_states` are built in `ir_gym` from robot list; conflict graph uses `agent_states` and communication range. If neighbor_states were empty in the past, the current graph-based participant selection uses `agent_states` and `_get_neighbors_in_range`, so it can return multiple participants.
- **Status**: **Implemented** (graph-based selection no longer relies only on the passed-in neighbor_states for building the set).

---

## 5. Deadlock detection setup (deadlock_detect_setup.md, deadlock par setup.md)

### 5.1 Two trigger types (SPEED_BUFFER, COMMON_POINT)

- **Desired**: Support COMMON_POINT (neighbor count + distance to target) and SPEED_BUFFER (mean speed below threshold).
- **Current**: `deadlock_detector` implements SPEED_BUFFER (and waypoint-stuck). Config has `TRIGGER_TYPE` but the only active trigger logic is speed-buffer and waypoint-stuck; there is no separate COMMON_POINT branch in the detector.
- **Status**: **Partially implemented** (SPEED_BUFFER + waypoint-stuck yes; COMMON_POINT no).

### 5.2 Participants: self + N(i) + N(N(i)) and MAPF build

- **Desired**: Participants = trigger agent + neighbors + neighbors of neighbors; then build MAPF graph, set start/goal, call PAR.
- **Current**: Participants are determined by local conflict graph + connected component (or fallback pair), then PAR coordinator builds environment and calls PNR solver. Different shape (graph-based) but same idea (local set, then MAPF).
- **Status**: **Implemented** (with graph-based extension).

### 5.3 UpdatePAR / UnitePAR (dynamic re-solve and merge)

- **Desired**: When a PAR group sees a non-PAR neighbor, re-solve MAPF (UpdatePAR); when two PAR groups are close, merge and re-solve (UnitePAR).
- **Current**: No `UpdatePAR` or `UnitePAR` logic; no re-solve when a non-participant enters the region or when two PAR groups meet.
- **Status**: **Not implemented.**

---

## 6. Mode 7 curriculum learning (MODE7_CURRICULUM_LEARNING_GUIDE.md)

- **Desired**: Stage 1/2/3 (and 4) training scripts, curriculum manager, mode7 YAMLs, model naming and report.
- **Current**: `train_process_obs_s1.py`–`train_process_obs_s4.py`, `curriculum_learning_manager.py`, and mode7 stage YAMLs exist under `policy_train/`; guide references match.
- **Status**: **Implemented.**

---

## 7. Long-range A* waypoints (multi-robot navigation with A* waypoints.md)

- **Desired**: LongRangeNavi (GlobalPathPlanner, WaypointManager, LongRangeConfig), enable_long_range_nav in env, reset builds grid and waypoint managers, step updates waypoints and robot.goal; policy_test_long_range.py and policy_test_long_range_with_deadlock.py; mode8_long_range.yaml.
- **Current**: `LongRangeNavi` has `GlobalPathPlanner`, `WaypointManager`, `LongRangeConfig`; `ir_gym` has `enable_long_range_nav`, `_waypoint_managers`, reset builds planner and managers, step updates waypoints and sets `robot.goal`; `policy_test_long_range.py` and `policy_test_long_range_with_deadlock.py` exist; `mode8_long_range.yaml` exists.
- **Status**: **Implemented.**

---

## 8. RL avoid PAR agent (RL avoid PAR agent.md)

- **Desired**: Replace ad-hoc “RL yielding” with RVO-based avoidance of PAR agents: either extend `config_vo_inf` with `par_agent_list` and treat PAR as VO, or include PAR agents in neighbor list for RVO.
- **Current**: No separate `par_agent_list` in `rvo_inter.config_vo_inf`. When the *current* agent is in PAR mode, PAR neighbors are excluded from RVO (`EXCLUDE_PAR_NEIGHBORS_IN_RVO`). For RL agents, other robots (including those in PAR) remain in `nei_state_list`, so RL agents do consider PAR agents as RVO neighbors. The document’s “minimal change” was to add PAR explicitly as a separate VO list; we did not add that, but RL does see PAR agents via the normal neighbor list.
- **Status**: **Partially implemented** (RL sees PAR in RVO via neighbors; no dedicated par_agent_list or special VO handling for PAR).

---

## 9. RL+CBS baseline (RL_CBS_baseline_implementation_plan.md, RL_CBS_data_pipeline.md)

### 9.1 CBS coordinator and config switch

- **Desired**: `cbs_coordinator.py` with same prepare/get_path interface as PAR; config `USE_CBS_INSTEAD_OF_PAR`; in deadlock branch call CBS or PAR and inject waypoints the same way.
- **Current**: `deadlock_resolution/cbs_coordinator.py` has `CBSCoordinator.prepare_cbs_execution` and `get_agent_path`; `ir_gym` checks `USE_CBS_INSTEAD_OF_PAR` and uses `cbs_coordinator` when True; waypoint injection is shared with PAR. `deadlock_cbs.json` and `policy_test_long_range_with_cbs.py` exist.
- **Status**: **Implemented.**

### 9.2 Config not overwritten after load

- **Desired**: When `enable_deadlock_resolution_mode(config_file)` loads config, `_initialize_deadlock_modules` must not overwrite `deadlock_config` so that `USE_CBS_INSTEAD_OF_PAR` is preserved.
- **Current**: In `ir_gym._initialize_deadlock_modules`, `self.deadlock_config = DeadlockConfig()` is done only when `self.deadlock_config is None`.
- **Status**: **Implemented.**

---

## 10. Overall design (rl deadlock v1.md)

- **Desired**: deadlock_resolution (detector, par_coordinator, par_executor, par_environment), mode_management (mode_controller, state_manager), config, ir_gym/mrnav integration, state machine rl_rvo ↔ par.
- **Current**: Module layout and flow match; PAR execution is now via waypoint injection and shared step path rather than separate set_position execution.
- **Status**: **Implemented.**

---

## Summary table

| Topic | Document(s) | Status |
|-------|-------------|--------|
| Trigger TTC/DMIN + non-progress | about_deadlock_trigger_and_setup | Implemented |
| Participants: conflict graph + component | about_deadlock_trigger_and_setup | Implemented |
| EXCLUDE_PAR_NEIGHBORS_IN_RVO | about_deadlock_trigger_and_setup | Implemented |
| PAR_PROTECTED_ZONE_SPEED_CAP (non-participant yield) | about_deadlock_trigger_and_setup | Not implemented |
| Per-agent PAR exit + timeout | about_deadlock_trigger_and_setup, agent_execute_reconstruct | Implemented |
| Metrics (makespan, flowtime, path length, etc.) | add_metrics | Implemented |
| Deadlock/runtime metrics aggregation | add_metrics | Partially implemented |
| PAR as waypoint only (no set_position) | agent_execute_reconstruct | Implemented |
| State manager reset / force rl_rvo | deadlock debug note | Implemented |
| COMMON_POINT trigger | deadlock_detect_setup, deadlock par setup | Not implemented |
| UpdatePAR / UnitePAR | orca deadlock combination, deadlock par setup | Not implemented |
| Mode 7 curriculum | MODE7_CURRICULUM_LEARNING_GUIDE | Implemented |
| Long-range A* waypoints | multi-robot navigation with A* waypoints | Implemented |
| RVO par_agent_list (explicit PAR VO) | RL avoid PAR agent | Partially implemented |
| RL+CBS baseline + config preservation | RL_CBS_* | Implemented |
| Overall deadlock + mode design | rl deadlock v1 | Implemented |

---

## Recommended next steps (if aligning fully with design docs)

1. **PAR_PROTECTED_ZONE_SPEED_CAP**: In `_step_with_deadlock_resolution`, for agents not in `deadlock_participants`, if inside or near the PAR region (e.g. distance to any participant &lt; threshold), cap action magnitude to a configurable value.
2. **COMMON_POINT trigger**: In `deadlock_detector`, add a branch when `TRIGGER_TYPE == 'COMMON_POINT'`: trigger when neighbor count ≥ MAPF_NUM and distance to current goal &lt; SIGHT_RADIUS (using existing agent_states/goal).
3. **UpdatePAR / UnitePAR**: In `_step_with_deadlock_resolution`, when in MAPF mode, detect (a) non-PAR neighbor entering participant region or (b) another PAR group within merge distance; then re-run participant selection and PAR/CBS prepare, and re-inject waypoints (with care to avoid livelock).
4. **Metrics**: Ensure `deadlock_logger` and TestLogger write resolution success rate, average resolution time, and participant count distribution into the same `metrics_summary.json` used by add_metrics.md; add optional `metrics_report.py` to compare two session directories.
