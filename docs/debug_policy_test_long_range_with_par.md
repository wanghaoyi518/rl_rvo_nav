# Debug: policy_test_long_range_with_par Integration Test Failures

This document records observed failure patterns, cause hypotheses, and attempted fixes for the full integration test `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`. It follows the repository's document system (see [structured_file_guidance.md](../../structured_file_guidance.md)) and is the single place to track debugging progress for this test. **Do not change code from this document**; use it to plan and log debugging only.

---

## 1. System Framework and Data Flow (from Paper + Code)

### 1.1 RL-MAPF Framework (from papers/l4dc_paper_outline_v1.md)

- **Hybrid architecture**: RL-based reactive navigation (default) + on-demand MAPF for deadlock resolution.
- **Layers**:
  - **Global path / waypoints**: Each agent has a long-range waypoint list; active target advances when within reach threshold.
  - **Navigation**: Two modes — **RL** (decentralized, waypoint-tracking) and **MAPF** (coordinated subplan for a local group).
  - **Base policy**: Pre-trained RL (e.g. RL-RVO) maps local obs + current waypoint → velocity; policy-agnostic wrapper.
  - **Safety / coordination**: Deadlock detection (progress-based) → form local group → run MAPF (PAR/CBS/rule_based) → inject waypoints → execute → return to RL when done.

- **Detection triggers** (deadlock_resolution):
  - Speed buffer: low average velocity + ≥1 slow neighbor not progressing + risk (TTC/dmin) + (optional) required number of non-progress neighbors.
  - Waypoint-stuck: same waypoint index for ≥ `WAYPOINT_STUCK_STEPS` while not at goal.
  - Single-agent fallback: when only one unfinished agent remains (configurable; disabled in current test).

- **Participant selection**: Local conflict graph (communication range, “active” nodes: slow, not progressing) → connected component containing **seed** → prioritize and clip by `MAX_PAR_PARTICIPANTS` → **hard lower bound** `MIN_PAR_PARTICIPANTS` (if below, return empty → no MAPF).

### 1.2 Data Flow (integration test)

1. **Entry**: `policy_test_long_range_with_par.py` → `post_train_with_deadlock.policy_test()` → `env.step_ir(abs_action_list)`.
2. **Step path** (when `enable_deadlock_resolution=True`):
   - `env.step_ir` → `_step_with_deadlock_resolution(action_list)` (ir_gym).
   - **Per step**: increment `step_count` and detector `step_counter`; build `agent_states`, `neighbor_states`.
   - **Per agent (loop `agent_id = 0..N-1`)**:
     - If mode is `rl_rvo`: call `deadlock_detector.detect_deadlock(agent_id, ...)`. If True → `get_deadlock_participants(agent_id, ...)` → if ≥1 participant and not cached, `_run_mapf_solver_and_build_paths(...)` → for each `valid_participant`: `state_manager.set_par_mode(participant_id, ...)`, inject waypoints (replace or set `_waypoint_managers[participant_id]`), optionally init `_par_tuple_group` (PAR solver).
   - **Waypoint alignment**: For all agents (including MAPF), `_waypoint_managers[aid].update(pos)` is called; current goal is written to `robot.goal`. Only for `rl_rvo` agents, `update_waypoint_history(aid, wp_idx)` is called for waypoint-stuck trigger.
   - **Actions**: Same `modified_action_list` (from RL policy) for all agents; MAPF agents get speed cap and non-PAR agents may get yielding near PAR participants. Then `_step_pure_rl(modified_action_list)` runs dynamics and RVO/collision.
   - **MAPF exit**: (1) **PAR tuple group**: when all participants reach current tuple row, advance index; when index at end, set all to `rl_rvo` and restore `_saved_lr_managers`. (2) **Per-agent**: non-tuple MAPF agents exit when waypoint manager reports no current goal, or `PAR_AGENT_MAX_STEPS` timeout, or stay on same waypoint ≥ 2× `FORCE_WAYPOINT_SWITCH_STEPS`.
3. **Observations / rewards / done**: From `_step_pure_rl` → `obs_move_reward_list` → RVO and collision checks; `collision_flag` → `robot.collision_check`, `done = True` on collision; long-range success = all agents’ waypoint manager `get_current_goal()` is None (final reached).

### 1.3 Test Configuration (from policy_test_long_range_with_par.py)

- `enable_deadlock_resolution=True`, solver `par`.
- `DeadlockConfig`: `REQUIRED_NON_PROGRESS_NEIGHBORS=3`, `SINGLE_AGENT_TRIGGER_ENABLED=False`.
- `MIN_PAR_PARTICIPANTS` is not set in script; default from config (e.g. `deadlock_config.py`) is **4** → MAPF only triggers when participant set has ≥ 4 agents (when single-agent path is disabled).

---

## 2. Observed Failure Patterns

| # | Pattern | Description |
|---|--------|--------------|
| **P1** | Wrong agents enter MAPF | The set of agents that switch to MAPF mode is not the “expected” deadlock group (e.g. we expect a specific subset in a bottleneck, but a different subset or different agents enter). |
| **P2** | Waypoint not advancing (stall until timeout) | Both RL and MAPF agents sometimes do not advance waypoints; they remain at the same target until episode hits `max_ep_len`. |
| **P3** | Collisions | Both RL and MAPF agents can collide with each other or with static obstacles; episode ends with collision failure. |

---

## 3. Cause Hypotheses and Attempt Log

### 3.1 P1: Not the expected agents enter the deadlock/MAPF group

**Observed**: Agents that enter the deadlock event / MAPF mode are not exactly the ones we expect (e.g. not the ones visibly stuck in a corridor/doorway).

**Design / tuning preferences (no code change yet)**:
- **First-detector wins**: Not preferred as the main lever; defer changing this for now.
- **Conflict graph ≠ true deadlock set**: Considered plausible; the graph can include agents that are slow/not progressing but not in the same bottleneck.
- **MIN_PAR_PARTICIPANTS**: Technically 2 is sufficient, but setting it to 2 in practice tends to trigger deadlock resolution too often. If lowering to 2, other detection parameters (e.g. cooldown, required non-progress neighbors, waypoint-stuck steps) may need to be tuned together so that deadlock does not fire too frequently.

#### 3.1.1 How `agent_neighbor_states` is defined

`agent_neighbor_states` is the **per-agent** neighbor view passed into the deadlock detector for that agent. It is produced as follows.

1. **Step-level construction (in `_step_with_deadlock_resolution`)**  
   - `ts = self.components['robots'].total_states()`  
   - `agent_states = self._get_agent_states_dict(ts[0])` → `{ agent_id: { 'position', 'velocity', 'goal' } }` for all agents.  
   - `neighbor_states = self._get_neighbor_states_dict(ts[1])` → see below.  
   - For each `agent_id`: `agent_neighbor_states = self._get_agent_neighbor_states(agent_id, agent_states, neighbor_states)`.

2. **`_get_neighbor_states_dict(nei_state_list)`** (ir_gym)  
   - **Does not use** `nei_state_list`; it re-fetches `ts = self.components['robots'].total_states()` and builds from `agent_states = _get_agent_states_dict(ts[0])`.  
   - **Radius**: `deadlock_config.get('COLLISION_WARNING_DISTANCE', 2.0)` (Euclidean).  
   - **Output**: A **nested** dict `nested[agent_id][neighbor_id] = { 'position', 'velocity' }` for every pair where distance between `agent_id` and `neighbor_id` ≤ radius.  
   - **Optional exclusion**: If `EXCLUDE_PAR_NEIGHBORS_IN_PIPELINE` is True (default), a pair is **omitted** from each other’s neighbor set when **both** are in mode `'mapf'`. So MAPF agents do not count each other as neighbors for this precomputed structure.  
   - **Note**: Neighbor entries do **not** include `'goal'`; only `position` and `velocity`.

3. **`_get_agent_neighbor_states(agent_id, agent_states, neighbor_states_nested)`** (ir_gym)  
   - **Primary path**: If `neighbor_states_nested.get(agent_id, {})` is a non-empty dict, return it. So for each agent, `agent_neighbor_states` is the precomputed `{ neighbor_id: { 'position', 'velocity' } }` from step 2.  
   - **Fallback**: If that dict is empty, build from `agent_states` by distance: same radius `COLLISION_WARNING_DISTANCE` (default 2.0); include every other agent within that radius; each value is `{ 'position', 'velocity' }` (no `goal`).  
   - So in both paths, the detector receives **neighbor state = position + velocity only**, and the **neighborhood is defined by radius = COLLISION_WARNING_DISTANCE** (default 2.0 m).

4. **How the detector uses it**  
   - **Velocity history**: `_update_all_velocity_histories(agent_id, agent_states, neighbor_states)` updates velocity history for neighbors using the passed-in `neighbor_states` (so the 2.0 m neighborhood affects whose history is updated).  
   - **Participant selection**: The detector does **not** use `agent_neighbor_states` for building the conflict graph. It uses `_get_neighbors_in_range(agent_id, agent_states, COMMUNICATION_RANGE)` (e.g. 7.0 m from config). So the **participant set** is driven by **COMMUNICATION_RANGE** and **agent_states**, not by the gym’s `agent_neighbor_states` (2.0 m, and with MAPF–MAPF exclusion in the precomputed path).  
   - **Speed-buffer trigger**: `check_speed_buffer_trigger` also builds “neighbors” via `_get_neighbors_in_range(agent_id, agent_states, comm_range)` (COMMUNICATION_RANGE), not from the passed-in neighbor_states. So **who counts as neighbor for detection/participant logic is COMMUNICATION_RANGE (e.g. 7.0)**, while **agent_neighbor_states** is the 2.0 m (optionally MAPF-excluded) view used for velocity history and any logic that explicitly iterates the passed-in neighbor dict.

**Summary**: `agent_neighbor_states` = per-agent dict of agents within **COLLISION_WARNING_DISTANCE** (default 2.0 m), each entry `{ 'position', 'velocity' }`; optionally MAPF–MAPF pairs excluded. Participant selection and “neighbors” in the detector use **COMMUNICATION_RANGE** (e.g. 7.0 m) and full **agent_states**, not this 2.0 m neighbor view.

**Hypotheses** (to be confirmed or rejected with runtime evidence):

| ID | Hypothesis | Rationale (code/design) | Status |
|----|------------|--------------------------|--------|
| H1a | **First-detector wins** | In `_step_with_deadlock_resolution`, the loop over `agent_id` is fixed order (0..N-1). The **first** agent in this order that has `current_mode=='rl_rvo'` and `detect_deadlock(agent_id,...)==True` triggers; `get_deadlock_participants(agent_id,...)` is then built with that agent as **seed**. So the participant set is the connected component containing that seed, not necessarily the “true” geometric deadlock cluster. | [ ] Confirmed / [ ] Rejected / [x] Inconclusive |
| H1b | **Conflict graph ≠ true deadlock set** | Participants come from `_build_local_conflict_graph(seed_id, ...)` → `_extract_component(graph, seed_id)` → filter by goal tolerance → prioritize and clip. The graph includes any “active” (slow, not progressing) agent within `COMMUNICATION_RANGE` and TTC/dmin edges. So the set can be larger or different from the visually stuck group (e.g. includes agents that are slow but not in the same bottleneck). **Runtime evidence**: changing `COMMUNICATION_RANGE` in `deadlock_config` (e.g. from 10.0 to a smaller value) clearly changes which agents enter the deadlock event and PAR participants in `par_init_ep000_step033` JSON and `debug-88fc1d.log`. | [x] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H1c | **MIN_PAR_PARTICIPANTS=4** | With default `MIN_PAR_PARTICIPANTS=4`, if the “true” deadlock has only 2–3 agents, `get_deadlock_participants` returns `[]` and no MAPF. So either no one enters MAPF, or the first agent that triggers has a large enough component (≥4) that may include extra agents. | [x] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H1d | **Neighbor state scope** | `agent_neighbor_states` is the 2.0 m (COLLISION_WARNING_DISTANCE) view; participant selection uses COMMUNICATION_RANGE (e.g. 7.0 m) and full `agent_states`. So detection “neighbors” and participant graph are **not** defined by `agent_neighbor_states`; see §3.1.1. If the 2.0 m vs 7.0 m or the MAPF–MAPF exclusion affects velocity history or other logic, detection could diverge from the intended local deadlock. **Runtime evidence so far**: changing COLLISION_WARNING_DISTANCE (2.0→1.0) produced no observable change in deadlock events, consistent with its low impact relative to COMMUNICATION_RANGE. | [ ] Confirmed / [ ] Rejected / [x] Inconclusive |

**Attempts** (fill as you try):

| # | What was tried | Result (success / failure / inconclusive) | Evidence (log lines / behavior) |
|---|----------------|-------------------------------------------|-----------------------------------|
| 1 | Change `COMMUNICATION_RANGE` in `deadlock_config` (e.g. reduce from 10.0 to a smaller value) and rerun `policy_test_long_range_with_par.py`. | success (for probing H1b) | PAR debug logs `par_init_ep000_step033.json/.png` and `debug-88fc1d.log` show that deadlock events and PAR participant sets change significantly when `COMMUNICATION_RANGE` is altered, confirming that the conflict graph / participant set is controlled by COMMUNICATION_RANGE rather than the 2.0m neighbor snapshot. |
| 2 | Change `COLLISION_WARNING_DISTANCE` (2.0→1.0) and rerun the same test. | inconclusive for P1 (low impact) | Behavior and deadlock events show no visible change; consistent with design that `agent_neighbor_states` (2m view) is not used for building the conflict graph or main neighbor sets in DeadlockDetector. |
| 3 | Add structured participant-selection traces (`deadlock_check_traces`) and rerun the fixed debug command. | success (for Issue 2 instrumentation) | The first detected event in a validated debug run shows `step=30`, `agent_id=1`, `loop_index=1`, `clipped_nodes=[1, 2]`, `returned_participants=[]`, `empty_return_reason=below_min_participants`, `solver_invoked=false`. This distinguishes "deadlock detected" from "MAPF started" and provides direct evidence that min-participant gating is active in P1. |
| 4 | Run a bounded participant sweep over `COMMUNICATION_RANGE in {3.0, 4.0, 5.0}` and `MIN_PAR_PARTICIPANTS in {2, 4}` via `bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh`. | success | Sweep summary at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/20260315_174737/summary/participant_sweep_summary.md` shows that `MIN_PAR_PARTICIPANTS=4` yields first-trigger `returned_participants=[]` in all three communication-range settings, while `MIN_PAR_PARTICIPANTS=2` yields first-trigger `returned_participants=[1,2]` or `[2,1]` and solver invocation at step 30. The trigger seed changes between `COMMUNICATION_RANGE=3.0` (`agent_id=2`) and `COMMUNICATION_RANGE>=4.0` (`agent_id=1`), confirming sensitivity to both parameters. |
| 5 | Make participant gating explicit and test-local for `policy_test_long_range_with_par.py`: add `deadlock_par_test_local.json`, set `MIN_PAR_PARTICIPANTS=2`, update debug config to match, rerun `bash scripts/run_all.sh test-par-debug`. | success | Before the caller-side config change, the first detected event was `step=30`, `agent_id=1`, `returned_participants=[]`, `empty_return_reason=below_min_participants`, and the first solver run was delayed to `step=53` with participants `[3, 6, 7, 4]`. After the change, the first detected event at `step=30` immediately returns `[1, 2]` and invokes the solver at `step=30`, matching the smaller early bottleneck that was repeatedly exposed in Issue 3. |
| 6 | Regression-check the bounded participant sweep after the caller-side config change. | success | Sweep summary at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/20260315_180052/summary/participant_sweep_summary.md` matches the earlier Issue 3 matrix row-for-row, so the explicit caller-side config did not invalidate the bounded evidence-gathering workflow. |
| 7 | Extend the participant sweep summary with `total_deadlock_detections`, `total_par_executions`, and `total_mode_switches`, then rerun the bounded sweep. | success | Sweep summary at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/20260315_183754/summary/participant_sweep_summary.md` now separates repeated deadlock detection from actual PAR intervention and mode churn. The new columns show that lowering `MIN_PAR_PARTICIPANTS` to 2 reduces raw deadlock detections (`953 -> 68` at `COMMUNICATION_RANGE=3.0`, `939 -> 44` at `4.0`, `938 -> 38` at `5.0`) while increasing actual PAR executions (`1 -> 19`, `1 -> 25`, `1 -> 34`) and mode switches (`4 -> 61`, `4 -> 87`, `4 -> 118`). |
| 8 | Add structured `waypoint_progression_traces` for RL and MAPF modes, then rerun `bash scripts/run_all.sh test-par-debug`. | success | Latest debug run at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_191113/` writes 8000 `waypoint_progression_traces` records to the episode JSON, one per agent-step. The trace now captures `manager_type`, `waypoint_index_before/after`, `stay_steps_before/after`, thresholds, `advancement_reason`, whether `update_waypoint_history` was called, and any post-step `mapf_exit_reason`. Example evidence: agent 6 force-switched in RL mode at `step=23` (`index 1 -> 2`, `stay_steps 19 -> 0`, `distance_to_goal_before=0.4405`, `advancement_reason=force_switch`, `update_waypoint_history_called=true`), while the same agent later hit a MAPF stay failure at `step=195` (`index 2 -> 2`, `stay_steps 39 -> 40`, `distance_to_current_goal=1.1402`, `advancement_reason=none`, `mapf_exit_reason=stay_fail(>=40)`). |

---

### 3.2 P2: Agents (RL or MAPF) do not advance waypoints — stall until episode timeout

**Observed**: Some agents, in either RL or MAPF mode, stay on the same waypoint until the episode ends by `max_ep_len`.

**Hypotheses**:

| ID | Hypothesis | Rationale (code/design) | Status |
|----|------------|--------------------------|--------|
| H2a | **Reach threshold too strict** | `WaypointManager.update(pos)` advances `_index` only when `dist <= _reach_threshold`. If threshold is too small or goal is slightly off, the agent never “reaches” and never advances. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H2b | **PAR tuple: all must reach** | For PAR tuple execution, the group advances to the next tuple only when **all** participants reach the current row. If one agent never reaches (e.g. blocked, wrong goal, or policy doesn’t drive there), the whole group stalls and no one advances. | [x] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H2c | **RL policy does not drive toward overridden goal** | MAPF agents keep using the **same** RL policy; only `robot.goal` is overridden to the current MAPF waypoint. If the policy ignores or underweights goal (e.g. strong avoidance), the agent may not move toward the waypoint. | [ ] Confirmed / [ ] Rejected / [x] Inconclusive |
| H2d | **Force-switch disabled or too high** | WaypointManager can force-advance after `_force_switch_steps` on the same waypoint (except last). If `force_switch_enabled=False` or `_force_switch_steps` is very large, agents can stall indefinitely. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H2e | **Waypoint index not updated for MAPF** | `update_waypoint_history(agent_id, wp_idx)` is called only when `mode == 'rl_rvo'`. So waypoint-stuck trigger does not see MAPF agents’ waypoint progress; this does not directly explain stall but can affect when detection fires for others. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |

**Attempts**:

| # | What was tried | Result | Evidence |
|---|----------------|--------|----------|
| 1 | Add structured `waypoint_progression_traces` and rerun `bash scripts/run_all.sh test-par-debug`. | success | Latest debug run at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_191113/` records 8000 waypoint traces with `advancement_reason`, `stay_steps`, `update_waypoint_history_called`, and `mapf_exit_reason`, proving that per-agent progression can be reconstructed from the episode JSON. |
| 2 | Add `mapf_execution_traces`, `tuple_progression_traces`, and `stall_classification`, then rerun `bash scripts/run_all.sh test-par-debug`. | success | Latest debug run at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_123628/` records 3738 MAPF execution traces and 971 tuple traces. The episode-level classifier returns `primary_class=tuple_group_blocking`, with `tuple_blocked_steps=943`, `max_tuple_blocked_streak=495`, `dominant_blocking_participant_id=0`, while still preserving secondary goal-misalignment evidence (`goal_misalignment_steps=1309`, `max_goal_misalignment_streak=54`, dominant agent 5). |
| 3 | Run `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples` after the instrumentation change. | success | The tuple-focused validation script exits with code 0 after Issue 5 instrumentation, so the diagnostic additions did not break the standalone tuple workflow. |
| 4 | Align PAR tuple-group `reach_threshold` with the injected PAR waypoint managers and rerun `bash scripts/run_all.sh test-par-debug`. | success (partial mitigation) | Debug run at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_124442/` still times out, but the dominant stall moves from `tuple_index=2` to `tuple_index=6`, showing that the previous `0.3` tuple-row gate was stricter than the PAR manager tolerance (`0.5`) and was prematurely holding rows closed. |
| 5 | Add non-final tuple-row force-advance after a sustained blocked streak (`2 * FORCE_WAYPOINT_SWITCH_STEPS`) and rerun both validations. | success (bounded behavior change) | Latest debug run at `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_124841/` still times out, but tuple progression now reaches `tuple_index=11` instead of stalling at `2` or `6`, and the episode JSON records 14 `row_advancement_reason=blocked_force_switch` events. The tuple-focused validation script still exits with code 0 after the change. |

---

### 3.3 P3: Collisions (agent–agent and agent–obstacle)

**Observed**: Collisions occur in both RL and MAPF modes (robot–robot and robot–obstacle).

**Hypotheses**:

| ID | Hypothesis | Rationale (code/design) | Status |
|----|------------|--------------------------|--------|
| H3a | **No separate collision-free execution for MAPF** | MAPF agents use the **same** dynamics and RVO as RL: `_step_pure_rl(modified_action_list)`. Only speed cap and yielding are applied; there is no guaranteed collision-free execution of the MAPF path (e.g. no direct application of MAPF plan as velocities/positions). So the policy can still choose velocities that lead to collision. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H3b | **Obstacle representation mismatch** | PAR/CBS sub-map or occupancy grid may use different dilation or bounds than the simulator’s collision check. So the MAPF plan might pass through cells that are considered free in the planner but collide in `obs_move_reward_list` (e.g. with lines/circles/polygons). | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H3c | **Yielding / speed cap insufficient** | Non-PAR agents are scaled down only when within `NON_PAR_YIELD_RADIUS` of a PAR participant. If radius or scale is too weak, or PAR agents move into others’ paths, mutual collisions can still occur. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |
| H3d | **RVO/ORCA and MAPF goals conflict** | RL policy (RVO-shaped) tries to avoid others while going to goal. When many agents are in MAPF with overlapping or crossing paths, the decentralized RVO can produce velocities that deviate from the MAPF waypoints and cause collisions. | [ ] Confirmed / [ ] Rejected / [ ] Inconclusive |

**Attempts**:

| # | What was tried | Result | Evidence |
|---|----------------|--------|----------|
| 1 | *(none yet)* | | |

---

## 4. Cross-Cutting Notes

- **Order of operations in step**: (1) Deadlock check and MAPF trigger (per agent in loop); (2) waypoint progression and goal alignment for all agents; (3) PAR tuple goal override (if active); (4) speed cap and yielding; (5) `_step_pure_rl` (dynamics + RVO + collision); (6) PAR tuple index advance and per-agent MAPF exit.
- **Contract vs implementation**: `deadlock_resolution.contract.md` states that if `get_deadlock_participants` returns fewer than `MIN_PAR_PARTICIPANTS`, the env does not start MAPF. The design doc says default `MIN_PAR_PARTICIPANTS=4`; the test does not override it.
- **Dead code**: In `deadlock_detector.get_deadlock_participants`, after `if len(participants) < min_participants: return []` there is an unreachable block (single-agent fallback that adds a closest neighbor); it never runs because of the `return []`.

---

## 5. Next Steps (to be filled during debugging)

### 5.1 Canonical Debug Reproduction

- **Fixed command**: `bash scripts/run_all.sh test-par-debug`
- **Direct equivalent**: `python rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py --enable_deadlock_resolution --long_range --robot_number 8 --debug_run --deadlock_config_file rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json`
- **Environment**: activate `conda activate rl_rvo_nav` before running the fixed command
- **Test-local debug config**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json`
- **Artifact root**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/<timestamp>/`
- **Debug-run behavior**: disable the test visualizer so the run exits after writing machine-readable artifacts
- **Expected artifact bundle**:
  - `artifacts/run_manifest.json`
  - `artifacts/result_long_range_with_par.txt`
  - `deadlock_logs/<timestamp>/...`
  - `test_logs/<timestamp>_test_with_par/...`
  - `figures/`
  - `gifs/`

### 5.2 Participant-Selection Trace Fields

- `deadlock_logs/<timestamp>/episode_data_*.json` now includes `episode_data.deadlock_check_traces`, one event per RL-agent deadlock check.
- Each event records `step`, `agent_id`, `loop_index`, `current_mode`, `trigger_source`, cooldown state, min-history state, and `average_velocity`.
- Each event also records the communication-range decision inputs: `communication_range_neighbors`, `slow_neighbors`, `non_progress_neighbors`, `risk_neighbors`.
- For detected events, the participant-construction path is preserved as `conflict_graph_adjacency`, `component_nodes`, `filtered_nodes`, `prioritized_nodes`, `clipped_nodes`, `returned_participants`, and `empty_return_reason`.
- Env-side execution context is appended as `solver_cache_key`, `solver_invoked`, `solver_cache_hit`, `solver_skipped_reason`, and `valid_participants`.

### 5.3 Initial Instrumentation Finding

- In the first detected trace from a validated debug run, deadlock was detected at `step=30` for `agent_id=1`, but MAPF did not start because the participant path reduced to `clipped_nodes=[1, 2]` and the final result was `returned_participants=[]` with `empty_return_reason=below_min_participants`.

### 5.4 Participant Sweep Command and Summary

- **Sweep command**: `bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh`
- **Latest sweep root**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/20260315_183754/`
- **Latest sweep summaries**:
  - `summary/participant_sweep_summary.md`
  - `summary/participant_sweep_summary.csv`
  - `summary/participant_sweep_summary.json`
- **Added counters**:
  - `total_deadlock_detections`: total deadlock detections recorded in the episode-level `stats`
  - `total_par_executions`: total PAR solver executions recorded in the same `stats`
  - `total_mode_switches`: total RL<->MAPF mode switches recorded in the same `stats`

| config_id | COMMUNICATION_RANGE | MIN_PAR_PARTICIPANTS | first_trigger_step | trigger_seed | returned_participants | empty_returns_due_to_min_gate | total_deadlock_detections | total_par_executions | total_mode_switches | episode_outcome | first_solver_step | first_solver_participants |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|
| `cr3p0_mp2` | 3.0 | 2 | 30 | 2 | `[1, 2]` | 49 | 68 | 19 | 61 | timeout | 30 | `[1, 2]` |
| `cr3p0_mp4` | 3.0 | 4 | 30 | 2 | `[]` | 952 | 953 | 1 | 4 | timeout | 53 | `[3, 6, 7, 4]` |
| `cr4p0_mp2` | 4.0 | 2 | 30 | 1 | `[1, 2]` | 19 | 44 | 25 | 87 | timeout | 30 | `[1, 2]` |
| `cr4p0_mp4` | 4.0 | 4 | 30 | 1 | `[]` | 938 | 939 | 1 | 4 | timeout | 53 | `[3, 6, 7, 4]` |
| `cr5p0_mp2` | 5.0 | 2 | 30 | 1 | `[1, 2]` | 4 | 38 | 34 | 118 | timeout | 30 | `[1, 2]` |
| `cr5p0_mp4` | 5.0 | 4 | 30 | 1 | `[]` | 937 | 938 | 1 | 4 | timeout | 53 | `[3, 6, 7, 4]` |

- **Conclusion**: participant selection is sensitive to both `COMMUNICATION_RANGE` and `MIN_PAR_PARTICIPANTS` in this bounded sweep.
- **Interpretation**:
  - `MIN_PAR_PARTICIPANTS=4` consistently suppresses the first detected two-agent component, which now moves H1c from inconclusive to confirmed.
  - `COMMUNICATION_RANGE` changes the earliest trigger seed (`agent_id=2` at 3.0 versus `agent_id=1` at 4.0/5.0) and strongly changes how often detections are dropped by the min-participant gate, which strengthens H1b.
  - The added counters show that `MIN_PAR_PARTICIPANTS=2` does not increase raw deadlock detections in this bounded test. Instead, it lowers repeated detections while increasing actual PAR executions and mode switches, which quantifies the tradeoff between passive detection churn and active MAPF intervention.

### 5.5 Issue 8 Config-Local Fix

- **Chosen caller-local fix**: keep `COMMUNICATION_RANGE=4.0` and make `MIN_PAR_PARTICIPANTS=2` explicit for this integration test only.
- **Non-debug config path**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_par_test_local.json`
- **Debug config path**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json`
- **Why this is the minimal change**:
  - the Issue 3 sweep showed that the first repeated deadlock component is a two-agent set at `step=30`
  - lowering `MIN_PAR_PARTICIPANTS` fixes that gating directly without changing conflict-graph construction or global defaults
  - `COMMUNICATION_RANGE=4.0` was retained because it already produces the same early two-agent set as 5.0 while remaining closer to the previous test-local setting

### 5.6 Before/After Evidence for Issue 8

- **Before (baseline debug run, old debug config with `MIN_PAR_PARTICIPANTS=4`)**
  - first detected event: `step=30`, `agent_id=1`, `returned_participants=[]`, `empty_return_reason=below_min_participants`
  - first solver event: `step=53`, `valid_participants=[3, 6, 7, 4]`
- **After (current debug run, explicit caller-local gating with `MIN_PAR_PARTICIPANTS=2`)**
  - run bundle: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_180014/`
  - first detected event: `step=30`, `agent_id=1`, `returned_participants=[1, 2]`, `empty_return_reason=None`
  - first solver event: `step=30`, `valid_participants=[1, 2]`
- **Regression check**
  - rerunning `bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh` produced `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/20260315_180052/summary/participant_sweep_summary.md`
  - the bounded sweep table is unchanged relative to the Issue 3 baseline, so the explicit caller-local fix did not distort the evidence-gathering harness

### 5.7 Issue 4 Waypoint Progression Instrumentation

- **Latest debug run**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_191113/`
- **Episode JSON with traces**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_191113/deadlock_logs/20260315_191113/episode_data_0_20260315_191128.json`
- **New debug field**: `episode_data.waypoint_progression_traces`
  - one record per agent per step in the deadlock-enabled step path
  - each record carries `mode`, `manager_type`, `waypoint_index_before/after`, `stay_steps_before/after`, `reach_threshold`, `force_switch_enabled`, `force_switch_steps`, `current_goal`, `distance_to_current_goal`, `advancement_reason`, `update_waypoint_history_called`, `update_waypoint_history_index`, and `mapf_exit_reason`
- **Validation command**:
  - `bash scripts/run_all.sh test-par-debug`
  - `rg -n "\"advancement_reason\"" rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260315_191113/deadlock_logs -g '*.json'`
- **Representative evidence**:
  - agent 6 shows a concrete RL-side force switch at `step=23`: `waypoint_index_before=1`, `waypoint_index_after=2`, `stay_steps_before=19`, `stay_steps_after=0`, `reach_threshold=0.2`, `force_switch_steps=20`, `distance_to_goal_before=0.4405`, `advancement_reason=force_switch`, `update_waypoint_history_called=true`
  - the same agent later shows a MAPF-side stall exit at `step=195`: `manager_type=par_manager`, `waypoint_index_before=2`, `waypoint_index_after=2`, `stay_steps_before=39`, `stay_steps_after=40`, `reach_threshold=0.5`, `distance_to_current_goal=1.1402`, `advancement_reason=none`, `update_waypoint_history_called=false`, `mapf_exit_reason=stay_fail(>=40)`
  - at timeout, agent 6 remains in MAPF with a measurable stall context at `step=1000`: `waypoint_index_after=8`, `stay_steps_after=418`, `distance_to_current_goal=3.8283`, `advancement_reason=none`, `mapf_exit_reason=null`
- **Interpretation**:
  - the new trace cleanly separates RL-side waypoint progression from MAPF-side progression
  - force-switch and stay-fail behavior are now directly observable from the episode JSON rather than inferred from console prints
  - this instrumentation is sufficient for Issue 5 to distinguish per-agent progression bugs from tuple/group-level blocking

### 5.8 Issue 5 MAPF Goal Alignment and Tuple Blocking

- **Latest debug run**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_123628/`
- **Episode JSON with traces**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_123628/deadlock_logs/20260316_123628/episode_data_0_20260316_123643.json`
- **New debug fields**:
  - `episode_data.mapf_execution_traces`
  - `episode_data.tuple_progression_traces`
  - `episode_data.stall_classification`
- **Validation commands**:
  - `bash scripts/run_all.sh test-par-debug`
  - `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`
- **Measured outcome**:
  - `stall_classification.primary_class = tuple_group_blocking`
  - `tuple_blocked_steps = 943`
  - `max_tuple_blocked_streak = 495`
  - `dominant_blocking_participant_id = 0`
  - `max_tuple_deficit_to_row_completion = 0.7200`
  - secondary MAPF alignment signal remains present: `goal_misalignment_steps = 1309`, `max_goal_misalignment_streak = 54`, `dominant_goal_misalignment_agent_id = 5`
- **Representative tuple-blocking evidence**:
  - blocked tuple trace at `step=31`, `tuple_index=1`, `blocking_participant_ids=[1]`, `max_deficit_to_row_completion=0.1957`, `all_reached=false`
  - blocker histogram over the timeout episode is headed by agent 0 (`612` blocked-tuple appearances), followed by agents 2 (`112`), 4 (`60`), 1 (`59`), and 5 (`51`)
- **Representative MAPF alignment evidence**:
  - for agent 6 at `step=1000`, `tuple_group_active=true`, `tuple_index=2`, `distance_to_current_goal_before=0.1717`, `distance_to_current_goal_after=0.1717`, `distance_delta_to_current_goal=0.0`, `consecutive_non_progress_steps=550`, but `goal_misaligned=false`, `forward_projection_after_cap=0.2878`, and `action_angle_to_goal_deg=16.40`
  - this means the agent is not obviously steering away from its current goal; instead, it remains in a no-progress state while the tuple row still has blockers elsewhere
- **Interpretation**:
  - H2b is now the primary confirmed explanation for the reproduced timeout episode
  - H2c is not rejected, because goal-misalignment traces do exist, but the measured primary class in the reproduced timeout is tuple-group blocking rather than poor goal tracking
  - this evidence is now strong enough to justify Issue 11 as the next fix path before attempting a per-agent waypoint-manager fix

### 5.9 Issue 11 Tuple-Group Progression Fix

- **Code path changed**: `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- **Behavior change 1**: tuple-group `reach_threshold` is now aligned with the injected PAR waypoint managers instead of staying at the tighter long-range default
  - before the fix, the reproduced run used `par_manager.reach_threshold=0.5` while tuple-row completion still used `reach_threshold=0.3`
  - this mismatch let agents be "close enough" for their per-agent PAR manager while the synchronized tuple row still refused to advance
- **Behavior change 2**: non-final tuple rows now force-advance after `2 * FORCE_WAYPOINT_SWITCH_STEPS` blocked steps
  - the row-level trace field `row_advancement_reason=blocked_force_switch` records these forced advances in the episode JSON
  - the final row is still not skipped; the change only applies when a next tuple exists
- **Behavior change 3**: final-row release scaffolding now exists, but it is intentionally narrower than the non-final force-advance path
  - tuple groups now carry both `blocked_force_switch_steps` and `final_row_release_steps`
  - when the active tuple row is the true final row, the env may release the whole PAR tuple group back to RL after `FORCE_WAYPOINT_SWITCH_STEPS` blocked steps, recording `row_advancement_reason=final_row_release` and `mapf_exit_reason=PAR tuple group released at final row`
- **Validation runs**:
  - threshold-alignment-only run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_124442/`
  - threshold-alignment + non-final row force-advance run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_124841/`
  - threshold-alignment + non-final row force-advance + final-row release scaffolding run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_133848/`
  - tuple-focused regression: `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples` (exit code 0 after the fix)
- **Before/after blocker summary**:
  - before Issue 11 (Issue 5 baseline at `20260316_123628`): timeout episode remained primarily blocked at `tuple_index=2`, with `tuple_blocked_steps=943`, `max_tuple_blocked_streak=495`, and dominant blocker `agent_id=0`
  - after threshold alignment (`20260316_124442`): the timeout still occurs, but progression reaches `tuple_index=6` before the dominant blocker shifts to `agent_id=6`
  - after non-final row force-advance (`20260316_124841`): the timeout still occurs, but progression reaches `tuple_index=11`, and the episode JSON records `blocked_force_switch_count=14`
  - after adding final-row release scaffolding (`20260316_133848`): the measured outcome is unchanged because the active stuck row is `tuple_index=11` out of `total_tuples=15`, so the new `final_row_release` branch is never entered in this reproduced episode
- **Latest measured outcome (`20260316_124841`)**:
  - `stall_classification.primary_class = tuple_group_blocking`
  - `tuple_blocked_steps = 919`
  - `max_tuple_blocked_streak = 428`
  - `dominant_blocking_participant_id = 6`
  - `stay_fail_exit_count = 11`
  - tuple rows visited: `0..11`
- **Latest measured outcome (`20260316_133848`)**:
  - `stall_classification.primary_class = tuple_group_blocking`
  - `tuple_blocked_steps = 919`
  - `max_tuple_blocked_streak = 428`
  - `dominant_blocking_participant_id = 6`
  - `stay_fail_exit_count = 11`
  - the stuck row remains `tuple_index=11`, but the trace now makes clear that this is **not** the final row for this group (`total_tuples=15`, `final_row_release_steps=20`)
  - no `row_advancement_reason=final_row_release` event is recorded in the episode JSON
- **Interpretation**:
  - the Issue 11 fix materially improves tuple progression and removes the earlier strict-threshold dead stop
  - the fix does **not** fully eliminate timeout in the reproduced integration episode; tuple-group blocking remains the primary measured stall class even after non-final row force-advance
  - the new evidence corrects an earlier assumption: the remaining blocker in the reproduced run is a **late non-final row** blocker, not yet a true final-row barrier
  - residual risk therefore remains around late-row blockers whose remaining episode budget is shorter than the current non-final `blocked_force_switch_steps`; this issue is best described as a bounded mitigation with clear evidence, not a full end-to-end resolution of P2

### 5.10 Issue 6 Collision Context Instrumentation

- **Code paths changed**:
  - `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py`
- **Behavior change**:
  - no collision or reward semantics changed
  - when deadlock `DEBUG_MODE=True`, the env now stores a per-step action-adjustment context (`raw_action`, `modified_action`, speed-cap application, yielding application, solver/participant metadata) and emits a structured `collision_events` record if `observation_reward()` sees a terminal collision
- **New collision event fields**:
  - `step`, `collision_type`, `robot_id`, `robot_mode`, `solver_type`, `participant_ids`, `tuple_index`, `manager_type`
  - `robot_position`, `robot_velocity`, `robot_goal`, `waypoint_index`
  - `raw_action`, `modified_action`, `speed_cap_applied`, `speed_cap_value`, `yield_applied`, `yield_scale`
  - `exclude_par_neighbors_in_rvo`, `neighbor_count_used_by_rvo`, `vo_flag`, `min_exp_time`, `terminal_collision_flag`
  - plus `other_*` fields for robot-robot collisions or obstacle metadata for robot-obstacle collisions
- **Validation runs**:
  - static check: `python -m py_compile rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
  - static check: `python -m py_compile rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py`
  - multi-episode debug run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_134700/`
- **Measured outcome (`20260316_134700`)**:
  - `episode_data_0_20260316_134715.json`: `collision_events=[]`
  - `episode_data_1_20260316_134728.json`: `collision_events=[]`
  - `episode_data_2_20260316_134745.json`: `collision_events=[]`
  - all three sampled episodes still ended as timeout-heavy runs rather than collision terminations
- **Interpretation**:
  - the instrumentation path is now in place and persisted in episode JSON
  - the current reproduced PAR debug workflow did **not** exercise a collision case in these three sampled episodes, so Issue 6 is implemented but only partially validated
  - the next collision-specific step should use either a known colliding seed/config or the planner-vs-simulator overlay workflow once a new collision episode is reproduced

### 5.11 Collision-Specific Repro Search

- **Code path changed**: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`
- **Behavior change**:
  - the test script now accepts `--seed`, but keeps the historical default `42`
  - this makes collision repro search and later fixed collision runs auditable in `run_manifest.json`
- **Search result summary**:
  - baseline debug config + `seed=1..8`: all sampled runs timed out, `collision_events=[]`
  - aggressive collision probe config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_aggressive.json`:
    - `seed=2`: robot-robot collision reproduced
    - `seed=3`: robot-robot collision reproduced
    - `seed=5`: robot-robot collision reproduced
    - `seed=6`: robot-robot collision reproduced
    - `seed=1,4,7,8`: timeout
- **Stable repro chosen for follow-up**:
  - config: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_aggressive.json`
  - seed: `3`
  - fixed entry point: `bash scripts/run_all.sh test-par-collision-debug`
  - repeated validation bundles:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_recheck_seed3/20260316_135831/`
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_recheck_seed3/20260316_135836/`
  - repeated outcome:
    - both reruns end in the same `robot_robot` collision at `step=230`
    - both reruns record the same collision pair `(robot_id=3, other_robot_id=7)`
- **Representative collision evidence**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_recheck_seed3/20260316_135831/deadlock_logs/20260316_135831/episode_data_0_20260316_135835.json`
  - the structured `collision_events[0]` record shows:
    - RL agent `3` collided with MAPF agent `7`
    - collision type: `robot_robot`
    - collision step: `230`
    - `other_participant_ids=[3, 7]`
    - `exclude_par_neighbors_in_rvo=true`
    - `yield_applied=false`
    - `speed_cap_applied=false` on the RL side
    - `vo_flag=true`, `min_exp_time=0`
- **Interpretation**:
  - the repo now has a stable, fixed collision repro path for Issue 6/7 follow-up work
  - the most direct observed collision class so far is mixed-mode robot-robot collision (`rl_rvo` vs `mapf`), not obstacle collision
  - this evidence strengthens H3a/H3d and gives a concrete seed/config for later overlay or mitigation work

### 5.12 Issue 7 Planner-vs-Simulator Overlay Status

- **Implementation status**:
  - `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py` now builds a dedicated overlay payload for `robot_obstacle` collision events only.
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py` now saves planner-vs-simulator overlay artifacts under:
    - `deadlock_logs/<timestamp>/collision_overlays/*.png`
    - `deadlock_logs/<timestamp>/collision_overlays/*.json`
  - each overlay JSON includes:
    - planner grid extent, resolution, dilation iteration count, and occupancy snapshot
    - collision point, planner cell index, occupied/free classification, and nearest occupied planner cell
    - the best available continuous path/waypoint list for the colliding agent
    - simulator obstacle geometry when the collision reports one explicitly (`circular`, `line`, `polygon`)
- **Fixed repro validation**:
  - reran `bash scripts/run_all.sh test-par-collision-debug`
  - latest bundle: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_152517/`
  - latest episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_152517/deadlock_logs/20260316_152517/episode_data_0_20260316_152520.json`
  - result: `collision_events` contains one collision, but it is still `robot_robot`, so `planner_simulator_collision_overlays=[]`
- **Bounded obstacle-collision search on the same collision config family**:
  - used the same aggressive collision probe config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_aggressive.json`
  - searched `seed=1..20`
  - summary artifact: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7/seed_search_20260316_152832.json`
  - observed outcomes:
    - `robot_robot` collisions for `seed=2,3,5,6,9,11,12,13,14,15,16,17,18,19`
    - `timeout` / no collision for `seed=1,4,7,8,10,20`
    - **no** `robot_obstacle` collision found in this bounded search
- **Current conclusion**:
  - the Issue 7 artifact generator is now implemented and ready
  - the currently fixed repro path remains a robot-robot collision repro, not an obstacle collision repro
  - H3b (`Obstacle representation mismatch`) remains **inconclusive** because no reproduced episode has yet exercised the new obstacle overlay path
  - until a real `robot_obstacle` episode is reproduced, Issue 7 is best treated as **implemented but awaiting obstacle-class validation**

### 5.13 Stable `robot_obstacle` Repro for Issue 7

- **Obstacle-focused probe configs added**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_safe_interagent.json`
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_mixed_mode.json`
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_wide_group.json`
  - tighter follow-up variants:
    - `collision_probe_obstacle_safe_interagent_tight.json`
    - `collision_probe_obstacle_mixed_mode_tight.json`
- **Why the successful probe was different from the robot-robot repro**:
  - the successful obstacle repro did **not** use the earlier aggressive mixed-mode collision config
  - instead, it restored inter-agent avoidance:
    - `EXCLUDE_PAR_NEIGHBORS_IN_RVO=false`
    - `NON_PAR_YIELDING_ENABLED=true`
  - this reduced early robot-robot failures and allowed one PAR-controlled agent to remain in the bottleneck long enough to hit the lower polygon wall
- **Search evidence saved**:
  - obstacle-search rankings: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7/obstacle_clearance_rankings_20260316_160431.json`
  - the ranking also shows several near-miss samples with positive clearance (for example `seed=8` with net clearance `0.01899 m`) that explain why earlier obstacle-search attempts looked close but did not collide
- **First confirmed obstacle-collision run**:
  - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7_obstacle_runs/safe_interagent/20260316_155415/`
  - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7_obstacle_runs/safe_interagent/20260316_155415/deadlock_logs/20260316_155415/episode_data_0_20260316_155430.json`
  - collision summary:
    - `collision_type=robot_obstacle`
    - `step=499`
    - `robot_id=1`
    - `robot_mode=mapf`
    - `solver_type=par`
    - `participant_ids=[1, 7, 6]`
    - `obstacle_type=polygon`
    - obstacle edge reported by simulator: `[4.9, 0.0, 4.9, 4.0]`
- **Overlay artifact generated successfully**:
  - PNG: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7_obstacle_runs/safe_interagent/20260316_155415/deadlock_logs/20260316_155415/collision_overlays/collision_overlay_ep000_step499_robot01_n001.png`
  - JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7_obstacle_runs/safe_interagent/20260316_155415/deadlock_logs/20260316_155415/collision_overlays/collision_overlay_ep000_step499_robot01_n001.json`
  - overlay result:
    - `classification=planner_occupied_sim_collision`
    - `planner_cell_occupied=true`
    - `grid_row=6`, `grid_col=9`
- **Repeated validation**:
  - fixed recheck root: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7_obstacle_recheck_seed19/`
  - summary file: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_seed_probe_issue7/obstacle_recheck_seed19_20260316_160537.json`
  - attempt 1:
    - run dir: `.../20260316_160507/`
    - `collision_type=robot_obstacle`
    - `collision_step=499`
    - `robot_id=1`
    - `overlay_count=1`
  - attempt 2:
    - run dir: `.../20260316_160523/`
    - `collision_type=robot_obstacle`
    - `collision_step=499`
    - `robot_id=1`
    - `overlay_count=1`
- **New fixed entry point**:
  - `bash scripts/run_all.sh test-par-obstacle-collision-debug`
  - direct equivalent:
    - `python rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py --enable_deadlock_resolution --long_range --robot_number 8 --debug_run --seed 19 --reach_threshold 0.15 --deadlock_config_file rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_safe_interagent.json`
- **Updated interpretation**:
  - Issue 7 is now exercised by a real obstacle-collision episode instead of a synthetic smoke test only
  - H3b (`Obstacle representation mismatch`) is no longer blocked on missing evidence
  - for this reproduced episode, the overlay says `planner_occupied_sim_collision`, which weakens the specific claim that the planner saw the impact cell as free
  - this does **not** yet fully reject H3b for all collision classes, but it makes planner/simulator free-vs-occupied mismatch less compelling for this specific stable repro

### 5.14 Issue 12 Collision Fix: Block PAR Force-Switch Shortcuts Through Occupied Planner Cells

- **Root-cause evidence from the stable obstacle sample**:
  - stable pre-fix repro:
    - config: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_safe_interagent.json`
    - command: `bash scripts/run_all.sh test-par-obstacle-collision-debug`
    - repeated pre-fix outcome: `robot_obstacle` at `step=499` for `robot_id=1`
  - overlay evidence from the pre-fix repro showed `planner_occupied_sim_collision`, not `planner_free_sim_collision`
  - the key per-agent waypoint evidence was **not** “planner missed the wall”; it was waypoint progression:
    - at `step=434`, agent `1` force-switched from waypoint index `3` toward next goal `[5.25, 4.25]` even though the straight segment from the agent position crossed occupied planner cell `(row=7, col=9)`
    - at `step=494`, the same agent force-switched again toward `[5.75, 4.25]`
    - the collision at `step=499` then happened while MAPF execution was chasing a goal on the far side of the lower wall
- **Implemented fix**:
  - added a narrow env-side force-switch guard for `mapf + par_manager` only
  - before allowing `WaypointManager` to force-advance, `gym_env` now checks whether the straight segment from the current agent position to the *next* PAR waypoint crosses an occupied cell in the planner-aligned occupancy grid
  - if the segment crosses an occupied cell, that force-switch is blocked and logged as:
    - `advancement_reason=blocked_force_switch`
    - `force_switch_blocked_reason=occupied_segment_to_next_waypoint`
    - plus the offending `next_goal` and occupied cell metadata
  - normal distance-based waypoint advancement is unchanged, and the generic long-range manager path is unchanged
- **Post-fix validation**:
  - latest fixed run:
    - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_161715/`
    - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_161715/deadlock_logs/20260316_161715/episode_data_0_20260316_161739.json`
  - outcome on the same seed/config:
    - `collision_events=[]`
    - `planner_simulator_collision_overlays=[]`
    - episode now ends by timeout at `step=1000`, not by obstacle collision at `step=499`
  - explicit blocked-force-switch evidence for agent `1`:
    - `agent1_blocked_force_switch_count=187`
    - first blocked event at `step=434`
    - representative trace:
      - `current_goal=[4.75, 4.25]`
      - `force_switch_guard_next_goal=[5.25, 4.25]`
      - `force_switch_guard_occupied_cell={"row": 7, "col": 9, "center": [4.75, 3.75], "sample_point": [4.602271204231963, 3.96625309415914]}`
  - the original collision window no longer shortcuts across the wall:
    - at `step=494`, agent `1` is still on waypoint index `3` with `advancement_reason=blocked_force_switch`
    - by `step=499`, the agent is following a new PAR goal on the safe side (`current_goal=[4.25, 3.75]`) instead of the old post-force-switch goal on the far side of the wall
- **Regression checks**:
  - `bash scripts/run_all.sh test-par-obstacle-collision-debug`
    - completed with timeout instead of the previous `robot_obstacle`
  - `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`
    - PAR and rule-based outputs still completed; CBS still hit the pre-existing local multiprocessing permission error (`PermissionError: [Errno 1] Operation not permitted`)
- **Interpretation after the fix**:
  - the targeted obstacle-shortcut collision class is removed for the fixed seed/config
  - the dominant remaining failure on this path is now timeout / tuple-group blocking, not obstacle collision
  - this strengthens the conclusion that the stable obstacle sample was primarily an execution/path-following issue, not a planner-vs-simulator free-cell mismatch

### 5.15 Issue 11 Follow-up: Prevent Active PAR Tuple Groups from Being Reset by New PAR Triggers

- **Root-cause evidence from the first post-collision-fix timeout**:
  - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_161715/`
  - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_161715/deadlock_logs/20260316_161715/episode_data_0_20260316_161739.json`
  - key symptoms:
    - `mode_switches=67`
    - `par_executions=21`
    - `stall_classification.primary_class=tuple_group_blocking`
    - `mapf->mapf` mode switches occurred `21` times
  - tuple reset evidence:
    - `tuple_progression_traces` showed only `tuple_index=0` and `tuple_index=1` dominating even though some groups had long waypoint tuples
    - participant sets were repeatedly replaced while a tuple group was already active, for example:
      - step `35`: `[1,3] -> [0,5]`
      - step `36`: `[0,5] -> [2,3,7]`
      - step `305`: previous `tuple_index=1` group `[3,4]` was replaced by `[0,1,2,7]`
    - the mode-switch log showed the matching `mapf->mapf` reinjections, for example agent `3` at step `36`, agent `2` at step `305`, and multiple `mapf->mapf` reinjections again around steps `734-975`
- **Implemented fix**:
  - while `_par_tuple_group` is active, gym_env now skips new PAR solver invocations instead of allowing a new deadlock event to overwrite the active tuple group
  - the deadlock check trace now records:
    - `solver_skipped_reason=active_par_tuple_group_in_progress`
    - `active_tuple_group_participants`
    - `active_tuple_group_index`
  - this preserves the current tuple group as the single source of truth until it completes or releases
- **Validation on the retained fix**:
  - validated run:
    - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_162645/`
    - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_162645/deadlock_logs/20260316_162645/episode_data_0_20260316_162657.json`
  - before/after comparison:
    - `mode_switches`: `67 -> 10`
    - `par_executions`: `21 -> 3`
    - `mapf->mapf` mode switches: `21 -> 0`
  - tuple progression became stable instead of repeatedly resetting:
    - `tuple_progression_traces` now span meaningful indices up to `19`
    - dominant participant sets are coherent long-lived groups:
      - `[6,7]` for `614` tuple steps
      - `[1,2]` for `193` tuple steps
      - `[2,5]` for `164` tuple steps
  - the new skip trace is exercised heavily, which is expected:
    - `active_par_tuple_group_in_progress` appears `715` times in `deadlock_check_traces`
    - representative skipped check:
      - `step=44`
      - `returned_participants=[1,4]`
      - active tuple group still `[1,2]` at `active_tuple_group_index=2`
- **Remaining blocker after the fix**:
  - the run still times out at `step=1000`
  - the dominant remaining blocker is no longer tuple-group reset churn; it is row advancement within the surviving tuple groups
  - the clearest example is the final `[6,7]` group:
    - the group starts around `step=387`
    - rows `0-4` advance by `all_reached`
    - from row `5` onward, repeated `blocked_force_switch` advances move agent `6`'s target farther left while its distance to the current tuple target grows from roughly `0.86` to about `3.8`
  - this means the retained fix solved one root cause cleanly, but it also exposed a second, narrower one: `blocked_force_switch` can outrun the dominant lagging participant once the tuple group is stable
- **Status**:
  - keep this fix
  - do **not** keep the later experimental `blocked_large_deficit_release` branch; local validation showed it increased churn again (`mode_switches` and `par_executions` rose) without resolving the timeout
  - do **not** keep the later experimental same-blocker drift guard on non-final `blocked_force_switch`
    - attempted run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_164241/`
    - key regression:
      - `deadlock_detections: 1482 -> 2704`
      - `mode_switches: 10 -> 6`
      - `par_executions: 3 -> 2`
      - `max_tuple_blocked_streak: 587 -> 748`
    - interpretation:
      - the guard did stop the repeated `[6,7]` force-advance pattern, but it also left the active tuple group stuck for much longer, so the env accumulated far more RL deadlock detections while still timing out
    - retained diagnostic value:
      - `tuple_progression_traces` now expose `dominant_blocking_participant_id` and `dominant_blocking_distance_to_target`, which remain useful for the next late-row blocker investigation

### 5.16 Same-Group Replan Feasibility Probe for Late-Row Blockers

- **Purpose**:
  - test whether the env could replace `blocked_force_switch` with a same-participant PAR replan once a stable tuple group reaches a late-row blocker
  - this probe is diagnostic only: it runs during debug mode, writes structured results into `tuple_progression_traces[*].same_group_replan_probe`, and does **not** change runtime behavior
- **Validation run**:
  - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_165541/`
  - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_165541/deadlock_logs/20260316_165541/episode_data_0_20260316_165556.json`
  - stability check:
    - `deadlock_detections=1482`
    - `mode_switches=10`
    - `par_executions=3`
  - these match the retained baseline, so the probe itself did not perturb the main PAR execution path
- **Probe result**:
  - total probe attempts: `19`
  - probe status summary:
    - `insufficient_valid_participants=19`
  - participant-set breakdown:
    - `[1,2]`: `2`
    - `[2,5]`: `3`
    - `[6,7]`: `14`
- **Key late-row finding for `[6,7]`**:
  - every same-group probe failed because the PAR preparation produced only **one** valid participant instead of two
  - representative samples:
    - step `453`, tuple `5`: `valid_participants=[6]`, probe statuses = `agent 6 -> has_path=True, path_length=3, at_goal=False`; `agent 7 -> has_path=False, path_length=0, at_goal=False`
    - step `533`, tuple `7`: `valid_participants=[6]`, probe statuses = `agent 6 -> has_path=True, path_length=5, at_goal=False`; `agent 7 -> has_path=False, path_length=0, at_goal=False`
    - step `693`, tuple `11`: the same pattern persists; the lagging blocker still gets a path while the other tuple member remains invalid for same-group replan
- **Interpretation**:
  - a same-participant replan is **not currently a drop-in replacement** for `blocked_force_switch`
  - the limiting factor is not “PAR returns the wrong two-agent tuple”; it is that, at late-row blocker time, the same-group probe does not produce two valid participants at all
  - the next narrower question is therefore not “should the env apply same-group replan?” but “why does the PAR caller/coordinator treat the non-blocking tuple member as invalid in this state?”

- **Root-cause finding from code + trace alignment**:
  - the invalid participant is dropped before the solver-local actor set is built
  - `PAREnvironment.compute_goal_positions()` skips a participant entirely when its current position is closer than `grid_resolution / 2` to its current goal:
    - code: `rl_rvo_nav/deadlock_resolution/par_environment.py:976-981`
    - current debug config uses `grid_resolution=0.5`, so the skip threshold is `0.25`
  - this matches the `[6,7]` late-row blocker traces exactly:
    - at step `453`, `agent 7` is only `0.1234` away from its current tuple goal, which is below `0.25`
    - the dry-run same-group probe therefore produces `valid_participants=[6]`
    - probe status details at the same step show:
      - `agent 6 -> has_path=True, path_length=3, at_goal=False`
      - `agent 7 -> has_path=False, path_length=0, at_goal=False`
  - the same pattern persists through later forced rows (`step=493`, `533`, `573`, ...): the blocker still gets a path while the near-goal participant is silently omitted from `goal_positions`

- **Implication**:
  - the current PAR caller semantics treat “already near the current MAPF waypoint” as “omit from replan goals,” which is reasonable for a fresh local solve but breaks the assumption that a same-group replan can preserve an active synchronized tuple group
  - any future fix in this area should likely focus on the PAR goal-building contract for active tuple-group participants, not on adding more env-side force-switch heuristics

- **Follow-up probe after preserving near-goal participants**:
  - validation run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_170901/`
  - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_170901/deadlock_logs/20260316_170901/episode_data_0_20260316_170919.json`
  - this probe changed only the dry-run state passed to PAR: every active tuple participant kept its goal even when it was already within `grid_resolution / 2` of that goal
  - main debug-run metrics stayed stable:
    - `deadlock_detections=1482`
    - `mode_switches=10`
    - `par_executions=3`
  - result:
    - the probe still failed `19/19` times
    - `[6,7]` no longer failed as “only agent 6 is valid”; it failed as “both participants end up with no path”
  - solver artifact for the first `[6,7]` blocked row:
    - `par_debug/par_init_ep000_step453.json`
    - initialization shows:
      - `start_positions = {6: [12,11], 7: [11,19]}`
      - `goal_positions = {6: [10,11], 7: [11,19]}`
      - so `agent 7` is preserved, but it enters the solver with `start == goal`
    - solver output then reports:
      - `solution.success = False`
      - both trajectories empty
      - `solution.meta.starts_equal_goals = {'6': False, '7': True}`
      - `solution.meta.stats.solve_trace[0].agent = 7`
      - `solution.meta.stats.solve_trace[0].reason = 'no_path'`

- **Updated interpretation**:
  - preserving near-goal participants fixes the **caller-side omission**, but it exposes a **solver-side limitation**: the current PNR path-building flow does not successfully handle a same-group replan when one retained participant enters with `start == goal`
  - this means a production fix should probably not be “enable same-group replan as-is”
  - the narrower next target is either:
    - make PAR/PNR treat `start == goal` participants as valid stationary actors in this replan path, or
    - solve only the moving subset and recompose a synchronized tuple with explicit stationary participants outside the solver

### 5.17 Moving-Subset Replan Prototype with Stationary Recomposition

- **Purpose**:
  - test the second narrower option from Section 5.16 without changing runtime behavior:
    - solve only the moving subset with PAR
    - keep already-reached tuple participants stationary
    - recompose a synchronized tuple preview in full participant order
- **Implementation scope**:
  - debug-only
  - written into `tuple_progression_traces[*].same_group_replan_probe.moving_subset_replan_probe`
  - does **not** change the active tuple group or MAPF execution path
- **Validation run**:
  - run dir: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_171507/`
  - episode JSON: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_171507/deadlock_logs/20260316_171507/episode_data_0_20260316_171527.json`
  - baseline stability remained intact:
    - `deadlock_detections=1482`
    - `mode_switches=10`
    - `par_executions=3`
- **Result for the main late-row blocker `[6,7]`**:
  - the full-group probe still fails with `status=insufficient_valid_participants`
  - the moving-subset probe succeeds for all `14` late-row force-advance points:
    - `moving_subset_status_counts = {'moving_subset_tuple_available': 14}`
  - representative sample at step `453`, tuple `5`:
    - moving participants: `[6]`
    - stationary participants: `[7]`
    - valid moving participants: `[6]`
    - `same_moving_subset=True`
    - `has_tuple_group=True`
    - recomposed tuple preview:
      - `[[6.25, 5.75], [5.75, 9.75]]`
      - `[[5.75, 5.75], [5.75, 9.75]]`
      - `[[5.25, 5.75], [5.75, 9.75]]`
  - later rows keep the same pattern:
    - probe tuple counts range from `3` to `12`
    - recomposed tuple count mean is about `8.07`
- **Interpretation**:
  - this is the first positive evidence-backed path for replacing late-row `blocked_force_switch`
  - the problem is **not** that the moving blocker lacks a feasible PAR path
  - the problem is that the current full-group same-participant replan cannot include stationary tuple members cleanly, while the moving subset can
- **Implication for the next behavior change**:
  - the most credible bounded fix is now:
    - keep `blocked_force_switch` as the fallback
    - but, before forcing the row forward, attempt a debug-validated moving-subset PAR replan
    - if it succeeds, recompose stationary participants back into a synchronized tuple instead of discarding them or forcing the whole row forward blindly

### 5.18 First Runtime Candidate: Broad Moving-Subset Replan Application

- **Behavior change**:
  - `gym_env/gym_env/envs/ir_gym.py` now tries a real moving-subset PAR replan before falling back to non-final `blocked_force_switch`
  - when that replan succeeds:
    - movers receive fresh PAR waypoint managers
    - stationary participants are held on the current tuple target via env-side `stationary_hold_targets`
    - the tuple trace records `row_advancement_reason=moving_subset_replan`
    - the tuple trace also records `moving_subset_runtime_result`
- **Validation run**:
  - command:
    - `source /home/haoyiwang/anaconda3/etc/profile.d/conda.sh && conda activate rl_rvo_nav && bash scripts/run_all.sh test-par-debug`
  - run directory:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172532/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172532/deadlock_logs/20260316_172532/episode_data_0_20260316_172553.json`
- **Observed result**:
  - the candidate really did apply:
    - `moving_subset_replan_count=13`
    - `stationary_hold_count=323`
  - but it was too broad:
    - applied groups were `[(1,2) x3, (2,5) x1, (0,5) x9]`
  - global episode stats worsened sharply relative to the stable baseline:
    - baseline (`20260316_164432`): `deadlock_detections=1482`, `mode_switches=10`, `par_executions=3`
    - broad runtime candidate (`20260316_172532`): `deadlock_detections=2141`, `mode_switches=46`, `par_executions=25`
- **Interpretation**:
  - the mechanism itself is viable
  - but the initial runtime trigger window was too permissive and caused excessive PAR churn on groups that were not the original target late-row blocker

### 5.19 Narrowed Runtime Candidate: Late-Row Gate + Single-Use Budget

- **Behavior change refinement**:
  - narrowed the runtime candidate so that moving-subset replan is attempted only when:
    - the current tuple row is late enough in the sequence (`late_row_min_index = max(1, min(5, len(tuples) - 2))`)
    - the tuple lineage has not already consumed its one allowed moving-subset runtime replan
  - all earlier diagnostic probes remain in place
- **Validation run**:
  - command:
    - `source /home/haoyiwang/anaconda3/etc/profile.d/conda.sh && conda activate rl_rvo_nav && bash scripts/run_all.sh test-par-debug`
  - run directory:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172824/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172824/deadlock_logs/20260316_172824/episode_data_0_20260316_172844.json`
- **Observed result**:
  - global stats improved substantially versus the broad runtime candidate:
    - broad runtime candidate (`20260316_172532`): `2141 / 46 / 25`
    - narrowed runtime candidate (`20260316_172824`): `1483 / 22 / 11`
    - values shown as `deadlock_detections / mode_switches / par_executions`
  - row advancement distribution:
    - `row_advancement_reason_counts = {'all_reached': 39, 'none': 910, 'moving_subset_replan': 5, 'blocked_force_switch': 10, 'final_row_release': 4}`
  - moving-subset runtime status distribution:
    - `{'applied': 5, 'replan_budget_exhausted': 2, 'not_late_row': 8}`
  - applied runtime replan groups were reduced to:
    - `(1,2)` at step `162`, tuple `7/10`
    - `(5,6)` at step `337`, tuple `5/26`
    - `(0,5)` at step `495`, tuple `5/25`
    - `(0,6)` at step `701`, tuple `5/25`
    - `(6,7)` at step `825`, tuple `5/34`
  - the original target blocker now does receive the bounded runtime path:
    - for `(6,7)`, `pair67_applied_count=1`
    - representative runtime result at step `825`:
      - moving participants: `[6]`
      - stationary participants: `[7]`
      - recomposed preview:
        - `[[6.25, 5.75], [5.75, 9.75]]`
        - `[[5.75, 5.75], [5.75, 9.75]]`
        - `[[5.25, 5.75], [5.75, 9.75]]`
- **Comparison with the pre-runtime stable baseline**:
  - baseline (`20260316_164432`):
    - `deadlock_detections=1482`
    - `mode_switches=10`
    - `par_executions=3`
    - `primary_class=tuple_group_blocking`
    - `max_tuple_blocked_streak=587`
    - `dominant_blocking_participant_id=6`
    - `max_tuple_deficit_to_row_completion=3.6101`
  - narrowed runtime candidate (`20260316_172824`):
    - `deadlock_detections=1483`
    - `mode_switches=22`
    - `par_executions=11`
    - `primary_class=tuple_group_blocking`
    - `max_tuple_blocked_streak=160`
    - `dominant_blocking_participant_id=0`
    - `max_tuple_deficit_to_row_completion=1.8633`
- **Interpretation**:
  - this narrowed runtime candidate is a real **bounded mitigation**, not a full fix
  - positive signal:
    - it preserves the low deadlock-detection count of the stable baseline
    - it cuts the worst tuple-blocked streak dramatically (`587 -> 160`)
    - it reduces the worst tuple deficit substantially (`3.61 -> 1.86`)
    - it finally applies the intended moving-subset recomposition to `(6,7)`
  - remaining risk:
    - the episode still times out
    - `mode_switches` and `par_executions` remain clearly above the stable baseline
    - the dominant blocker family has shifted rather than disappeared
    - `stall_classification.primary_class` remains `tuple_group_blocking`
- **Current status**:
  - keep the narrowed runtime candidate as the current experimental behavior because it is materially better than the broad runtime attempt and measurably reduces tuple-stall severity
  - do **not** treat it as the final resolution of P2
  - the next debugging step should explain why the residual blocker shifts toward later `(0,5)` / `(0,6)` / `(6,7)` families even after the narrowed moving-subset runtime path succeeds

### 5.20 Residual Blocker Handoff Analysis After the Narrowed Runtime Candidate

- **Goal**:
  - explain why `tuple_group_blocking` remains the primary class even after the narrowed moving-subset runtime candidate reduces the worst blocked streak
- **Source run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172824/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_172824/deadlock_logs/20260316_172824/episode_data_0_20260316_172844.json`
- **Observed tuple-group handoff chain**:
  - the narrowed candidate no longer churns arbitrarily; instead, the episode now advances through a clearer handoff sequence of two-agent blockers:
    - `(1,2)` from steps `30 -> 277`
    - `(5,6)` from steps `278 -> 359`
    - `(0,5)` from steps `360 -> 516`
    - `(0,6)` from steps `517 -> 762`
    - `(6,7)` from steps `763 -> 1000`
  - mode switches match that same chain exactly:
    - `(1,2)` released at `277`
    - `(5,6)` entered at `278`
    - `(5,6)` released at `359`
    - `(0,5)` entered at `360`
    - `(0,5)` released at `516`
    - `(0,6)` entered at `517`
    - `(0,6)` completed at `762`
    - `(6,7)` entered at `763`
- **Most important residual pattern**:
  - the original target pair `(6,7)` does receive the new runtime path once:
    - `step=825`, `tuple_index=5/34`, `row_advancement_reason=moving_subset_replan`
    - moving participant: `[6]`
    - stationary participant: `[7]`
  - that recomposed group then progresses quickly through the first two rows:
    - `step=826`: row `0/3` becomes `all_reached`
    - `step=827`: row `1/3` becomes `all_reached`
  - but the final row is not actually resolved before release:
    - `step=847`: row `2/3` triggers `final_row_release`
    - at that same step:
      - dominant blocker is still agent `6`
      - `dominant_blocking_distance_to_target=0.9191`
      - agent `7` is still being held at the stationary target with distance `0.1799`
- **Waypoint and execution evidence around the `(6,7)` bounce-back**:
  - agent `6`:
    - `step=826`: advances by distance to goal `[5.75, 5.75]`
    - `step=827`: advances by distance to goal `[5.25, 5.75]`
    - `step=847`: still on that same goal with `distance_to_current_goal=0.9246`
  - agent `7`:
    - from `step=826` through `847`: `advancement_reason=stationary_hold`, holding goal `[5.75, 9.75]`
    - hold distance stays nearly constant at `0.1799`
  - MAPF execution traces show both agents are already in a non-progress regime before release:
    - agent `6`: consecutive non-progress count reaches `25` by step `847`
    - agent `7`: consecutive non-progress count reaches `83` by step `847`
- **Immediate post-release behavior**:
  - the pair is released from MAPF at `step=847`
  - for `step=848..850`, both agents immediately detect deadlock again, but the participant selection path returns `[]` with `empty_return_reason=below_min_participants`
  - at `step=851`, the same pair `(6,7)` re-enters PAR:
    - mode switch reason: `Deadlock detected by agent 6 (PAR solver)`
    - tuple progression restarts with a fresh `31`-row group
- **Interpretation**:
  - after the narrowed moving-subset runtime candidate, the dominant residual issue is no longer “same-group replan infeasible”
  - the stronger residual issue is now **release-to-retrigger bounce-back**, especially on the recomposed final row
  - in other words:
    - the env can now build a feasible moving-subset recomposition for `(6,7)`
    - but the current `final_row_release` policy is willing to release that recomposed group while the dominant mover is still materially short of the held final row
    - that release hands the same unresolved geometry back to RL, which then retriggers PAR almost immediately
- **Implication for the next bounded fix**:
  - the next fix should target final-row release semantics rather than broadening participant replan logic again
  - the most evidence-backed question is now:
    - how to prevent a recomposed final row from releasing back to RL while the dominant moving participant is still far enough away that the same pair will retrigger within a few steps

### 5.21 Rejected Experiment: Simple Release-Delay Gate for Recomposed Final Rows

- **Experimental change**:
  - for tuple groups that had already applied a runtime `moving_subset_replan`, final-row release was temporarily delayed from `final_row_release_steps` to `max(final_row_release_steps, blocked_force_switch_steps)`
  - the hypothesis was that this would prevent immediate `(6,7)` release-to-retrigger bounce-back
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_184408/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_184408/deadlock_logs/20260316_184408/episode_data_0_20260316_184427.json`
- **Observed result**:
  - global stats shifted to:
    - `deadlock_detections=1676`
    - `mode_switches=14`
    - `par_executions=7`
  - compared with the earlier narrowed runtime candidate (`20260316_172824`):
    - detections became worse (`1483 -> 1676`)
    - mode switches improved (`22 -> 14`)
    - PAR executions improved (`11 -> 7`)
  - but the residual stall did not resolve:
    - `stall_classification.primary_class` remained `tuple_group_blocking`
    - `max_tuple_blocked_streak` stayed at `160`
    - `max_tuple_deficit_to_row_completion` worsened (`1.8633 -> 2.3168`)
- **Why this experiment was rejected**:
  - the delay gate did suppress the earlier `(6,7)` bounce-back chain
  - but it did so mainly by stretching other recomposed final rows much longer before release:
    - `(1,2)` final row was deferred from blocked step `20` through `39`, then still released at `40` with dominant blocker distance about `0.793`
    - `(0,5)` final row was deferred through blocked step `39`, then still released at `40` with dominant blocker distance about `0.945`
    - `(5,6)` final row was deferred through blocked step `39`, then still released at `40` with dominant blocker distance about `2.383`
  - the most damaging outcome is that the release delay did **not** encode any geometry-based success condition; it only waited longer and then released anyway, even when the dominant mover was still far from the held final row
  - that means the experiment mostly converted “early bounce-back” into “longer occupancy followed by another unresolved release”
- **Current disposition**:
  - this release-delay experiment was **reverted**
  - the repository should stay on the earlier narrowed moving-subset runtime candidate from `20260316_172824`, which remains the better bounded mitigation
- **Updated interpretation**:
  - a time-only release delay is not sufficient
  - any future final-row-release fix needs to be geometry-aware or progress-aware, not just “wait 20 more steps and then release anyway”

### 5.22 Debug-Only Final-Row Release Retrigger Probe

- **Diagnostic change**:
  - added a debug-only `final_row_release_retrigger_probe` inside `gym_env/gym_env/envs/ir_gym.py`
  - the probe runs only when `row_advancement_reason=final_row_release` is about to happen
  - it does **not** change runtime behavior; it only snapshots the `DeadlockDetector` internals, rewrites the would-be released participants back to their saved long-range goals, evaluates deadlock detection/participant selection for those participants, then restores detector state
  - the probe writes one structured object into the corresponding `tuple_progression_traces` record
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_191103/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_191103/deadlock_logs/20260316_191103/episode_data_0_20260316_191122.json`
- **Observed top-level stats**:
  - `deadlock_detections=1489`
  - `mode_switches=22`
  - `par_executions=11`
  - compared with the earlier narrowed runtime candidate (`20260316_172824`), the main execution shape is unchanged (`mode_switches` and `par_executions` stayed the same), while `deadlock_detections` rose slightly (`1483 -> 1489`)
- **Probe outcomes across all final-row releases in this run**:
  - four final-row release points carried probe output
  - status distribution:
    - `no_retrigger`: `1`
    - `same_group_retrigger`: `2`
    - `detected_but_empty`: `1`
- **Per-release evidence**:
  - `step=277`, participants `[1,2]`, tuple `4/5`
    - probe status: `no_retrigger`
    - both seeds (`1`, `2`) would be released back to saved long-range goals and would **not** immediately trigger deadlock
  - `step=359`, participants `[5,6]`, tuple `2/3`
    - probe status: `same_group_retrigger`
    - seed `6` would immediately return the same pair `[5,6]`
    - seed `5` would instead return a different pair `[0,5]`
    - this is the first clear sign that final-row release can hand control back to RL in a state that is already unstable enough to produce either self-retrigger or the next blocker handoff family
  - `step=516`, participants `[0,5]`, tuple `1/2`
    - probe status: `same_group_retrigger`
    - both seeds would immediately return `[0,5]`
  - `step=847`, participants `[6,7]`, tuple `2/3`
    - probe status: `detected_but_empty`
    - both seeds (`6`, `7`) would immediately detect deadlock, but participant selection would return `[]` with `empty_return_reason=below_min_participants`
    - this matches the already observed post-release chain:
      - `step=848..850`: deadlock detected again but returns `[]`
      - `step=851`: the same pair `(6,7)` re-enters PAR
- **Interpretation**:
  - the probe validates that the residual issue really is release-time visible, not only a later emergent effect
  - more specifically:
    - some final-row releases are safe (`[1,2]`)
    - some are immediately unsafe because they would reselect the same pair or hand off into the next blocker family (`[5,6]`, `[0,5]`)
    - the most important late-row case `(6,7)` shows a slightly subtler but still unsafe pattern:
      - release would not immediately reselect `[6,7]`
      - but it would immediately put both seeds back into “deadlock detected but no participants returned” mode, which is exactly the bounce-back prefix that precedes the real retrigger at `step=851`
- **Implication for the next bounded fix**:
  - if a future runtime fix touches `final_row_release`, it should not be a pure time delay
  - the stronger evidence-backed direction is:
    - make final-row release conditional on geometry/progress **or**
    - explicitly suppress release when the probe predicts `same_group_retrigger` or `detected_but_empty`

### 5.23 Rejected Experiment: Runtime Final-Row Release Gate Driven Directly by the Probe

- **Experimental change**:
  - temporarily turned the debug probe into a runtime gate:
    - if `final_row_release_retrigger_probe.status` was `same_group_retrigger` or `detected_but_empty`
    - the env blocked `final_row_release` for that step and kept the tuple group active
  - this was intentionally narrow and only touched the already instrumented final-row release path
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_210700/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_210700/deadlock_logs/20260316_210700/episode_data_0_20260316_210720.json`
- **Observed result**:
  - top-level stats changed to:
    - `deadlock_detections=1988`
    - `mode_switches=19`
    - `par_executions=10`
  - compared with the probe-only baseline (`20260316_191103`):
    - `mode_switches` improved slightly (`22 -> 19`)
    - `par_executions` improved slightly (`11 -> 10`)
    - but `deadlock_detections` became much worse (`1489 -> 1988`)
  - stall classification remained `tuple_group_blocking`
  - `max_tuple_blocked_streak` worsened (`160 -> 200`)
  - the env recorded `blocked_final_row_release` `14` times
- **Why this experiment was rejected**:
  - the gate did succeed at preventing some premature releases
  - but in practice it mostly converted release/retrigger churn into longer-lived active tuple occupancy
  - representative blocked sequence:
    - `(5,6)` stayed on final row from `step=359` onward with repeated `probe_status=same_group_retrigger`
    - the dominant blocker distance remained around `1.04 -> 0.98`, yet the overall system accumulated many more deadlock detections elsewhere
  - the net effect was not “resolved geometry”, but “hold the final row longer and let other parts of the episode accumulate more detector activity”
- **Current disposition**:
  - this runtime gate was **reverted**
  - the repository should stay on the probe-only state from `20260316_210807`
  - reverted validation run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260316_210807/`
  - reverted stats returned to:
    - `deadlock_detections=1489`
    - `mode_switches=22`
    - `par_executions=11`
- **Updated interpretation**:
  - the probe is valuable as diagnosis
  - but using it as a direct binary release gate is too blunt
  - the next fix, if any, needs to be more local than “do not release whenever the probe looks unsafe”

### 5.24 Debug-Only Final-Row Partial-Release Probe

- **Diagnostic change**:
  - added a second debug-only probe, `final_row_partial_release_probe`
  - it runs only when the full `final_row_release_retrigger_probe` already marks the release as unsafe (`same_group_retrigger` or `detected_but_empty`)
  - the hypothetical scenario is:
    - release only the stationary participant(s) that are already within the final-row reach threshold back to their saved long-range goals
    - keep the still-moving participant(s) on the current MAPF/final-row goal
  - like the earlier release probe, this one snapshots and restores `DeadlockDetector` state and writes evidence only into `tuple_progression_traces`
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_091948/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_091948/deadlock_logs/20260317_091948/episode_data_0_20260317_092008.json`
- **Observed top-level stats**:
  - `deadlock_detections=1492`
  - `mode_switches=22`
  - `par_executions=11`
  - compared with the previous probe-only run (`20260316_210807`), the overall execution shape is effectively unchanged (`22 / 11` remains the same), while `deadlock_detections` moved slightly (`1489 -> 1492`)
- **Partial-release probe coverage**:
  - full release probe still appeared at `4` final-row releases
  - partial release probe was relevant at `3` of them, exactly the ones where full release had already been classified as unsafe
  - status distribution:
    - `same_group_retrigger`: `2`
    - `detected_but_empty`: `1`
- **Per-case evidence**:
  - `step=359`, participants `[5,6]`, tuple `2/3`
    - full release status: `same_group_retrigger`
    - partial release split:
      - stationary released: `[6]`
      - moving retained in MAPF: `[5]`
    - result:
      - seed `6` still immediately detects deadlock and returns `[5,6]`
    - interpretation:
      - releasing only the stationary participant does **not** remove the unstable geometry for this pair
  - `step=516`, participants `[0,5]`, tuple `1/2`
    - full release status: `same_group_retrigger`
    - partial release split:
      - stationary released: `[5]`
      - moving retained in MAPF: `[0]`
    - result:
      - seed `5` still immediately detects deadlock and returns `[0,5]`
  - `step=847`, participants `[6,7]`, tuple `2/3`
    - full release status: `detected_but_empty`
    - partial release split:
      - stationary released: `[7]`
      - moving retained in MAPF: `[6]`
    - result:
      - seed `7` still immediately detects deadlock, but participant selection returns `[]` with `empty_return_reason=below_min_participants`
- **Interpretation**:
  - this is a strong negative result:
    - the unsafe release cases are **not** made safe simply by “release only the stationary one”
  - in every unsafe final-row case tested, the partial-release probe produced the same qualitative outcome as the full-release probe
  - that means the residual issue is deeper than release granularity alone; the unstable geometry persists even when the mover remains in MAPF and only the stationary participant is hypothetically handed back to RL
- **Implication for the next bounded fix**:
  - do **not** pursue a runtime partial-release patch next
  - the next useful investigation should move one level deeper, toward why the retained mover/final-row geometry is itself unstable, rather than trying another release policy tweak

### 5.25 Debug-Only Final-Row Instability Probe

- **Diagnostic change**:
  - added a third debug-only probe, `final_row_instability_probe`
  - it runs only for the same dangerous final-row cases that already triggered `final_row_partial_release_probe`
  - the goal is to separate three possibilities:
    - the mover’s final-row target is itself geometrically bad
    - the stationary participant is physically blocking the mover’s final-row corridor
    - the mover has a seemingly valid target/corridor but still fails to converge under RL/RVO execution
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_093527/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_093527/deadlock_logs/20260317_093527/episode_data_0_20260317_093546.json`
- **Observed top-level stats**:
  - `deadlock_detections=1492`
  - `mode_switches=22`
  - `par_executions=11`
  - compared with the previous partial-release-probe run (`20260317_091948`), the episode-level execution shape is effectively unchanged
- **Instability-probe coverage**:
  - probe count: `3`
  - exactly the three dangerous final-row cases previously identified:
    - `(5,6)` at `step=359`
    - `(0,5)` at `step=516`
    - `(6,7)` at `step=847`
- **What the probe measured**:
  - for the mover:
    - `distance_to_row_target`
    - `distance_to_current_goal`
    - whether `current_goal_matches_row_target`
    - desired-velocity projection toward the final-row target
    - `mapf_non_progress_steps`
    - whether the straight segment from mover position to row target crosses an occupied planner cell
  - for each mover/stationary pair:
    - current pair distance
    - target pair distance
    - distance from the mover-to-target segment to the stationary participant’s current position
    - distance from the same segment to the stationary participant’s held target
    - corridor clearance after subtracting the two robots’ combined collision radii
- **Key observations**:
  - for all three dangerous cases, the mover’s `current_goal` already matches the final-row target:
    - `(5,6)`: `goal_match=True`
    - `(0,5)`: `goal_match=True`
    - `(6,7)`: `goal_match=True`
  - for all three dangerous cases, the mover’s straight segment to the final-row target does **not** cross an occupied planner cell:
    - `segment_to_row_target_blocked_cell=None`
  - for all three dangerous cases, the mover’s desired velocity remains positively aligned with the final-row target:
    - desired forward projection is essentially `1.0`
  - but the mover still shows non-trivial or sustained non-progress:
    - `(5,6)`: mover `5` had `mapf_non_progress_steps=17`
    - `(6,7)`: mover `6` had `mapf_non_progress_steps=25`
  - stationary participants are also not sitting directly on top of the mover’s final-row corridor:
    - `(5,6)`: corridor clearance after subtracting both robots’ radii is about `3.19`
    - `(0,5)`: corridor clearance is about `0.91`
    - `(6,7)`: corridor clearance is about `3.52`
  - especially for `(6,7)`, the stationary participant is not even within current communication range:
    - current pair distance `4.13`
    - `within_communication_range_current=False`
- **Interpretation**:
  - this is the strongest evidence so far that the remaining instability is **not** primarily:
    - an obstacle-grid mismatch on the mover’s direct final-row segment
    - a stationary participant physically sitting on the mover’s straight-line corridor
    - a mismatched “current goal vs row target” issue
  - instead, the retained mover already has:
    - the right target
    - a clear straight segment in planner space
    - a desired velocity aligned with that target
  - and yet it still accumulates non-progress while the episode remains classified as `tuple_group_blocking`
- **Updated interpretation**:
  - the final-row problem now looks less like “bad target geometry” and more like an **execution-level convergence failure** in the local multi-agent dynamics
  - more concretely:
    - the mover is trying to go to the right place
    - the stationary participant is not literally occupying the mover’s direct target corridor
    - but the closed-loop RL/RVO execution still fails to produce stable completion of the final-row approach
- **Implication for the next bounded fix**:
  - the next useful step should target final-row execution dynamics, not release policy
  - the best candidate question now is:
    - why does a mover with a valid target and aligned desired velocity still fail to reduce its residual final-row error under the current RL/RVO control loop?

### 5.26 Debug-Only Final-Row Execution-Dynamics Probe

- **Diagnostic change**:
  - added a fourth debug-only probe, `final_row_execution_dynamics_probe`
  - it runs on the same dangerous final-row cases that already triggered `final_row_release_retrigger_probe` and `final_row_partial_release_probe`
  - unlike `final_row_instability_probe`, this probe does not focus on geometry; it joins the final-row case to the same-step MAPF execution trace so we can see:
    - the mover's desired velocity
    - the raw policy action
    - the post-cap applied action
    - the actual post-step velocity
    - speed-cap / yielding flags
    - RVO-context fields (`neighbor_count_used_by_rvo`, `vo_flag`, `min_exp_time`)
    - a few lightweight diagnostic tags
- **Validation run**:
  - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_094602/`
  - episode JSON:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_094602/deadlock_logs/20260317_094602/episode_data_0_20260317_094621.json`
- **Observed top-level stats**:
  - `deadlock_detections=1492`
  - `mode_switches=22`
  - `par_executions=11`
  - compared with the previous instability-probe run (`20260317_093527`), the episode-level shape is unchanged
- **Execution-dynamics coverage**:
  - probe count: `3`
  - exactly the same dangerous final-row cases:
    - `(5,6)` at `step=359`
    - `(0,5)` at `step=516`
    - `(6,7)` at `step=847`
- **Key observations by case**:
  - `(5,6)`:
    - mover `5` does **not** merely underperform a good command; its raw action is already pointed the wrong way
    - `desired_velocity_vector=[-0.975, -0.220]`, but `raw_action=[0.220, -0.014]`
    - `raw_action_angle_to_goal_deg=163.66`
    - `raw_action_forward_projection_to_goal=-0.212`
    - actual motion follows that bad command instead of correcting it:
      - `actual_velocity_angle_to_goal_deg=174.17`
      - `actual_velocity_forward_projection_to_goal=-0.209`
    - no env-side attenuation explains this:
      - `speed_cap_applied=false`
      - `yield_applied=false`
      - `vo_flag=false`
  - `(0,5)`:
    - mover `0` points roughly toward the right goal, but the command is tiny relative to the desired velocity
    - `desired_velocity_speed=1.0`
    - `raw_action_speed=0.0399`
    - `applied_vs_desired_speed_ratio=0.0399`
    - heading is not catastrophic:
      - `raw_action_angle_to_goal_deg=45.82`
      - `raw_action_forward_projection_to_goal=0.0278`
    - actual motion stays tiny as well:
      - `actual_velocity_speed=0.04`
      - `distance_delta_to_current_goal=0.0035`
  - `(6,7)`:
    - mover `6` has a directionally reasonable raw action, but it is still much smaller than the desired velocity
    - `desired_velocity_speed=1.0`
    - `raw_action_speed=0.2619`
    - `applied_vs_desired_speed_ratio=0.2619`
    - then actual motion shrinks even further:
      - `actual_velocity_speed=0.09`
      - `actual_vs_applied_speed_ratio=0.3436`
      - `actual_velocity_angle_to_goal_deg=49.26`
    - again, this is not explained by the existing env-side guards:
      - `speed_cap_applied=false`
      - `yield_applied=false`
      - `vo_flag=false`
- **Cross-case pattern**:
  - all three dangerous final-row movers still saw `neighbor_count_used_by_rvo=8`, so they were not in a "neighbor-free" execution regime
  - but none of them were under immediate VO pressure at the sampled release step:
    - `vo_flag=false`
    - `min_exp_time=null`
  - the env's explicit action modifiers are not the primary explanation in these cases:
    - `speed_cap_applied=false`
    - `yield_applied=false`
- **Updated interpretation**:
  - the remaining final-row failures are **not one single mechanism**
  - at least two execution families now show up:
    - **policy action generation failure**:
      - exemplified by `(5,6)`, where the raw action already opposes the goal
    - **weak command / weak realized motion near final-row residuals**:
      - exemplified by `(0,5)` and `(6,7)`, where the raw action is goal-aligned but much smaller than the desired velocity, and in `(6,7)` the realized motion shrinks even further
  - this means the next bounded diagnosis should focus on the MAPF execution controller itself:
    - why the RL policy emits such weak or even reversed commands in late/final-row PAR execution
    - and, separately, why some already-small commands degrade further into tiny realized motion

### 5.27 Policy-Side Action Probe and Final-Row Direction Counterfactuals

- **Diagnostic change**:
  - added `policy_action_traces` in the deadlock episode JSON
  - these records are emitted only for `DEBUG_MODE + mapf` agents and capture:
    - actor mean `policy_mu`
    - sampled increment action
    - current velocity before the step
    - resulting absolute action command passed into env
    - the robot-state observation slice (`cur_vel`, `des_vel`, heading, radius)
    - the external/moving-state chunks seen by the policy
  - added two env-side counterfactual probes for dangerous final-row cases:
    - `final_row_desired_velocity_probe`
    - `final_row_scaled_desired_direction_probe`
- **Validation runs**:
  - policy-side probe run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095337/`
    - episode JSON:
      - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095337/deadlock_logs/20260317_095337/episode_data_0_20260317_095356.json`
  - desired-velocity counterfactual run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095720/`
    - episode JSON:
      - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095720/deadlock_logs/20260317_095720/episode_data_0_20260317_095740.json`
  - scaled-desired-direction counterfactual run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095921/`
    - episode JSON:
      - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_095921/deadlock_logs/20260317_095921/episode_data_0_20260317_095940.json`
- **Observed top-level stats**:
  - all three runs remained on the same reproduction baseline:
    - `deadlock_detections=1492`
    - `mode_switches=22`
    - `par_executions=11`
- **Policy-side observations**:
  - the bad final-row command is already present in the actor output; it is not introduced by env-side action modifiers
  - `(5,6)` at `step=359`:
    - `policy_mu=[0.1505, -0.0687]`
    - desired velocity from observation: `[-0.98, -0.22]`
    - `policy_mu_angle_to_desired_deg=142.81`
    - `policy_mu_forward_projection_to_desired=-0.1318`
    - absolute env command also opposes the goal
  - `(0,5)` at `step=516`:
    - `policy_mu=[-0.2006, -0.0415]`
    - desired velocity from observation: `[1.0, 0.06]`
    - `policy_mu_angle_to_desired_deg=171.75`
    - `policy_mu_forward_projection_to_desired=-0.2028`
    - after adding the current velocity, the absolute env command becomes only slightly forward and extremely small
  - `(6,7)` at `step=847`:
    - `policy_mu=[-0.2339, -0.1080]`
    - desired velocity from observation: `[-1.0, -0.03]`
    - direction is roughly correct (`23.08 deg`), but still weak relative to the desired speed
  - for all three cases, the sampled action stayed extremely close to `policy_mu` because `std_factor=0.001`
    - so this is effectively deterministic actor behavior, not sampling noise
- **Counterfactual controller observations**:
  - a naive full-speed desired-velocity fallback is **not** safe in these dangerous final-row cases:
    - `(5,6)`: `desired_action_vo_flag=true`, `min_exp_time=0.815`
    - `(0,5)`: `desired_action_vo_flag=true`, `min_exp_time=0.474`
    - `(6,7)`: `desired_action_vo_flag=true`, `min_exp_time=0.769`
  - however, an equally fast command that only rotates the action onto the desired direction **is** locally safe in all three cases:
    - `(5,6)`: `scaled_action_vo_flag=false`
    - `(0,5)`: `scaled_action_vo_flag=false`
    - `(6,7)`: `scaled_action_vo_flag=false`
  - in all three, the scaled desired-direction action kept the same reference speed as the actual policy action but made the forward projection to goal positive and locally safe
- **Updated interpretation**:
  - the evidence now points to a more specific failure mode:
    - the current policy often chooses a poor direction or over-braking direction near dangerous final-row states
    - but the local RVO context does **not** require abandoning the desired direction entirely
    - what appears feasible is:
      - keep the locally conservative action magnitude
      - correct only the direction toward the desired final-row target
  - this is much narrower than “replace policy with full desired velocity,” which the counterfactual probe explicitly rejected
- **Most likely next bounded fix candidate**:
  - a debug-gated runtime experiment for dangerous final-row MAPF movers only:
    - if the actor output is non-progressing and a same-speed desired-direction action is locally safe, substitute that corrected-direction command for the current step
  - importantly, this candidate is now evidence-backed in a way that the earlier full desired-velocity fallback was not

### 5.28 Final-Row Desired-Direction Runtime Experiment and Diff-Drive Mismatch

- **Runtime experiment change**:
  - added a debug-gated env-side fallback behind deadlock config keys:
    - `FINAL_ROW_DIRECTION_CORRECTION_ENABLED`
    - `FINAL_ROW_DIRECTION_CORRECTION_DEBUG_ONLY`
    - `FINAL_ROW_DIRECTION_CORRECTION_MIN_NON_PROGRESS_STEPS`
    - `FINAL_ROW_DIRECTION_CORRECTION_WEAK_SPEED_RATIO`
  - when enabled, the env inspects active **final-row** PAR movers before dynamics:
    - the mover must still be off the row target
    - it must already have sustained MAPF non-progress
    - its current action must either oppose the goal or be too weak relative to desired velocity
    - a same-speed desired-direction replacement must remain locally safe under the current RVO context
  - if all of those hold, the env substitutes that corrected-direction action for the current step and records the decision inside `mapf_execution_traces`
- **Validation run**:
  - runtime experiment run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_103739/`
    - episode JSON:
      - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_103739/deadlock_logs/20260317_103739/episode_data_0_20260317_103758.json`
- **Observed top-level stats**:
  - with the runtime experiment enabled:
    - `deadlock_detections=1540`
    - `mode_switches=18`
    - `par_executions=10`
  - compared with the pre-experiment narrowed baseline (`20260317_095921`):
    - `deadlock_detections` worsened slightly: `1492 -> 1540`
    - `mode_switches` improved: `22 -> 18`
    - `par_executions` improved slightly: `11 -> 10`
    - `tuple_blocked_steps` improved slightly: `929 -> 920`
    - `max_tuple_blocked_streak` stayed unchanged at `160`
    - `max_deficit_to_row_completion` improved slightly: `1.8633 -> 1.7992`
- **What actually happened**:
  - the fallback **did** trigger:
    - `considered=130`
    - `applied=27`
  - representative corrected cases:
    - `(5,6)` final row at `step=359`, mover `5`
      - raw action opposed the goal (`164.95 deg`)
      - corrected action became goal-aligned (`0.0 deg`)
    - later `(6,7)`-family final-row steps around `step=761..773`, mover `6`
      - several weak or misaligned actions were replaced with goal-aligned same-speed actions
- **Critical new observation**:
  - in the key applied cases, the corrected world-frame action did **not** produce translational motion:
    - `step=359, agent 5`
      - corrected `applied_action=[-0.1957, -0.0423]`
      - `actual_velocity_speed=0.0`
    - `step=772, agent 6`
      - corrected `applied_action=[-0.2942, -0.0252]`
      - `actual_velocity_speed=0.0`
  - this is now explained by the simulator robot model rather than by RVO:
    - `mode8_long_range.yaml` uses `robot_mode: 'diff'`
    - `_step_pure_rl(..., vel_type='omni')` still routes through `mobile_robot.move_from_omni()`
    - `move_from_omni()` converts the world-frame omni command into differential-drive control with `omni2diff()`
    - for the corrected cases, the robot heading was still far from the corrected world-frame target direction:
      - `step=359, agent 5`: heading-to-corrected-action gap `118.28 deg`
      - `step=772, agent 6`: heading-to-corrected-action gap `150.53 deg`
    - under `omni2diff()`, both of those produce:
      - `predicted_diff_linear_v=0.0`
      - `predicted_diff_angular_w=1.0`
    - which matches the measured `actual_velocity_speed=0.0`
- **Updated interpretation**:
  - the desired-direction runtime fallback is **not** the right fix family in its current world-frame form
  - it improved the commanded action direction, but it ignored the fact that execution still passes through a differential-drive conversion layer
  - so the remaining failure is not just “policy picked a bad direction”; it is now also a **control-space mismatch**:
    - a world-frame omni correction can still collapse into pure turning after `omni2diff()`
  - because of this mismatch, the canonical debug config has been reset to keep `FINAL_ROW_DIRECTION_CORRECTION_ENABLED=false`
  - this experiment should be treated as **evidence gathered and rejected as the current runtime fix path**, not as an accepted mitigation
- **Canonical repro reset check**:
  - after disabling the experiment again in the canonical debug config, `test-par-debug` returned to the prior narrowed baseline:
    - run: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_104631/`
    - stats: `deadlock_detections=1492`, `mode_switches=22`, `par_executions=11`
    - `final_row_direction_correction_considered=0`, `applied=0`

### 5.29 Final-Row Diff-Drive Feasibility Probe

- **Instrumentation change**:
  - added a new debug-only `final_row_diff_drive_feasibility_probe`
  - for the same dangerous final-row cases already covered by:
    - `final_row_release_retrigger_probe`
    - `final_row_partial_release_probe`
    - `final_row_execution_dynamics_probe`
  - the env now also projects four world-frame actions through the robot's actual diff-drive execution layer:
    - `raw_action_diff_drive_projection`
    - `applied_action_diff_drive_projection`
    - `desired_velocity_diff_drive_projection`
    - `same_speed_desired_direction_diff_drive_projection`
  - the same `*_diff_drive_projection` objects are also copied into `mapf_execution_traces` so they can be correlated step-by-step outside the final-row probe
- **Validation run**:
  - canonical debug run:
    - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_105404/`
    - episode JSON:
      - `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/20260317_105404/deadlock_logs/20260317_105404/episode_data_0_20260317_105424.json`
  - top-level stats stayed at the narrowed baseline:
    - `deadlock_detections=1492`
    - `mode_switches=22`
    - `par_executions=11`
  - so the new probe did not change canonical runtime behavior
- **Observed dangerous final-row cases**:
  - three dangerous cases emitted the new probe:
    - `step=359`, participants `[5,6]`, mover `5`
    - `step=516`, participants `[0,5]`, mover `0`
    - `step=847`, participants `[6,7]`, mover `6`
- **Key observations**:
  - `(5,6)` at `step=359`:
    - current `raw/applied` action is actually diff-drive feasible:
      - heading error `21.05 deg`
      - predicted linear ratio vs world speed `0.93`
      - no linear clip-to-zero
    - but both desired-direction counterfactuals are diff-drive infeasible **at this instant**:
      - `desired_velocity_heading_error=-175.29 deg`
      - `predicted_linear_v=0.0`
      - `predicted_turn_in_place=true`
    - this means the earlier world-frame desired-direction fallback failed here for a deeper reason:
      - not because diff-drive broke an otherwise feasible command
      - but because “point directly at the goal right now” is itself not a diff-feasible command from the current heading
  - `(0,5)` at `step=516`:
    - there is **no** diff-drive feasibility failure:
      - current `raw/applied` action keeps `0.92` of its world-frame speed after projection
      - desired velocity is also diff-feasible (`0.92` linear ratio)
    - so this case remains a weak-command / controller-output issue rather than a diff-drive mismatch issue
  - `(6,7)` at `step=847`:
    - current `raw/applied` action is not clipped to zero, but diff-drive still shrinks it materially:
      - heading error `-69.86 deg`
      - predicted linear ratio vs world speed `0.34`
    - desired velocity is also diff-drive feasible, but only moderately so:
      - heading error `-54.99 deg`
      - predicted linear ratio `0.57`
    - so this family is not “turn-in-place only,” but it is still a strong heading-feasibility attenuation case
- **Updated interpretation**:
  - dangerous final-row movers are **not** all the same failure family
  - the new probe separates at least three subfamilies:
    - a **goal-opposing but forward-feasible** policy action (`step=359`)
    - a **weak but diff-feasible** policy action (`step=516`)
    - a **moderately heading-attenuated** diff-drive tracking case (`step=847`)
  - this means the next bounded fix candidate should not be “always rotate action to desired direction in world frame”
  - if we pursue a runtime fallback later, it should be **heading-feasible for diff-drive**, not merely goal-aligned in world coordinates
  - a promising next diagnostic question is:
    - can we build a diff-feasible counterfactual that improves goal progress without triggering the `omni2diff()` zero-linear-velocity collapse seen in the rejected world-frame desired-direction experiment?

1. Add instrumentation (e.g. logs) to record: which agent first triggered detection each step; the participant set; per-agent mode and waypoint index each step; collision events with agent IDs and modes.
2. Run the test, reproduce failures, and fill **Attempts** and **Status** for each hypothesis using log evidence.
3. After confirming root causes, document fixes in this section and link to code changes; then update hypotheses status and attempt table.

---

## 6. References

- Paper outline: `papers/l4dc_paper_outline_v1.md`
- Architecture: `docs/architecture.md`, `rl_rvo_nav/docs/architecture.md`
- Deadlock contract: `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md`
- Deadlock design: `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md`
- Gym integration: `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- Integration test: `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`

---

## 7. Codex-Friendly GitHub Issue Backlog (Planning Snapshot)

This section stores the current planning-only backlog for debugging `policy_test_long_range_with_par.py`. It is written for agent-driven implementation loops and should be kept in sync with actual evidence gathered in this document. Do not create a parallel backlog document for the same test; update this section instead.

### 7.1 Epic Summary

- Start by making the failing PAR integration run reproducible through one fixed debug entry point, one debug config, and one artifact layout.
- Add machine-readable evidence for the three failure families separately: participant selection, waypoint stall, and collision.
- Use the participant-selection hypotheses from this document to drive a bounded trace and a small parameter sweep before changing any behavior.
- Use a dedicated waypoint trace plus MAPF goal-alignment and tuple-blocker metrics to separate per-agent stall causes from tuple-group stall causes.
- Capture collision context independently from stall analysis so obstacle mismatch and execution-safety problems do not get conflated.
- Generate a planner-vs-simulator overlay for obstacle collisions before touching grid or obstacle semantics.
- Keep all early issues diagnostic or test-local; do not change default algorithm semantics until earlier evidence is in hand.
- Split behavior changes into separate issues for config-local participant gating, seed-order arbitration, per-agent waypoint progression, tuple-group progression, and collision mitigation.
- Update existing docs only: this debug document, the runbook, env design/integration docs, and relevant contract docs.

### 7.2 GitHub Issues

#### Issue 1

## Title
Add a fixed `test-par-debug` entry point and debug artifact contract for `policy_test_long_range_with_par`

## Goal
Create one canonical, repeatable debug run for the failing integration test that always uses the same script entry point, debug config, seed settings, and artifact layout.

## Why this issue exists now
Every later issue depends on comparing evidence across runs. Right now the repo has the test entry point, but not a dedicated debug contract for this failure.

## Relevant failure pattern(s)
- P1 Wrong agents enter MAPF
- P2 Waypoint not advancing
- P3 Collisions

## Related hypothesis/hypotheses
- Supports H1a-H1d, H2a-H2e, H3a-H3d by making them reproducible

## Relevant files and docs to read first
- `structured_file_guidance.md`
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/docs/repro/runbook.md`
- `scripts/run_all.sh`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py`

## Constraints
- Do not change default algorithm semantics for non-debug runs.
- Update the existing fixed entry point in `scripts/run_all.sh` instead of inventing a parallel workflow.
- Any config added here must be test-local and debug-only.
- No duplicate docs; update existing runbook and debug doc only.

## Deliverables
- A dedicated `test-par-debug` entry point in `scripts/run_all.sh`.
- One test-local debug config file for this integration test.
- A run manifest artifact containing command, config path, config hash, git SHA, world, model, seed values, and output directory.
- A documented artifact directory convention for logs, summaries, and plots from this test.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `bash scripts/run_all.sh test-par-debug`

## Expected evidence or acceptance criteria
- Two successive runs produce the same manifest schema and config hash.
- The artifact directory structure is stable and documented.
- This debug document can point to one canonical command instead of ad hoc invocations.

## Out of scope
- Adding participant, waypoint, or collision instrumentation.
- Fixing any behavior bug.

## Documentation updates required
- Update `rl_rvo_nav/docs/repro/runbook.md` because it is the run SoT and should point to the fixed debug entry point.
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` because it should record the canonical repro command and artifact layout for this exact failure.

## Suggested labels
- `debug`
- `repro`
- `local-first`

## Depends on
- None

## Can run in parallel with
- None

#### Issue 2

## Title
Instrument deadlock trigger evaluation and participant-set construction for P1

## Goal
Log exactly how each RL agent is evaluated for deadlock and how the returned participant set is constructed for every MAPF trigger.

## Why this issue exists now
P1 cannot be resolved until we can reconstruct whether the wrong group came from seed order, graph scope, or participant gating.

## Relevant failure pattern(s)
- P1 Wrong agents enter MAPF

## Related hypothesis/hypotheses
- H1a First-detector wins
- H1b Conflict graph != true deadlock set
- H1c `MIN_PAR_PARTICIPANTS=4`
- H1d Neighbor state scope

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/deadlock_resolution/deadlock_detector.py`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`

## Constraints
- Instrumentation must be gated by debug mode.
- Do not change participant-selection behavior in this issue.
- Use machine-readable structured output, not only console prints.

## Deliverables
- Structured logs for every RL agent deadlock check containing:
- `step`, `agent_id`, `loop_index`, `current_mode`, cooldown state, min-history state, average velocity, trigger source.
- `COMMUNICATION_RANGE` neighbors, slow neighbors, non-progress neighbors, risk neighbors.
- Conflict graph adjacency, extracted component, filtered nodes, prioritized order, clipped order.
- `MIN_PAR_PARTICIPANTS`, `MAX_PAR_PARTICIPANTS`, returned participants, empty-return reason, solver cache key, solver-invoked vs skipped.
- One sample artifact from a reproduced failing run.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `rg -n "\"returned_participants\"" rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logs -g '*.json'`

## Expected evidence or acceptance criteria
- For the first MAPF event, the log is sufficient to explain why each included agent was included and why each excluded agent was excluded.
- The log distinguishes `detected deadlock but returned []` from `did not detect deadlock`.
- The log makes seed order explicit per step.

## Out of scope
- Parameter sweeps.
- Config tuning.
- Behavioral fixes.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the new evidence fields and initial findings.
- Update `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md` because `DEBUG_MODE` logging behavior is part of the module's exposed debug contract.

## Suggested labels
- `debug`
- `instrumentation`
- `local-first`

## Depends on
- Issue 1

## Can run in parallel with
- Issue 4
- Issue 6

#### Issue 3

## Title
Run a bounded participant-sensitivity sweep over `COMMUNICATION_RANGE` and `MIN_PAR_PARTICIPANTS`

## Goal
Measure how participant sets and MAPF trigger outcomes change under a small test-local config matrix, using the trace from Issue 2.

## Why this issue exists now
This document already shows `COMMUNICATION_RANGE` matters, and leaves `MIN_PAR_PARTICIPANTS` as an open question. This needs quantified evidence before any P1 fix.

## Relevant failure pattern(s)
- P1 Wrong agents enter MAPF

## Related hypothesis/hypotheses
- H1b Conflict graph != true deadlock set
- H1c `MIN_PAR_PARTICIPANTS=4`

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/docs/repro/runbook.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md`
- `rl_rvo_nav/config/deadlock_config.py`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`

## Constraints
- Use test-local configs only.
- Do not change global defaults in `rl_rvo_nav/config/deadlock_config.py`.
- Keep the sweep small and reviewable.
- This issue is still evidence-gathering, not a fix.

## Deliverables
- A scripted sweep entry point for this test.
- A summary table with one row per config containing first trigger step, trigger seed, returned participants, number of empty returns due to min-participant gating, and episode outcome.
- A short conclusion in this debug document stating whether P1 is mostly graph-scope-sensitive, min-gate-sensitive, or both.

## Validation commands
- `bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh`

## Expected evidence or acceptance criteria
- The sweep produces a bounded table, not free-form notes.
- The table makes it clear whether the "wrong group" is stable or highly sensitive to these two parameters.
- This debug document records which P1 hypotheses moved from inconclusive to confirmed or rejected.

## Out of scope
- Applying any config or code changes outside the sweep harness.
- Changing the conflict-graph algorithm.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` because it is the single debugging SoT.
- Update `rl_rvo_nav/docs/repro/runbook.md` because repeated sweep commands should be documented as fixed entry points.

## Suggested labels
- `debug`
- `analysis`
- `parameter-sweep`
- `local-first`

## Depends on
- Issue 1
- Issue 2

## Can run in parallel with
- Issue 4
- Issue 6

#### Issue 4

## Title
Instrument per-agent waypoint progression for RL and MAPF modes

## Goal
Record exactly how waypoint indices, thresholds, and stay counters evolve for each agent on each step.

## Why this issue exists now
P2 currently looks like "stuck until timeout," but that hides whether the problem is thresholding, force-switch, missing history updates, or a separate tuple barrier.

## Relevant failure pattern(s)
- P2 Waypoint not advancing

## Related hypothesis/hypotheses
- H2a Reach threshold too strict
- H2d Force-switch disabled or too high
- H2e Waypoint index not updated for MAPF

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/LongRangeNavi/docs/long_range_navigation.contract.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/LongRangeNavi/waypoint_manager.py`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`

## Constraints
- Instrumentation must be debug-gated.
- Do not change waypoint progression semantics in this issue.
- Keep logging machine-readable and step-local.

## Deliverables
- Structured per-step waypoint logs containing:
- `step`, `agent_id`, `mode`, `manager_type`, waypoint index before/after, total waypoints.
- `stay_steps` before/after, `reach_threshold`, force-switch enabled flag, force-switch step limit.
- Current goal, distance to current goal, `reached`, `final_reached`, advancement reason (`distance`, `force_switch`, `none`).
- Whether `update_waypoint_history` was called, and with which index.
- Any MAPF exit reason emitted that step.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `rg -n "\"advancement_reason\"" rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logs -g '*.json'`

## Expected evidence or acceptance criteria
- For a stalled agent, the logs show whether its waypoint index ever changed.
- If the index did not change, the logs show the exact threshold/counter context.
- If the index did change, the logs show whether it was distance-based or force-switch-based.

## Out of scope
- Classifying tuple-group blockers.
- Changing thresholds or counters.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the new waypoint evidence.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because waypoint update order and PAR manager injection are implementation details whose current doc is too stale for this debug path.

## Suggested labels
- `debug`
- `instrumentation`
- `local-first`

## Depends on
- Issue 1

## Can run in parallel with
- Issue 2
- Issue 6

#### Issue 5

## Title
Add MAPF goal-alignment metrics and tuple-blocker classification for stalled episodes

## Goal
Classify each stall as tuple-group blocking, poor MAPF goal tracking, or another measured non-progress mode.

## Why this issue exists now
Issue 4 explains waypoint-manager state, but not whether the RL-controlled MAPF execution is actually moving toward the overridden goal or whether a tuple row is blocking everyone else.

## Relevant failure pattern(s)
- P2 Waypoint not advancing

## Related hypothesis/hypotheses
- H2b PAR tuple: all must reach
- H2c RL policy does not drive toward overridden goal

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/test_mapf_waypoint_tuples.py`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`

## Constraints
- This issue is diagnostic only.
- Do not change speed caps, yielding, or tuple semantics here.
- Reuse Issue 4 trace output where possible.

## Deliverables
- Per-step MAPF execution metrics containing:
- Goal vector, desired velocity vector, applied action before/after cap, angle-to-goal, forward projection, distance delta to current goal, consecutive non-progress steps.
- Per-step tuple metrics containing tuple index, per-participant distance to target, `all_reached`, blocking participant IDs, and max deficit to row completion.
- One episode-level stall classifier output with a single primary class per timeout episode.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`

## Expected evidence or acceptance criteria
- A timeout episode can be labeled with one measured primary stall class.
- At least one stalled episode shows either a concrete blocking participant or a concrete repeated goal-misalignment pattern.
- This debug document records which of H2b or H2c remains plausible after measurement.

## Out of scope
- Fixing stalls.
- Changing tuple progression logic.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with stall-class evidence and hypothesis status.
- Update `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md` because tuple execution and MAPF goal override behavior are caller-integration details.

## Suggested labels
- `debug`
- `analysis`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 4

## Can run in parallel with
- Issue 7

#### Issue 6

## Title
Instrument structured collision context for agent-agent and agent-obstacle failures

## Goal
Capture enough structured state at collision time to distinguish execution-safety failures from planner/world mismatches.

## Why this issue exists now
P3 currently terminates the episode, but the repo does not preserve enough context to tell whether the root cause is RVO execution, PAR tracking, yielding, or obstacle mismatch.

## Relevant failure pattern(s)
- P3 Collisions

## Related hypothesis/hypotheses
- H3a No separate collision-free MAPF execution
- H3c Yielding / speed cap insufficient
- H3d RVO/ORCA and MAPF goals conflict

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py`

## Constraints
- Instrumentation must be debug-gated.
- Do not change collision or reward semantics.
- If precise obstacle identity is unavailable, log closest obstacle type and coordinates.

## Deliverables
- Structured collision event logs containing:
- `step`, collision type, involved agent IDs or obstacle metadata.
- Involved agent modes, solver type, participant set, tuple index or PAR manager index.
- Positions, velocities, current goals, waypoint indices, raw actions, modified actions.
- Whether speed cap or yielding was applied, `EXCLUDE_PAR_NEIGHBORS_IN_RVO`, neighbor count used by RVO, `vo_flag`, `min_exp_time`, and terminal `collision_flag`.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `bash scripts/run_all.sh test-par-debug --num_episodes 3`

## Expected evidence or acceptance criteria
- Each collision can be classified from logs without attaching a debugger.
- The log distinguishes collisions that happen in pure RL mode from collisions that happen in MAPF mode.
- The log preserves enough context to compare against Issue 7 overlays later.

## Out of scope
- Any collision fix.
- Any obstacle-grid alignment change.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with representative collision evidence.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because collision handling and modified-action application are implementation details in the env step path.

## Suggested labels
- `debug`
- `instrumentation`
- `local-first`

## Depends on
- Issue 1

## Can run in parallel with
- Issue 2
- Issue 4

#### Issue 7

## Title
Generate planner-vs-simulator obstacle overlays for MAPF collision episodes

## Goal
Test whether obstacle collisions come from occupancy-grid mismatch or from execution diverging off an otherwise valid plan.

## Why this issue exists now
H3b is a separate bounded hypothesis and should be falsified before changing speed caps, yielding, or RVO interaction rules.

## Relevant failure pattern(s)
- P3 Collisions

## Related hypothesis/hypotheses
- H3b Obstacle representation mismatch

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/deadlock_resolution/par_environment.py`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logger.py`

## Constraints
- Diagnostic only.
- Do not change occupancy, dilation, or solver behavior in this issue.
- Reuse Issue 6 collision artifacts when possible.

## Deliverables
- A debug artifact generator that exports:
- World bounds, grid resolution, dilation settings, occupancy snapshot.
- Continuous MAPF path or waypoint list for the colliding agent.
- Collision point and nearest obstacle geometry.
- A PNG and JSON overlay stating whether the collision point mapped to an occupied or free planner cell.

## Validation commands
- `bash scripts/run_all.sh test-par-debug --num_episodes 3`

## Expected evidence or acceptance criteria
- At least one obstacle collision episode produces an overlay.
- The overlay explicitly distinguishes `planner_free_sim_collision` from `planner_occupied_sim_collision`.
- This debug document records whether H3b remains plausible after reviewing the overlay.

## Out of scope
- Fixing obstacle mismatch.
- Changing deadlock workspace semantics.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the overlay evidence.
- Update `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md` because deadlock workspace and occupancy-grid semantics live there.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because the env builds and consumes the occupancy grid.

## Suggested labels
- `debug`
- `analysis`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 6

## Can run in parallel with
- Issue 5

#### Issue 8

## Title
Make participant gating explicit and test-local if Issue 3 shows config-level P1 failure

## Goal
If the participant sweep shows that the wrong MAPF group is primarily caused by test-local `COMMUNICATION_RANGE` or `MIN_PAR_PARTICIPANTS` settings, move those overrides into a dedicated config for this integration test without changing global defaults.

## Why this issue exists now
This debug document already notes that the script leaves `MIN_PAR_PARTICIPANTS` at the default and that `COMMUNICATION_RANGE` clearly changes participants. That is a bounded caller-side fix if the sweep confirms it.

## Relevant failure pattern(s)
- P1 Wrong agents enter MAPF

## Related hypothesis/hypotheses
- H1b Conflict graph != true deadlock set
- H1c `MIN_PAR_PARTICIPANTS=4`

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/rl_rvo_nav/docs/integrations/gym_env.md`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py`
- `rl_rvo_nav/config/deadlock_config.py`
- Issue 3 sweep output

## Constraints
- Only proceed if Issue 3 shows config-level gating is the dominant P1 cause.
- Do not change global defaults in `rl_rvo_nav/config/deadlock_config.py`.
- Any behavior change must include a regression check against the Issue 3 baseline.

## Deliverables
- A dedicated test-local config path for this integration test.
- Minimal caller-side wiring so the fixed debug entry point selects that config.
- Before/after participant summary showing why the test-local override was chosen.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`
- `bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh`

## Expected evidence or acceptance criteria
- The chosen MAPF group aligns better with the intended bottleneck under the test-local config.
- Deadlock resolution does not trigger at an obviously unacceptable rate relative to the sweep baseline.
- The change is isolated to this test path.

## Out of scope
- Changing conflict-graph construction.
- Changing step-order seed arbitration.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the evidence and rationale.
- Update `rl_rvo_nav/rl_rvo_nav/docs/integrations/gym_env.md` because this is a caller-side integration choice for a specific test workflow.

## Suggested labels
- `debug`
- `behavior-change`
- `regression-required`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 2
- Issue 3

## Can run in parallel with
- None; only do this if Issue 3 says config-level gating is the dominant P1 cause.

#### Issue 9

## Title
Remove step-order seed bias if Issue 2 confirms "first-detector wins" is driving P1

## Goal
Change the env to choose a deadlock trigger seed by an explicit per-step arbitration rule instead of whichever agent is encountered first in the loop.

## Why this issue exists now
H1a is a narrow, falsifiable implementation hypothesis with a bounded code path in the env step loop.

## Relevant failure pattern(s)
- P1 Wrong agents enter MAPF

## Related hypothesis/hypotheses
- H1a First-detector wins

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- Issue 2 trace output

## Constraints
- Only proceed if Issue 2 shows multiple same-step detections where loop order changes the chosen seed.
- Preserve conflict-graph construction, cooldown rules, and participant gating in this issue.
- Include a regression check against the Issue 2 trace evidence.

## Deliverables
- Step-level seed-arbitration logic with an explicit priority rule.
- A small regression artifact or test proving the chosen seed is no longer an incidental function of loop order.
- Before/after evidence for the first reproduced MAPF event.

## Validation commands
- `bash scripts/run_all.sh test-par-debug`

## Expected evidence or acceptance criteria
- The first reproduced MAPF event is explained by the explicit arbitration rule, not by loop index.
- The chosen seed is invariant for the same recorded step state.
- No unintended change is made to graph construction or participant gating.

## Out of scope
- Config tuning.
- Changing the conflict graph itself.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the before/after evidence.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because this is an env-step implementation change.
- Update `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md` because trigger arbitration is part of the caller-side integration semantics.

## Suggested labels
- `debug`
- `behavior-change`
- `regression-required`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 2
- Issue 3

## Can run in parallel with
- None; do this only if Issue 2 confirms H1a.

#### Issue 10

## Title
Fix per-agent waypoint-manager stall if Issues 4-5 confirm an individual progression bug

## Goal
Apply the smallest per-agent waypoint progression fix if the stall is caused by thresholding, force-switch behavior, or missing per-agent progress bookkeeping.

## Why this issue exists now
P2 has a distinct per-agent code path separate from tuple-group execution, and it should be fixed independently if that path is the confirmed root cause.

## Relevant failure pattern(s)
- P2 Waypoint not advancing

## Related hypothesis/hypotheses
- H2a Reach threshold too strict
- H2d Force-switch disabled or too high
- H2e Waypoint index not updated for MAPF

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/LongRangeNavi/docs/long_range_navigation.contract.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/LongRangeNavi/waypoint_manager.py`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- Issues 4-5 evidence

## Constraints
- Only proceed if Issues 4-5 point to the per-agent path, not tuple-group blocking.
- Preserve semantics outside the confirmed failure path.
- Include a regression check on both the focused tuple test and the full failing integration run.

## Deliverables
- The minimal code change needed in per-agent waypoint progression.
- Before/after stall summaries for the reproduced failing case.
- A regression artifact showing no obvious regression in waypoint completion behavior.

## Validation commands
- `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`
- `bash scripts/run_all.sh test-par-debug`

## Expected evidence or acceptance criteria
- The previously stalled agent now advances or exits for the confirmed reason.
- The fix does not create a new obvious regression in long-range waypoint completion.
- This debug document clearly records which hypothesis was fixed.

## Out of scope
- Tuple-group progression logic.
- Collision mitigation.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the fixed hypothesis and evidence.
- Update `rl_rvo_nav/LongRangeNavi/docs/long_range_navigation.contract.md` if reach-threshold or force-switch semantics changed.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because PAR-injected waypoint-manager behavior is an implementation detail there.

## Suggested labels
- `debug`
- `behavior-change`
- `regression-required`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 4
- Issue 5

## Can run in parallel with
- None; likely overlaps the same files as Issue 11.

#### Issue 11

## Title
Fix PAR tuple-group stall if Issue 5 confirms row-level blocking is the dominant cause

## Goal
Apply the smallest tuple-group progression fix if stalled episodes are caused by the "all must reach" row barrier rather than by individual waypoint-manager behavior.

## Why this issue exists now
Tuple-group execution is a separate bounded code path and should be fixed independently from per-agent progression.

## Relevant failure pattern(s)
- P2 Waypoint not advancing

## Related hypothesis/hypotheses
- H2b PAR tuple: all must reach

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/test_mapf_waypoint_tuples.py`
- Issue 5 classifier output

## Constraints
- Only proceed if Issue 5 confirms tuple-row blocking as the dominant stall class.
- Preserve non-tuple MAPF behavior.
- Include regression checks on both the tuple-focused script and the failing integration run.

## Deliverables
- The minimal tuple-group progression fix.
- Before/after blocker summaries for a reproduced stalled episode.
- A regression artifact showing tuple runs progress without breaking non-tuple behavior.

## Validation commands
- `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`
- `bash scripts/run_all.sh test-par-debug`

## Expected evidence or acceptance criteria
- The previously blocked tuple episode no longer stays on the same tuple index without a measured blocker.
- Non-tuple runs are not changed unexpectedly.
- This debug document records the exact tuple behavior change and why it was safe.

## Out of scope
- Per-agent threshold or force-switch fixes.
- Collision mitigation.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with the evidence and fixed hypothesis.
- Update `rl_rvo_nav/gym_env/docs/env.design.md` because tuple-group execution is an env implementation detail.
- Update `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md` because tuple execution is part of how gym_env consumes deadlock-resolution output.

## Suggested labels
- `debug`
- `behavior-change`
- `regression-required`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 4
- Issue 5

## Can run in parallel with
- None; likely overlaps the same files as Issue 10.

#### Issue 12

## Title
Apply one evidence-backed collision fix in the MAPF execution path

## Goal
Implement the smallest collision fix once Issues 6-7 identify whether the reproduced failure is caused by obstacle-grid mismatch or by MAPF execution safety in the env.

## Why this issue exists now
P3 should be fixed only after the repo can distinguish planner/world mismatch from execution-level safety failure.

## Relevant failure pattern(s)
- P3 Collisions

## Related hypothesis/hypotheses
- H3a No separate collision-free MAPF execution
- H3b Obstacle representation mismatch
- H3c Yielding / speed cap insufficient
- H3d RVO/ORCA and MAPF goals conflict

## Relevant files and docs to read first
- `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- `rl_rvo_nav/gym_env/docs/env.design.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md`
- `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md`
- `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`
- Issues 6-7 evidence

## Constraints
- Do not start until Issues 6-7 identify a primary collision class.
- Only one fix family per PR: obstacle/grid alignment or execution-safety tuning, not both.
- Include regression checks against P1 and P2 evidence, not only collision count.

## Deliverables
- One bounded collision fix.
- Before/after collision summaries on the reproduced case.
- Regression evidence showing participant selection and waypoint progression did not regress.

## Validation commands
- `bash scripts/run_all.sh test-par-debug --num_episodes 3`
- `python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples`

## Expected evidence or acceptance criteria
- The targeted collision class is reduced or eliminated in the reproduced case.
- P1 and P2 metrics from earlier issues do not obviously worsen.
- This debug document records which collision hypothesis was fixed and which were rejected.

## Out of scope
- Participant-selection fixes.
- Waypoint-stall fixes outside the collision root cause.

## Documentation updates required
- Update `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` with before/after evidence.
- Update `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md` if the fix changes exposed deadlock config semantics such as obstacle margin or solver-facing safety semantics.
- Update `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md` if the fix changes occupancy-grid or solver-side implementation details.
- Update `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md` if the fix changes env-side speed cap, yielding, or neighbor-exclusion behavior.

## Suggested labels
- `debug`
- `behavior-change`
- `regression-required`
- `cloud-safe-after-evidence`

## Depends on
- Issue 1
- Issue 5
- Issue 6
- Issue 7

## Can run in parallel with
- None

### 7.3 Dependency Graph

```text
1 -> 2, 4, 6
2 -> 3, 8, 9
4 -> 5, 10, 11
6 -> 7, 12
3 -> 8, 9
5 -> 10, 11, 12
7 -> 12
```

### 7.4 Recommended Execution Order

1. Issue 1
2. Issues 2, 4, and 6 in parallel
3. Issue 3 after Issue 2
4. Issue 5 after Issue 4
5. Issue 7 after Issue 6
6. Choose at most one primary P1 fix path first: Issue 8 or Issue 9
7. Choose the confirmed P2 fix path: Issue 10 or Issue 11, or both only if Issue 5 shows two distinct root causes
8. Issue 12 last, using the stabilized P1/P2 evidence as regression guards

### 7.5 Minimal AGENTS.md Additions

- Treat `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md` as the only debugging SoT for `policy_test_long_range_with_par`; append evidence here instead of creating new same-topic notes.
- For repeated debug workflows, update existing scripted entry points under `scripts/run_all.sh` or add a documented script under `scripts/`; do not make raw one-off shell commands the canonical workflow.
- Any deadlock/PAR behavior change must update the matching existing doc in the same PR: caller integration in `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md`, env implementation in `rl_rvo_nav/gym_env/docs/env.design.md`, and the relevant contract doc when public semantics change.
