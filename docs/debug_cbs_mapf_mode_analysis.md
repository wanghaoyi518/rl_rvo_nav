# CBS as MAPF Solver — Debug Analysis (No Agent Enters MAPF Mode)

## Step 0: Document-First Positioning

### 1) Documents Read

| Document | Path | Key content |
|----------|------|-------------|
| doc-index | `rl_rvo_nav/docs/doc-index.md` | Index to architecture, module docs, integrations |
| architecture | `rl_rvo_nav/docs/architecture.md` | gym_env → deadlock_resolution (PAR + optional CBS), mode_management; call chain for deadlock/MAPF |
| deadlock_resolution README | `rl_rvo_nav/deadlock_resolution/docs/README.md` | Module overview |
| deadlock_resolution contract | `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.contract.md` | Detector, PARCoordinator, PARExecutor, CBSCoordinator (optional); agent_states, MAPFSearchResult |
| deadlock_resolution design | `rl_rvo_nav/deadlock_resolution/docs/deadlock_resolution.design.md` | Triggers, PAR/CBS, file layout |
| gym_env integration | `rl_rvo_nav/gym_env/docs/integrations/deadlock_resolution.md` | Step path: detect → participants → prepare_par_execution / CBS → set_par_mode, waypoint injection |
| RL_CBS_data_pipeline | `RL_CBS_data_pipeline.md` | CBS config application, failure point #1 (config overwrite — already fixed in code) |

### 2) Contract Constraints Relevant to Bug

- **Interface**: When deadlock is detected, env calls either `PARCoordinator.prepare_par_execution` or `CBSCoordinator.prepare_cbs_execution(agent_states, deadlock_participants)`. CBS returns `self` on success or `None` on failure; paths via `get_agent_path(agent_id)`.
- **Config**: `USE_CBS_INSTEAD_OF_PAR` must be True and loaded **before** the second `_initialize_deadlock_modules()` so that `cbs_coordinator` is created. Contract: config loaded in `enable_deadlock_resolution_mode(config_file)` and not overwritten (only create `DeadlockConfig()` when `self.deadlock_config is None`).
- **Dependency**: CBSCoordinator uses `cbs_mapf.planner.Planner` (external package `cbs-mapf`). If import fails, `prepare_cbs_execution` returns `None`.
- **Grid**: CBS uses `gym_env._build_occupancy_grid_for_long_range()`; same workspace as long-range/PAR.

### 3) Bug Classification

- **(a) Implementation vs contract**: Possible. Contract says “CBS returns self or None”; implementation does, but when it returns `None` the env silently keeps RL_RVO (no agent enters MAPF). So behavior is consistent with contract, but the **reason** for repeated None (e.g. import failure, no solution) is not surfaced to the user.
- **(b) Integration**: Possible. Integration doc says “prepare_par_execution” and “if solution success” set MAPF; for CBS, “solution success” = `cbs_solution` truthy. If `prepare_cbs_execution` always returns None (e.g. missing `cbs-mapf`), integration correctly does not switch to MAPF, but the **caller does not log why**.
- **(c) Contract incomplete**: Minor. Contract could explicitly state: “When CBS is enabled, failure of prepare_cbs_execution (import/grid/solver) yields None; caller should not assume MAPF mode is set.”
- **(d) Environment/config/runbook**: Likely. “No agent enters MAPF” is consistent with: (1) `cbs-mapf` not installed → ImportError in CBSCoordinator → always None; (2) config file not found so `USE_CBS_INSTEAD_OF_PAR` never True; (3) deadlock never detected (trigger/cooldown), so MAPF branch never runs.

**Conclusion**: Primarily **(d)** environment/config/runbook (e.g. missing `cbs-mapf` or config not loaded), with **(a)/(b)** improved by adding explicit logging when CBS is chosen but solution is None so that contract/integration behavior is observable.

---

## Step 1: Call Chain and Root Cause Hypotheses

### 1) Call Chain (A = deadlock_resolution + CBS, B = gym_env)

- **Entry**: `python -m rl_rvo_nav.policy_test_with_deadlock.policy_test_long_range_with_cbs` (default `--enable_deadlock_resolution` and `--long_range`).
- **Env creation**: `gym.make('mrnav-v1', ..., enable_deadlock_resolution=True)` → `ir_gym.__init__` → `_initialize_deadlock_modules()` (creates `deadlock_config` with defaults, `cbs_coordinator=None` when `USE_CBS_INSTEAD_OF_PAR` false).
- **CBS config**: Script sets `config_file` to `policy_test_with_deadlock/deadlock_cbs.json` if exists → `env.enable_deadlock_resolution_mode(config_file)` → `deadlock_config.load_from_file(config_file)` → `_initialize_deadlock_modules()` again (config preserved) → `use_cbs=True` → `cbs_coordinator = CBSCoordinator(...)`.
- **Per step**: `env.step(action)` → `_step_with_deadlock_resolution` → for each agent in `rl_rvo`: `detect_deadlock` → if True, `get_deadlock_participants` → `prepare_cbs_execution(agent_states, deadlock_participants)` (if `use_cbs` and `cbs_coordinator`) → if `cbs_solution` truthy, validate paths → `set_par_mode` + waypoint injection; else `continue` (no MAPF).

**Key files**:
- `rl_rvo_nav/gym_env/gym_env/envs/ir_gym.py`: `_initialize_deadlock_modules` (lines 593–674), `enable_deadlock_resolution_mode` (676–687), `_step_with_deadlock_resolution` (704–956, CBS branch 765–867).
- `rl_rvo_nav/deadlock_resolution/cbs_coordinator.py`: `prepare_cbs_execution` (39–156), `from cbs_mapf.planner import Planner` (116).

### 2) Minimal Reproduction Path

1. From repo root (so `rl_rvo_nav` and `gym_env` are importable):  
   `cd /home/haoyiwang/Desktop/RL_RVO && python -m rl_rvo_nav.policy_test_with_deadlock.policy_test_long_range_with_cbs --world_name mode8_long_range.yaml --robot_number 8 --num_episodes 1 --once`
2. **Observation points**:  
   - After env creation: is “Using CBS for MAPF” or “Using PAR for MAPF” printed?  
   - When deadlock is detected: is “CBS: …” or “CBS INIT: Agent … path length” printed, or “CBS: prepare_cbs_execution failed, keeping RL_RVO”?

### 3) Root Cause Hypotheses (Ordered by Likelihood)

| # | Hypothesis | Location to check | Verification |
|---|------------|--------------------|---------------|
| 1 | **cbs_mapf import fails** → `prepare_cbs_execution` always returns None | `cbs_coordinator.py` L116–119 | Run script; if you see `CBS: cbs_mapf not available: ...` then `pip install cbs-mapf` (and deps). Or add at startup: try import and print “CBS solver: available” / “CBS solver: not available”. |
| 2 | **CBS config not applied** (use_cbs stays False) | `ir_gym.enable_deadlock_resolution_mode` (config_file None or load after init overwrite), `_initialize_deadlock_modules` use_cbs read | After `enable_deadlock_resolution_mode(config_file)` print `deadlock_config.get('USE_CBS_INSTEAD_OF_PAR')` and whether `cbs_coordinator is not None`. |
| 3 | **Deadlock never detected** so MAPF branch never runs | `deadlock_detector.detect_deadlock` (trigger, cooldown, waypoint history) | Existing DEBUG prints in ir_gym (“DEBUG: Agent … triggered deadlock detection”); if these never appear, relax trigger or check step_counter / waypoint updates. |

---

## Next Steps (Step 2 / 3)

- Add **diagnostic prints**: (1) At end of `_initialize_deadlock_modules`: “Using CBS for MAPF” vs “Using PAR for MAPF”; (2) In `_step_with_deadlock_resolution` when `use_cbs and not cbs_solution`: one-line message so user sees CBS failed.
- Optionally: at script or env init, try `from cbs_mapf.planner import Planner` and print availability.
- After fixing: update contract/design/integration/runbook per Step 3 (CBS failure semantics, dependency, troubleshooting).

---

## Diagnosing "CBS: No solution found"

When deadlock is detected but CBS returns no solution:

1. **Enable STA* debug output**  
   In your deadlock config (e.g. `deadlock_cbs.json`) set `"CBS_DEBUG": true`. Then re-run; the `cbs-mapf` low-level STA* will print messages such as "Open set is empty" (no path) or path length. This shows whether failure is in the low-level single-agent planner or in CBS high-level search.

2. **Single-process CBS**  
   The coordinator calls `planner.plan(..., max_process=1)` to avoid multiprocessing spawn issues (e.g. under some test runners or IDEs). If you still see no solution, the cause is the map/agents/constraints, not process count.

3. **Typical causes of no path**  
   - Start or goal inside (or too close to) dilated obstacles: check that start/goal grid cells are free in the same grid used by CBS.  
   - **Obstacle margin (shared)**: Deadlock config `DEADLOCK_OBSTACLE_MARGIN_CELLS` dilates the long-range grid used by both PAR and CBS; set to 1 or more for a safety margin. CBS can add extra margin with `CBS_ROBOT_RADIUS_CELLS` (default 0).  
   - Grid extent: CBS grid is built from obstacle bbox plus boundary points; ensure all start/goal (col, row) lie inside that extent (see `deadlock_resolution.design.md`).
