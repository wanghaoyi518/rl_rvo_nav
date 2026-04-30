# CBS Implementation Notes and Troubleshooting

This document records code changes, config options, and troubleshooting for the CBS (Conflict-Based Search) path in deadlock resolution. It accompanies [deadlock_resolution.design.md](deadlock_resolution.design.md) and [deadlock_resolution.contract.md](deadlock_resolution.contract.md).

---

## 1. CBS Trajectory Visualization (deadlock_logger + ir_gym)

### 1.1 Goal

Save CBS solver trajectory figures under `deadlock_logs/<run_timestamp>/cbs_debug/` for each CBS attempt (success or failure), similar to PAR’s `par_debug/` outputs.

### 1.2 Code touch points

- **deadlock_logger.py**: `save_cbs_trajectory_visualization(agent_states, cbs_coordinator=None, participants=None)` — builds start/goal/path plot and writes `cbs_ep{ep}_step{step}_n{n}.png` and `.json`. Called from ir_gym on both CBS success and CBS failure paths.
- **ir_gym.py**: On CBS success, call `save_cbs_trajectory_visualization(agent_states, cbs_solution)` **before** `log_par_preparation` so that a serialization error in logging does not skip the viz. On CBS failure, call `save_cbs_trajectory_visualization(agent_states, None, deadlock_participants)`.

### 1.3 Fixes applied (numpy and control flow)

| Issue | Cause | Fix |
|-------|--------|-----|
| **No PNGs in run folder** | `log_par_preparation` ran before viz and raised on non–JSON-serializable `CBSCoordinator`, so the exception handler skipped `save_cbs_trajectory_visualization`. | Call `save_cbs_trajectory_visualization` before `log_par_preparation` when `use_cbs`. In `log_par_preparation`, detect CBSCoordinator and skip serializing it (log a short CBS line instead). |
| **ValueError: truth value of an array is ambiguous** | Using `or` / boolean context on numpy arrays (e.g. `participants or []`, `state.get('position') or state.get('pos')`, `goal = state.get('goal') or state.get('target')`). | Avoid any `or`/`if x` on values that may be numpy arrays. Use explicit `is None` checks and `list(_cp)` / `list(participants)` so participants is always a list. For pos/goal, use `pos = state.get('position'); if pos is None: pos = state.get('pos')` (and similarly for goal). |
| **Viz title unclear when CBS fails** | Title was generic “no solution”. | When `cbs_coordinator is None`, set title to “CBS failed (solver returned no solution)” so it is clear the solver failed rather than “solution with empty paths”. |

### 1.4 Shared plotting with PAR

PAR and CBS trajectory figures are drawn by the same helper in **deadlock_logger.py**: `_plot_mapf_trajectory_figure(starts_cont, goals_cont, cont_paths, min_x, max_x, min_y, max_y, title, resolution=0, obstacle_grid=None, extent=None, draw_rect=False)`. It draws starts (circles), goals (crosses), paths (lines or dashed start–goal), optional grid ticks by resolution, optional obstacle raster, and optional bounds rectangle. PAR calls it with `resolution`, obstacle overlay, and `draw_rect=True`; CBS calls it with `resolution=0` and `draw_rect=False`. Only the data preparation and output paths (par_debug vs cbs_debug) differ.

### 1.5 Debug trace file

- At the start of `save_cbs_trajectory_visualization`, a line is appended to `cbs_debug/_trace.txt` (timestamp + “entered”). On early return (no participants, or no all_x/all_y), a line is appended (“early_return: no participants” or “early_return: no all_x/all_y …”). On exception, the exception type and message (and full traceback) are appended. Use this to confirm the function is called and to see why it might return without writing a figure.

---

## 2. CBS Solver Returning No Solution (cbs_coordinator)

### 2.1 Observed behavior

- All CBS attempts yield “no solution” (figures show only start/goal, no path; JSON has `success: false`, `paths: {}`).
- PAR on the same scenario returns paths.

### 2.2 Root causes and fixes

| Cause | Fix in cbs_coordinator.py |
|-------|----------------------------|
| **Start or goal in obstacle** | After `_continuous_to_grid`, check that the cell is free (`grid[row][col] == 0` and not in obstacle set). If not, `snap_to_free(col, row, max_radius=4)` to a nearby free cell; if none, skip that agent or return None. |
| **Duplicate goal cells** | CBS cannot assign two agents to the same goal cell. If multiple agents share the same goal (e.g. same waypoint), after building `goals_list` run a pass: for each duplicate goal cell, assign a distinct free cell in a small ring around the goal (not in `used_goal_cells` and not in `starts_list`). If no such cell exists, return None. |
| **Planner runs indefinitely** | `cbs_mapf`’s `planner.plan()` runs in a subprocess with no timeout; on large or hard instances it can hang. | Run the planner in a separate process; main process waits on `result_queue.get(timeout=CBS_TIMEOUT_SEC)`. On timeout, terminate the child and return None. Config: `CBS_TIMEOUT_SEC` (default 15.0 seconds). |

### 2.3 Code layout for timeout

- **Module-level worker** `_cbs_plan_worker(starts_list, goals_list, static_obstacles, plan_config, result_queue)`: imports `cbs_mapf.planner.Planner`, builds planner, calls `plan(...)`, puts `("ok", result)` or `("error", str(e))` into the queue.
- **Main path**: start `multiprocessing.Process` with that worker, then `result_queue.get(timeout=timeout_sec)`. On timeout (exception from `get`), terminate and join the process, then return None. Config keys passed in `plan_config`: `CBS_MAX_ITER`, `CBS_LOW_LEVEL_MAX_ITER`, `CBS_ROBOT_RADIUS_CELLS`.

---

## 3. Config Options (CBS)

| Key | Default | Description |
|-----|---------|-------------|
| `USE_CBS_INSTEAD_OF_PAR` | false | When true, use CBSCoordinator instead of PAR for MAPF after deadlock. |
| `CBS_DEBUG` | false | When true, print CBS grid/solver messages (e.g. “No solution found”, “Missing start/goal”). |
| `DEADLOCK_OBSTACLE_MARGIN_CELLS` | 1 | Dilation applied to the shared deadlock grid (PAR and CBS). |
| `CBS_MAX_ITER` | 200 | High-level CBS iteration limit. |
| `CBS_LOW_LEVEL_MAX_ITER` | 100 | Low-level Space-Time A* iteration limit per agent. |
| `CBS_ROBOT_RADIUS_CELLS` | 0 | Extra margin in grid cells for CBS only. |
| `CBS_TIMEOUT_SEC` | 15.0 | Max seconds to wait for `planner.plan()` in subprocess; on timeout CBS is treated as failed. |

Config is read from deadlock config (e.g. `deadlock_cbs.json`) and merged with defaults in `DeadlockConfig`.

---

## 4. Troubleshooting Checklist

1. **No figures in `deadlock_logs/<run>/cbs_debug/`**  
   - Check `cbs_debug/_trace.txt`: if “entered” lines appear but no PNGs, look for “early_return” or “exception” lines.  
   - If `_trace.txt` is missing, `save_cbs_trajectory_visualization` is not called (e.g. wrong branch in ir_gym or different logger instance).

2. **All figures show “CBS failed (solver returned no solution)”**  
   - Enable `CBS_DEBUG` and re-run; look for “CBS: No solution found”, “Missing start/goal”, “Could not assign distinct goal”, or “Timeout exceeded”.  
   - Increase `CBS_MAX_ITER` / `CBS_LOW_LEVEL_MAX_ITER` if the solver is giving up; reduce them or reduce grid size if the goal is to fail fast.  
   - Ensure start/goal are in free cells (snap_to_free and distinct-goal logic should handle this; if not, check grid building and `offset_x`/`offset_y`/resolution).

3. **Process hangs and must be killed**  
   - Before timeout was added: `planner.plan()` could run forever.  
   - After: planning runs in a subprocess with `CBS_TIMEOUT_SEC`; main process will return None after timeout. If it still hangs, ensure the code path uses the timeout (subprocess + queue.get(timeout=...)).

4. **CBS paths empty while PAR has paths**  
   - Same scenario: often due to duplicate goals (two agents, same goal cell) or start/goal in obstacles. Distinct-goal and snap_to_free address this.  
   - If still empty, compare grid and coordinates used by PAR vs CBS (PAR may use a cropped submap; CBS uses full long-range grid).

---

## 5. Standalone CBS test script

A standalone script exercises the CBS module and its interface without the full gym/simulator:

- **Path**: `deadlock_resolution/test_cbs_standalone.py`
- **Purpose**: Call `CBSCoordinator.prepare_cbs_execution(agent_states, participants)` and `get_agent_path(agent_id)` with a **mock gym env** that only provides `_build_occupancy_grid_for_long_range()` and `offset_x`/`offset_y`. Use this to see what grid/start/goal are used and whether CBS returns a solution.

**Run (from repo root RL_RVO):**

```bash
python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py
```

**Options:**

- `--json <path>` — Replay starts/goals from a `cbs_debug` JSON (e.g. `deadlock_logs/<run>/cbs_debug/cbs_ep000_step050_n001.json`) to reproduce a real run on a simple grid.
- `--timeout N` — Set `CBS_TIMEOUT_SEC` (default 15).
- `--debug` — Set `CBS_DEBUG=True` so the coordinator prints internal messages.
- `--max-iter N`, `--low-level-iter N` — Pass-through to CBS planner.

**What it does:**

1. **Test 1 (synthetic)**: Builds a 15×50 free grid, two agents with distinct start and goal in continuous coords, calls `prepare_cbs_execution`, prints result (None vs self) and path lengths from `get_agent_path`.
2. **Test 2 (replay)**: If `--json` is given, loads participants, starts, and goals from the JSON, builds a 20×60 grid (resolution 0.5), and runs CBS with the same interface.

If `cbs_mapf` is not installed, the script reports “CBS: cbs_mapf not available” and “Result: None (CBS failed)”. Install with `pip install cbs-mapf` to test the full path.

**Observed result**: Replaying a real run's cbs_debug JSON (same participants, starts, goals, including duplicate goal [13.75, 4.75]) on the standalone script's empty 20x60 grid yields **CBS success** and paths of length 7 per agent. So the same inputs succeed without obstacles and fail in the full run. Conclusion: failure in the full pipeline is due to the **real grid** from `_build_occupancy_grid_for_long_range()` (obstacles, dilation, or size), not the solver or distinct-goal logic. Next steps: try a cropped/local grid around participants, or increase `CBS_TIMEOUT_SEC` / `CBS_MAX_ITER` for the real grid.

**Verification data saved when running policy_test_long_range_with_cbs**: For each CBS attempt, the run now writes under `deadlock_logs/<run>/cbs_debug/`:

- `cbs_ep{ep}_step{step}_n{n}.json` — participants, starts/goals in **continuous** space, paths (same as before).
- `cbs_ep{ep}_step{step}_n{n}_grid.json` — **real grid** used at CBS init: `grid` (2D list, 0=free, 1=obstacle), `resolution`, `offset_x`, `offset_y`, `grid_height`, `grid_width`, `obstacle_count`, `order_ids`, `starts_list` and `goals_list` in **grid coordinates** (col, row) passed to the planner. This allows replay with the same grid in the standalone test or external scripts to verify why CBS failed or succeeded.

---

## 6. References

- [deadlock_resolution.design.md](deadlock_resolution.design.md) — grid convention, boundary points, obstacle margin.
- [docs/debug_cbs_mapf_mode_analysis.md](../../docs/debug_cbs_mapf_mode_analysis.md) — “no agent enters MAPF” and CBS config/dependency.
- [gym_env/docs/integrations/deadlock_resolution.md](../../gym_env/docs/integrations/deadlock_resolution.md) — step flow and when CBS is invoked.
