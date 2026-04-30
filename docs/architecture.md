# Architecture and Module Boundaries

## Purpose

This document describes the high-level architecture of the `rl_rvo_nav` project: module responsibilities, dependency directions, and key data/control flows. It is the single source of truth for "who depends on whom" and prevents changes at the wrong layer.

## High-Level Modules

| Module | Responsibility |
|--------|----------------|
| `gym_env/` | Gym-compatible multi-robot navigation environment; wraps `ir_sim.env_base`, RVO, optional deadlock resolution and long-range waypoints. |
| `deadlock_resolution/` | Deadlock detection (speed/waypoint-stuck triggers), PAR (Push and Rotate) coordination and execution, optional CBS. |
| `mode_management/` | Mode switching (RL_RVO vs MAPF/PAR), state storage per agent, PAR status tracking. |
| `LongRangeNavi/` | Global A* path planning, waypoint sparsification, per-agent waypoint progression. |
| `python_pnr/` | PNR solver (Push and Rotate), SubMap, ISearch (A*), MAPF types; used by deadlock_resolution and LongRangeNavi. |
| `rl_rvo_nav/` (core package) | Policy training, policy test (with/without deadlock, long-range), config loading, checkpoint handling. |
| `config/` | Shared config (if any); world YAMLs live under `policy_train/`, `policy_test/`, `policy_test_with_deadlock/`, `pre_trained_model/`. |

## Dependency Directions

```
rl_rvo_nav (policy_train, policy_test, policy_test_with_deadlock)
    │
    ├── gym_env (Gym env: mrnav-v1 → ir_gym)
    │       │
    │       ├── deadlock_resolution (DeadlockDetector, PARCoordinator, PARExecutor, PAREnvironment; optional CBSCoordinator)
    │       │       └── python_pnr (PushAndRotate, SubMap, ISearch, MAPF types)
    │       │
    │       ├── mode_management (ModeController, StateManager)  [used inside gym_env when deadlock enabled]
    │       │       └── depends on deadlock_resolution (DeadlockDetector, PARCoordinator)
    │       │
    │       └── LongRangeNavi (GlobalPathPlanner, WaypointManager, LongRangeConfig)
    │               └── python_pnr (SubMap, ISearch)
    │
    └── config / world YAMLs (world_name, robot_number, etc.)
```

- **gym_env** must not depend on `rl_rvo_nav` (no policy imports inside gym_env).
- **deadlock_resolution** may use **python_pnr** and is used by **gym_env** (and thus by **mode_management** when embedded in env).
- **LongRangeNavi** may use **python_pnr**; it is used by **gym_env** for waypoint generation and progression.
- **mode_management** depends on **deadlock_resolution**; the env wires them together when `enable_deadlock_resolution=True`.

## Key Call Chains and Data Flows

### 1. Training / evaluation (no deadlock)

- Entry: `rl_rvo_nav/policy_train/train_process*.py` or `rl_rvo_nav/policy_test/policy_test*.py`.
- `gym.make('mrnav-v1', world_name=..., ...)` → `mrnav` → `ir_gym`.
- Each step: `env.step(action, vel_type='omni')` → `mrnav.step(...)` → original non-deadlock rollout path; observation/reward/done come from one RVO + movement transition.

### 2. Evaluation with long-range waypoints (no deadlock)

- Entry: e.g. `rl_rvo_nav/policy_test/policy_test_long_range.py`.
- `gym.make('mrnav-v1', ..., enable_long_range_nav=True, long_range_config={...})`.
- On reset: `ir_gym` builds occupancy grid, creates `GlobalPathPlanner`, per-agent `WaypointManager` with sparse waypoints; each step updates waypoint progress and sets `robot.goal` to current waypoint until final goal.

### 3. Evaluation with deadlock resolution (PAR)

- Entry: e.g. `rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_deadlock.py` or `policy_test_with_deadlock.py`.
- `gym.make('mrnav-v1', ..., enable_deadlock_resolution=True)`; optionally `env.enable_deadlock_resolution_mode(config_file)` after creation.
- Each step: `env.step(action, vel_type='omni')` → `mrnav.step(...)` → `ir_gym._step_with_deadlock_resolution(action)`:
  - Build agent_states / neighbor_states from robots.
  - For each agent in RL_RVO mode: `DeadlockDetector.detect_deadlock` → if true, `get_deadlock_participants` → `PARCoordinator.prepare_par_execution` (builds PAR env, calls `python_pnr` PNR solver) → `StateManager.set_par_mode` for participants → `PARExecutor` used to get MAPF waypoints / set_position.
  - Agents in MAPF mode: `PARExecutor.execute_par_step` (move to start / follow PAR path); completion or timeout → switch back to RL_RVO via state manager.
  - Waypoint managers (long-range or PAR-injected) drive `robot.goal`; observation/reward/done computed after dynamics.

### 4. Mode and state

- **StateManager** (in mode_management): per-agent mode (`rl_rvo` / `mapf`), PAR status (move_to_par_pos, par_exec, wait_for_finish), par_solution, par_path, path_index.
- **ModeController** (in mode_management): decides when to switch to PAR (`DeadlockDetector.detect_deadlock`, narrow-corridor conditions) and when to switch back (PAR complete, goal reached, timeout).
- The env uses StateManager and (when deadlock enabled) ModeController; it does not expose mode_management as a top-level API to the policy.

## Script and Config Entry Points

- **Install**: From repo root `rl_rvo_nav`: `bash setup.sh` (pip install -e . and -e ./gym_env).
- **World configs**: YAML files under `policy_train/`, `policy_test/`, `policy_test_with_deadlock/`, `pre_trained_model/` (e.g. `mode8_long_range.yaml`, `mode7_stage4_complex+.yaml`). Loaded via `world_name` passed to `gym.make`.
- **Deadlock config**: Optional JSON/YAML passed to `env.enable_deadlock_resolution_mode(config_file)`; see deadlock_resolution contract for parameters.
- **Training**: `python -m rl_rvo_nav.policy_train.train_process` (or train_process_obs_s1/s2/s3/s4, curriculum_learning_manager) with args for world, model save path, etc.
- **Test**: `python -m rl_rvo_nav.policy_test.policy_test` or `policy_test_long_range`; with deadlock: `python -m rl_rvo_nav.policy_test_with_deadlock.policy_test_long_range_with_deadlock` (or policy_test_with_deadlock, policy_test_long_range_with_cbs).

## Impact of Changes

- **gym_env interface** (observation_space, action_space, step, reset, enable_deadlock_resolution_mode): affects all training and test scripts; document in [gym_env contract](gym_env/docs/env.contract.md).
- **deadlock_resolution** (detector config, PAR coordinator/executor, CBS): affects only runs with `enable_deadlock_resolution=True`; document in [deadlock_resolution contract](deadlock_resolution/docs/deadlock_resolution.contract.md).
- **LongRangeNavi** (planner, waypoint manager, config): affects only runs with `enable_long_range_nav=True`; document in [long_range_navigation contract](LongRangeNavi/docs/long_range_navigation.contract.md).
- **mode_management**: affects only when deadlock resolution is enabled; document in [mode_management contract](mode_management/docs/mode_management.contract.md).
