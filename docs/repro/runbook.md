# Reproduction and Runbook

## Purpose

This document is the single source of truth for: environment setup, data/world preparation, how to run training and evaluation, where outputs go, and common failures.

## Environment Setup

- **Python**: 3.x (compatible with gym 0.23.1, numpy, scipy, torch if used).
- **Install** (from repository root `rl_rvo_nav`):
  ```bash
  bash setup.sh
  ```
  This runs `pip install -e .` and `pip install -e ./gym_env`.
- **Dependencies** (from `setup.py`): matplotlib, numpy, scipy, transforms3d, gym==0.23.1, Pathlib, mpi4py, cbs-mapf. The project also uses `ir_sim` (env_base) and optionally PyTorch for policy training/test.
- **Version fixing**: Prefer a dedicated venv or conda env and record versions (e.g. `pip freeze > requirements.txt`) for reproducibility.
- **Conda environment**: For local runs in this repo, activate `conda activate rl_rvo_nav` before using the fixed entry points below.

## Data and World Preparation

- **Worlds**: No external dataset. Worlds are defined by YAML files under:
  - `rl_rvo_nav/policy_train/` (e.g. `train_world.yaml`, `mode7_stage4_complex+.yaml`)
  - `rl_rvo_nav/policy_test/`, `rl_rvo_nav/policy_test_with_deadlock/` (e.g. `mode8_long_range.yaml`, `mode7_stage4_complex+.yaml`)
  - `rl_rvo_nav/pre_trained_model/` (e.g. `test_world.yaml`, `mode8_static_corridor.yaml`)
- **Usage**: Pass `world_name=<basename>.yaml` to `gym.make('mrnav-v1', world_name=..., ...)`. The loader resolves paths relative to the process CWD or known config directories (see gym_env and ir_sim conventions).
- **Pre-trained models**: Stored under `rl_rvo_nav/policy_train/model_save/` or `rl_rvo_nav/pre_trained_model/`. Scripts take `--model_path` and `--model_name` (and `--arg_name` for saved args pickle).

## Running Experiments

All commands are run from the repository root that contains `rl_rvo_nav` (so that `gym_env` and `rl_rvo_nav` are importable). Use the following **fixed entry points** only.

### Training

- **Basic DRL RVO**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_train.train_process --world_name <world>.yaml [other args]
  ```
  Similar: `train_process_s1.py`, `train_process_s2.py` (see script for exact args).
- **Curriculum (Mode 7)**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_train.train_process_obs_s1   # Stage 1
  python -m rl_rvo_nav.rl_rvo_nav.policy_train.train_process_obs_s2   # Stage 2
  python -m rl_rvo_nav.rl_rvo_nav.policy_train.train_process_obs_s3   # Stage 3
  python -m rl_rvo_nav.rl_rvo_nav.policy_train.train_process_obs_s4   # Stage 4
  ```
  Or use `curriculum_learning_manager.py` if maintained.
- **Model/args output**: Typically under `rl_rvo_nav/policy_train/model_save/` (checkpoints and args pickle).

### Evaluation (no deadlock)

- **Short-range**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_test.policy_test --world_name <world>.yaml --model_path ... --model_name ... --arg_name ...
  ```
- **Long-range waypoints**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_test.policy_test_long_range --world_name mode8_long_range.yaml --long_range --robot_number 8 --num_episodes 10 [--render] [--save]
  ```

### Evaluation (with deadlock resolution)

- **Long-range + deadlock (PAR)**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_test_with_deadlock.policy_test_long_range_with_deadlock --world_name mode8_long_range.yaml --enable_deadlock_resolution --long_range --robot_number 8 --num_episodes 10 [--deadlock_config_file <path>] [--render] [--save]
  ```
- **Fixed debug run for `policy_test_long_range_with_par.py`**:
  ```bash
  bash scripts/run_all.sh test-par-debug
  ```
  Direct equivalent:
  ```bash
  python rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py --enable_deadlock_resolution --long_range --robot_number 8 --debug_run --deadlock_config_file rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json
  ```
  This uses the test-local debug config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json`, disables the test visualizer for bounded debug runs, and writes a dedicated artifact bundle under `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/<timestamp>/`.
- **Fixed collision-focused debug repro for `policy_test_long_range_with_par.py`**:
  ```bash
  bash scripts/run_all.sh test-par-collision-debug
  ```
  Direct equivalent:
  ```bash
  python rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py --enable_deadlock_resolution --long_range --robot_number 8 --debug_run --seed 3 --deadlock_config_file rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_aggressive.json
  ```
  This uses the collision-probe config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_aggressive.json`, which disables non-PAR yielding, lets PAR agents exclude PAR neighbors in RVO, and raises PAR tracking speed to make collision cases reproducible. In local validation, `seed=3` reproduced the same robot-robot collision at step `230` in repeated runs.
- **Fixed obstacle-collision debug repro for `policy_test_long_range_with_par.py`**:
  ```bash
  bash scripts/run_all.sh test-par-obstacle-collision-debug
  ```
  Direct equivalent:
  ```bash
  python rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/policy_test_long_range_with_par.py --enable_deadlock_resolution --long_range --robot_number 8 --debug_run --seed 19 --reach_threshold 0.15 --deadlock_config_file rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_safe_interagent.json
  ```
  This uses the obstacle probe config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/collision_probe_configs/collision_probe_obstacle_safe_interagent.json`, which restores inter-agent avoidance (`EXCLUDE_PAR_NEIGHBORS_IN_RVO=false`, `NON_PAR_YIELDING_ENABLED=true`) while keeping `MIN_PAR_PARTICIPANTS=2`, `GOAL_TOLERANCE=0.15`, and `DEADLOCK_OBSTACLE_MARGIN_CELLS=0`. Before the PAR force-switch guard fix, `seed=19` reproduced the same `robot_obstacle` collision at step `499` in repeated runs and wrote one planner-vs-simulator overlay per run. After the fix, the same command is kept as a regression check: the obstacle collision no longer reproduces on the fixed code path, and the run now times out instead of entering the old obstacle-shortcut failure.
- **Default PAR integration test run**:
  ```bash
  bash scripts/run_all.sh test-par
  ```
  This now passes the explicit caller-local config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_par_test_local.json`, which keeps `COMMUNICATION_RANGE=4.0` and sets `MIN_PAR_PARTICIPANTS=2` for this integration test without changing global deadlock defaults.
- **Participant sensitivity sweep for `policy_test_long_range_with_par.py`**:
  ```bash
  bash scripts/run_policy_test_long_range_with_par_participant_sweep.sh
  ```
  This runs a bounded `COMMUNICATION_RANGE x MIN_PAR_PARTICIPANTS` matrix using test-local configs generated from the PAR debug config and writes sweep outputs under `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/participant_sweeps/<timestamp>/`. The sweep produces `summary/participant_sweep_summary.{md,csv,json}` plus per-config console logs and run bundles. The summary includes `total_deadlock_detections`, `total_par_executions`, and `total_mode_switches` so repeated detection, actual MAPF intervention, and mode churn can be compared separately.
- **Short-range + deadlock**:
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_test_with_deadlock.policy_test_with_deadlock --world_name ... [same model/arg options]
  ```
- **Long-range + CBS** (if supported):
  ```bash
  python -m rl_rvo_nav.rl_rvo_nav.policy_test_with_deadlock.policy_test_long_range_with_cbs ...
  ```

### Visualization / logging

- Use `--render` for on-screen render and `--save` for saving figures/animations when supported by the script.
- Deadlock logs (if enabled): under `rl_rvo_nav/policy_test_with_deadlock/deadlock_logs/` or as configured by the test script.

## Output and Artifact Mapping

- **Training**: Checkpoints and args in `rl_rvo_nav/policy_train/model_save/` (or path given by `--model_path`). Naming depends on script (e.g. pre_train_check_point_1000.pt, pre_train args pickle).
- **Test metrics**: Scripts often write to `result*.txt` in the package directory (e.g. `rl_rvo_nav/result_long_range_with_deadlock.txt`) and figures to a subfolder (e.g. `figure_long_range_with_deadlock`, `gif_long_range_with_deadlock`). See each test script for exact paths.
- **Debug PAR run artifacts**: `bash scripts/run_all.sh test-par-debug` writes a single run bundle under `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_runs/<timestamp>/` with:
  - `artifacts/run_manifest.json`
  - `artifacts/result_long_range_with_par.txt`
  - `deadlock_logs/<timestamp>/...`
  - optional `deadlock_logs/<timestamp>/collision_overlays/` when a `robot_obstacle` collision is reproduced under debug mode
  - `test_logs/<timestamp>_test_with_par/...`
  - `figures/` and `gifs/`
- **Paper figures/tables**: Map experiment config (world, robot_number, model, deadlock on/off) to the corresponding result file and figure directory; document any post-processing in the same runbook or a separate reproducibility note.

## Troubleshooting

- **CUDA / GPU**: If policy uses PyTorch with CUDA, ensure drivers and torch match. For CPU-only, scripts should run without CUDA.
- **ImportError (gym_env, rl_rvo_nav, LongRangeNavi, deadlock_resolution)**: Run from the repo root so that `rl_rvo_nav` (the top-level package) and `gym_env` are on PYTHONPATH; `setup.sh` installs them in editable mode.
- **ir_sim / env_base not found**: Install the `ir_sim` package (intelligent-robot-simulator) as required by gym_env.
- **cbs-mapf or python_pnr import errors**: Ensure `pip install -e .` was run from `rl_rvo_nav` and that any nested `python_pnr` or cbs-mapf dependency is available.
- **Deadlock resolution not triggering**: Check deadlock config (TRIGGER_TYPE, SMALL_SPEED, EPISODE_START_DELAY, WAYPOINT_STUCK, etc.) and that `enable_deadlock_resolution=True` and optionally `enable_deadlock_resolution_mode(config_file)` was called.
- **No agent enters MAPF when using CBS**: (1) Confirm startup uses CBS (e.g. when running `policy_test_long_range_with_cbs.py`, the script sets `DEADLOCK_SOLVER='cbs'` in code; if you pass `--deadlock_config_file`, that JSON must not override solver to `par` or `rule_based`). (2) If CBS is used but agents still do not enter MAPF, check for "CBS: cbs_mapf not available" (then run `pip install cbs-mapf`) or "CBS: prepare_cbs_execution failed" (then check CBS logs for grid/solver failure). (3) If deadlock is never detected, no MAPF branch runs — relax trigger parameters (e.g. `REQUIRED_NON_PROGRESS_NEIGHBORS`) or check DEBUG for "triggered deadlock detection". (4) To isolate CBS: run the standalone test `python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py` (optionally with `--json path/to/cbs_ep000_step050_n001.json` and `--debug`); see [deadlock_resolution/docs/cbs_implementation_notes.md](../deadlock_resolution/docs/cbs_implementation_notes.md).
- **Long-range waypoints not updating**: Ensure `enable_long_range_nav=True` and `long_range_config` has correct grid_resolution and waypoint_min_spacing; check that world YAML and obstacles match expected layout.
- **Memory**: Large robot_number or long episodes may increase memory; reduce batch size or episode length in scripts if needed.
