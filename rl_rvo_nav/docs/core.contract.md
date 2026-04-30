# Core Package Contract (rl_rvo_nav)

## Scope and goals

The `rl_rvo_nav` core package provides **training** and **evaluation** entry points for DRL RVO navigation: it creates the Gym environment, loads or builds the policy, and runs training or test loops. It also defines where checkpoints, args, and result files are stored.

**Out of scope**: Environment semantics (gym_env contract), deadlock or long-range behavior (those modules’ contracts).

## Public entry points (scripts)

All are run as `python -m rl_rvo_nav.policy_train.<script>` or `python -m rl_rvo_nav.policy_test.<script>` or `python -m rl_rvo_nav.policy_test_with_deadlock.<script>` from the repository root.

### Training

- **train_process.py**, **train_process_s1.py**, **train_process_s2.py**: Basic DRL RVO training; args include world_name, model save path, etc.
- **train_process_obs_s1.py** … **train_process_obs_s4.py**: Curriculum stages (Mode 7); world and curriculum stage specific.
- **curriculum_learning_manager.py**: Optional manager for multi-stage curriculum (if used).

**Output**: Checkpoints and args pickle under `rl_rvo_nav/policy_train/model_save/` (or path given by script args).

### Evaluation (no deadlock)

- **policy_test/policy_test.py**: Short-range evaluation; args: world_name, model_path, model_name, arg_name, robot_number, num_episodes, render, save, etc.
- **policy_test/policy_test_long_range.py**: Long-range waypoint evaluation; same plus --long_range, --grid_resolution, --waypoint_spacing, --reach_threshold.

### Evaluation (with deadlock)

- **policy_test_with_deadlock/policy_test_with_deadlock.py**: Short-range with deadlock resolution.
- **policy_test_with_deadlock/policy_test_long_range_with_par.py**: Long-range with PAR-based deadlock resolution. If `--deadlock_config_file` is omitted, this script uses the caller-local config `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_par_test_local.json`.
- **policy_test_with_deadlock/policy_test_long_range_with_deadlock.py**: Long-range with deadlock resolution; --enable_deadlock_resolution, optional --deadlock_config_file.
- **policy_test_with_deadlock/policy_test_long_range_with_cbs.py**: Long-range with CBS-based resolution (if supported).

**Output**: Result text files (e.g. result_long_range_with_deadlock.txt) and figure/gif directories under the package or path specified by script; deadlock logs under policy_test_with_deadlock/deadlock_logs if configured.

### Pre-trained test

- **pre_trained_model/policy_test_pre_train.py**: Test using pre-trained checkpoint and args from pre_trained_model.

## Configuration and policy loading

- **World**: Passed as `world_name` to `gym.make('mrnav-v1', world_name=..., ...)`. YAMLs live under policy_train, policy_test, policy_test_with_deadlock, pre_trained_model.
- **Model**: Scripts accept --model_path, --model_name, --arg_name; policy is loaded (e.g. torch) and applied to observations each step.
- **Deadlock**: When --enable_deadlock_resolution, env is created with enable_deadlock_resolution=True; optional --deadlock_config_file is passed to env.enable_deadlock_resolution_mode().
  - `policy_test_long_range_with_par.py` is a special caller-side integration: when no config file is provided it applies the explicit test-local config `deadlock_par_test_local.json` instead of relying on module defaults, so participant gating for this experiment is auditable.

## Errors and assumptions

- Scripts assume they are run from a directory such that `gym_env` and `rl_rvo_nav` are importable (after pip install -e . and -e ./gym_env).
- Missing world YAML or model file yields script-level errors (file not found, etc.).
- Policy interface (input shape, output action shape) must match env observation_space and action_space; see gym_env contract.

## Versioning

- New CLI flags are backward compatible if optional.
- Changing default paths or return format of training (e.g. checkpoint naming) may break automation; document in runbook.

## Consumers

- **Users and automation**: Run the above scripts as documented in [docs/repro/runbook.md](../../docs/repro/runbook.md). Integration of the core package with the Gym env is documented in [rl_rvo_nav/docs/integrations/gym_env.md](integrations/gym_env.md).
