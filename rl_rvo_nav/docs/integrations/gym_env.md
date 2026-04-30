# Integration: rl_rvo_nav (core) → gym_env

## Caller and callee

- **Caller**: `rl_rvo_nav` training and test scripts (policy_train/*.py, policy_test/*.py, policy_test_with_deadlock/*.py).
- **Callee**: `gym_env` (Gym env `mrnav-v1`).

## Where and how they are used

- **Creation**: Scripts call `gym.make('mrnav-v1', world_name=..., robot_number=..., neighbors_region=..., neighbors_num=..., robot_init_mode=..., env_train=..., reward_parameter=..., goal_threshold=..., full=..., enable_long_range_nav=..., long_range_config=..., enable_deadlock_resolution=...)`. Some scripts then call `env.enable_deadlock_resolution_mode(deadlock_config_file)` if a config file is provided.
- **Reset**: `obs_list = env.reset(mode=...)` at the start of each episode; scripts use the returned obs for the first policy input.
- **Step**: In a loop, scripts get actions from the policy (one per robot), then call `obs_list, reward_list, done_list, info_list = env.step(actions, vel_type='omni')`. They use obs_list for the next policy input and reward_list/done_list for logging or early termination. When all(done_list), the episode ends and scripts may call reset again or break.
- **Render**: Scripts may call `env.render()` or `env.render(save=True, path=...)` when --render or --save is set.
- **Optional**: `env.set_test_logger(logger)` for waypoint/episode logging; `env.get_current_mode(agent_id)` for debugging.

## Dependencies and assumptions

- **Observation/action shape**: Scripts assume obs_list and action list length equal robot_number; each element matches observation_space and action_space from the env contract. Policy network must match the same shapes.
- **Step return**: Scripts assume step returns (obs_list, reward_list, done_list, info_list); no exception from step under normal inputs. When deadlock resolution is on, some agents may be in MAPF mode and their actions may be adjusted by the env before execution, but the caller still uses the same `env.step(...)` contract for all agents.
- **Determinism**: For reproducibility, scripts may set random seeds (numpy, torch, random) before creating the env and running episodes.

## Configuration and injection

- **World**: Chosen via --world_name; script resolves path or passes basename; env (or ir_sim) loads the YAML. World YAMLs are under policy_train, policy_test, policy_test_with_deadlock, pre_trained_model.
- **Model**: Scripts load checkpoint and args (e.g. pickle) from --model_path and --model_name, --arg_name; they use args for env kwargs (e.g. neighbors_region, neighbors_num, reward_parameter) and for policy config.
- **Deadlock config selection**: Deadlock-enabled scripts may pass `--deadlock_config_file` to `env.enable_deadlock_resolution_mode(...)`. For `policy_test_with_deadlock/policy_test_long_range_with_par.py`, the caller now makes the participant-gating choice explicit:
  - default PAR test path uses `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_par_test_local.json`
  - debug PAR path uses `rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/debug_policy_test_long_range_with_par.json`
  - both paths keep `COMMUNICATION_RANGE=4.0` but explicitly set `MIN_PAR_PARTICIPANTS=2` for this integration test, based on the bounded participant sweep recorded in `rl_rvo_nav/docs/debug_policy_test_long_range_with_par.md`
- **Mock/test**: Unit tests can use a minimal world YAML and small robot_number; integration tests use the same gym.make as production scripts.

## Error handling and fallback

- If env raises (e.g. invalid world file), the script fails; no fallback env. Scripts may catch and log, then exit with non-zero code.
- If step returns unexpected lengths, scripts may raise or log; contract guarantees list lengths equal robot_number.

## Conversion and validation

- Actions from policy are typically numpy or list; env accepts action per agent and normalizes to list internally if a single action is passed. No extra conversion in scripts beyond what the policy outputs.
