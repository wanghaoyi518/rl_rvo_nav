# Environment Contract (gym_env)

## Scope and goals

The `gym_env` module provides a Gym-compatible multi-robot navigation environment used for RL training and evaluation. It is the **single entry point** for simulation: observation space, action space, step semantics, reset, and optional features (deadlock resolution, long-range waypoints) are defined here.

**Out of scope**: Policy implementation, training loops, or logging format are defined in the core package and test scripts, not in this contract.

## Registered environment

- **ID**: `mrnav-v1`
- **Entry point**: `gym_env.envs:mrnav` (class `mrnav`).
- **Creation**: `gym.make('mrnav-v1', world_name=..., **kwargs)`.

## Constructor (gym.make) parameters

- **world_name** (str): Basename or path of world YAML (e.g. `mode8_long_range.yaml`). Required for map/obstacles/robots.
- **neighbors_region** (float, default 5): Neighbor region for RVO.
- **neighbors_num** (int, default 10): Max number of neighbors.
- **robot_number** (int): Number of robots (often passed by caller).
- **robot_init_mode** / **dis_mode** (int): Initial disposition mode for robot spawn (e.g. 8 for long-range scenarios).
- **env_train** (bool, default True): Training vs evaluation mode (affects dynamics/reward if applicable).
- **reward_parameter** (tuple): Reward weights (passed through to RVO reward).
- **goal_threshold** (float): Distance to goal to mark done.
- **full** (bool): Full observation flag if supported.
- **enable_long_range_nav** (bool, default False): If True, waypoints are generated via LongRangeNavi and per-agent goals follow waypoints until final goal.
- **long_range_config** (dict or LongRangeConfig): Keys include `grid_resolution`, `waypoint_min_spacing`, `reach_threshold`, `waypoint_separation_manhattan`; see LongRangeNavi contract.
- **enable_deadlock_resolution** (bool, default False): If True, step uses deadlock detection and PAR (or CBS) resolution; requires optional call to `enable_deadlock_resolution_mode(config_file)` after creation for config.
- **kwargs**: Passed to inner `ir_gym` (e.g. `circle`, `square`, `interval` for reset regions).

## Observation and action spaces

- **observation_space**: `gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)` (from inner ir_gym).
- **action_space**: `gym.spaces.Box(low=[-1,-1], high=[1,1], dtype=np.float32)` (normalized velocity).

Exact meaning of the 5-dim observation is defined by the inner env (ir_gym) and RVO; typically includes relative goal, velocity, and neighbor information.

## Step semantics

- **step(action, vel_type='diff', stop=True, **kwargs)**  
  - This is the canonical caller-facing step API for both regular RL and deadlock-enabled rollouts.
  - If deadlock resolution is enabled: delegates to `_step_with_deadlock_resolution(action)` once, computes RVO reward plus movement reward once, and advances physics once.
  - Otherwise: applies RVO dynamics and computes observation/reward/done per agent.
  - **Returns**: `(obs_list, reward_list, done_list, info_list)` — lists of length `robot_number`.  
  - **Idempotency**: Not guaranteed; step advances simulation by one timestep.

- **step_ir(action, vel_type='diff', stop=True, **kwargs)**
  - Legacy compatibility entry point.
  - When deadlock resolution is disabled, preserves the original non-deadlock rollout path.
  - When deadlock resolution is enabled, it must safely delegate to `step(...)` and must not perform a separate pre-deadlock physics transition.

## Reset

- **reset(mode=0, **kwargs)**  
  - Resets the inner env (robots, goals, obstacles from world). When long-range is enabled, rebuilds occupancy grid and per-agent waypoint managers. When deadlock is enabled, resets detector/state manager/executor state.  
  - **Returns**: Initial observation list (or first obs list consistent with step return shape).

## Deadlock resolution mode (optional)

- **enable_deadlock_resolution_mode(config_file=None)**: Enables (or re-enables) deadlock resolution and loads config from `config_file` if provided. Must be called after `gym.make` if config is not set at construction.
- **disable_deadlock_resolution_mode()**: Disables deadlock resolution; step reverts to pure RL.
- **is_in_deadlock_resolution_mode()**: Returns whether deadlock resolution is active.
- **get_current_mode(agent_id)**: Returns `'rl_rvo'` or `'mapf'` for the agent (for logging/debugging).

## Render and utilities

- **render(mode='human', save=False, path=None, i=0, **kwargs)**: Renders the current state; if save, can write figure to path.
- **set_test_logger(test_logger)**: Sets a logger reference for waypoint/episode logging (used by test scripts).

## Errors and assumptions

- **World YAML**: Must be loadable by `ir_sim.env_base`; invalid file or missing fields may raise during construction or reset.
- **action**: If a single action is passed, it is converted to a list of length 1; otherwise a list of length `robot_number` is expected for multi-robot.
- **Bounds**: The env may enforce boundaries via inner `_enforce_boundaries(action)`; violations are handled internally (no guaranteed exception contract).

## Performance and resources

- Step cost depends on robot_number, neighbors_num, and whether deadlock resolution and long-range are enabled. PAR solver calls can be expensive for large participant sets.
- No hard latency guarantee; suitable for offline training and batched evaluation.

## Versioning and compatibility

- Adding new kwargs to `gym.make` is backward compatible if defaulted.
- Changing observation shape or step return structure is breaking; callers (policy, test scripts) must be updated.
- New optional features (e.g. new trigger types) should be gated by config and not change default behavior.

## Example usage

```python
import gym
import gym_env

env = gym.make(
    'mrnav-v1',
    world_name='mode8_long_range.yaml',
    robot_number=8,
    neighbors_region=5,
    neighbors_num=10,
    robot_init_mode=8,
    env_train=False,
    enable_long_range_nav=True,
    long_range_config={'grid_resolution': 0.5, 'waypoint_min_spacing': 2.0, 'reach_threshold': 0.3},
    enable_deadlock_resolution=True
)
if enable_deadlock_resolution:
    env.enable_deadlock_resolution_mode('path/to/deadlock_config.yaml')

obs_list = env.reset(mode=8)
for t in range(max_steps):
    actions = [policy(obs) for obs in obs_list]
    obs_list, reward_list, done_list, info_list = env.step(actions)
    if all(done_list):
        break
env.close()
```

## Consumers

- **rl_rvo_nav.policy_train**: Training scripts create `mrnav-v1` and run step/reset in a loop. See [integrations/gym_env.md](../rl_rvo_nav/docs/integrations/gym_env.md).
- **rl_rvo_nav.policy_test** and **rl_rvo_nav.policy_test_with_deadlock**: Evaluation scripts create env with optional long-range and deadlock; use same step/reset contract. See same integration doc.
