# Gym Environment Module Docs

## Purpose

This directory contains documentation for the `gym_env` module, which provides simulation and Gym-compatible environments (`mrnav-v1`) for the RL RVO navigation system.

## Files

- **env.contract.md**: Contract for the environment interface (Gym API, constructor args, step/reset, deadlock and long-range options). Callers must depend only on this.
- **env.design.md**: Internal design (ir_gym, mrnav, RVO, grid building, waypoint/PAR flow).

## Integrations (this module as caller)

- **integrations/deadlock_resolution.md**: How gym_env uses deadlock_resolution and mode_management.
- **integrations/long_range_navigation.md**: How gym_env uses LongRangeNavi.

## Conventions

- Callers should rely only on `env.contract.md` and not on implementation details in `env.design.md`.
- When changing the public API or step/reset semantics, update the contract first and then the design doc.
