# Global Documentation Index

This file is the single source of truth for where to find documentation in the `rl_rvo_nav` repository. Repository-level index (root): [../../docs/doc-index.md](../../docs/doc-index.md).

## Project-level docs

- [docs/README.md](README.md): Entry point and conventions for all docs.
- [docs/architecture.md](architecture.md): Module boundaries, dependency directions, and main call chains.
- [docs/repro/runbook.md](repro/runbook.md): Environment setup, run commands, outputs, troubleshooting.
- [docs/doc-maintenance.md](doc-maintenance.md): How to maintain and extend the documentation system.

## Module-level docs

### gym_env

- [gym_env/docs/README.md](../gym_env/docs/README.md): Module overview.
- [gym_env/docs/env.contract.md](../gym_env/docs/env.contract.md): Environment contract (Gym API, deadlock/long-range options).
- [gym_env/docs/env.design.md](../gym_env/docs/env.design.md): Internal design (ir_gym, mrnav, RVO, grid building).

### deadlock_resolution

- [deadlock_resolution/docs/README.md](../deadlock_resolution/docs/README.md): Module overview.
- [deadlock_resolution/docs/deadlock_resolution.contract.md](../deadlock_resolution/docs/deadlock_resolution.contract.md): Contract for detector, PAR coordinator, executor.
- [deadlock_resolution/docs/deadlock_resolution.design.md](../deadlock_resolution/docs/deadlock_resolution.design.md): Internal design (triggers, PAR env, PNR solver usage).

### mode_management

- [mode_management/docs/README.md](../mode_management/docs/README.md): Module overview.
- [mode_management/docs/mode_management.contract.md](../mode_management/docs/mode_management.contract.md): Contract for ModeController and StateManager.

### LongRangeNavi

- [LongRangeNavi/docs/README.md](../LongRangeNavi/docs/README.md): Module overview.
- [LongRangeNavi/docs/long_range_navigation.contract.md](../LongRangeNavi/docs/long_range_navigation.contract.md): Contract for GlobalPathPlanner and WaypointManager.

### python_pnr

- [python_pnr/docs/README.md](../python_pnr/docs/README.md): Internal PNR/SubMap/ISearch utilities (no public contract required for callers of gym_env/deadlock/LongRangeNavi).

### rl_rvo_nav (core package)

- [rl_rvo_nav/docs/README.md](../rl_rvo_nav/docs/README.md): Core package overview.
- [rl_rvo_nav/docs/core.contract.md](../rl_rvo_nav/docs/core.contract.md): Public entry points (training/test scripts, policy loading).

## Integration docs (caller → callee)

- [gym_env/docs/integrations/deadlock_resolution.md](../gym_env/docs/integrations/deadlock_resolution.md): How gym_env uses deadlock_resolution and mode_management.
- [gym_env/docs/integrations/long_range_navigation.md](../gym_env/docs/integrations/long_range_navigation.md): How gym_env uses LongRangeNavi.
- [rl_rvo_nav/docs/integrations/gym_env.md](../rl_rvo_nav/docs/integrations/gym_env.md): How policy_train and policy_test use the Gym environment.

## Design vs implementation

- [docs/design_vs_implementation.md](design_vs_implementation.md): Comparison of desired features from design/requirement notes with current codebase; lists implemented, partial, and not implemented items.

## Debug and troubleshooting

- [docs/debug_cbs_mapf_mode_analysis.md](debug_cbs_mapf_mode_analysis.md): Analysis for "no agent enters MAPF mode" when using CBS as MAPF solver (call chain, root cause hypotheses, verification steps). See also runbook troubleshooting for CBS.
- [docs/debug_policy_test_long_range_with_par.md](debug_policy_test_long_range_with_par.md): Debug log for integration test `policy_test_long_range_with_par.py` — observed patterns (wrong MAPF group, waypoint stall, collisions), cause hypotheses, and attempted fixes. No code changes; document only.
- [deadlock_resolution/docs/cbs_implementation_notes.md](../deadlock_resolution/docs/cbs_implementation_notes.md): CBS implementation notes — changes and troubleshooting (trajectory viz, numpy fixes, snap-to-free, distinct goals, timeout), config options, and checklist. Updated alongside code changes.

Keep this index updated when adding or renaming modules or documents.
