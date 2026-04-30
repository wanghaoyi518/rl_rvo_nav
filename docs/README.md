Project Documentation Overview
==============================

This directory contains the project-level documentation for the `rl_rvo_nav` repository. It acts as the main entry point and navigation hub for all other docs. The repository root also has a top-level [docs/](../../docs/) (doc-index, architecture summary, runbook pointer) and [scripts/run_all.sh](../../scripts/run_all.sh) for fixed run entry points.

Files in this directory:

- `doc-index.md`: Global documentation index (single source of truth for where to find what).
- `architecture.md`: High-level architecture and data/control-flow across modules.
- `repro/runbook.md`: Environment, data, and run/reproduce instructions for experiments.

Conventions:

- Module-specific documentation lives in each module's own `docs/` directory.
- Contract documents use the `*.contract.md` suffix.
- Internal design documents use the `*.design.md` suffix.
- Integration notes live under `docs/integrations/` in the caller module.

Current active direction:

- The active documentation should track the restored GitHub codebase and avoid stale experiment plans.
