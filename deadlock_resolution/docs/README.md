# Deadlock Resolution Module Docs

## Purpose

This directory contains documentation for the `deadlock_resolution` module: deadlock detection (speed-buffer and waypoint-stuck triggers) and resolution via Push and Rotate (PAR) or CBS.

## Files

- **deadlock_resolution.contract.md**: Contract for DeadlockDetector, PARCoordinator, PARExecutor (and PAREnvironment/CBSCoordinator as needed). Callers must depend only on this.
- **deadlock_resolution.design.md**: Internal design (triggers, participant selection, PAR env build, PNR solver usage).
- **cbs_implementation_notes.md**: CBS implementation notes, code/changelog, config options, and troubleshooting (viz, no solution, timeout, duplicate goals).

## Conventions

- Callers rely only on the contract; the env and mode_management document their use in `docs/integrations/`.
- When changing detection semantics or PAR API, update the contract first, then the design doc and integration notes.
