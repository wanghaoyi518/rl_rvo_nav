# Mode Management Module Docs

## Purpose

This directory contains documentation for the `mode_management` module: per-agent mode state (rl_rvo vs mapf) and mode transition logic when deadlock resolution is enabled.

## Files

- **mode_management.contract.md**: Contract for StateManager and ModeController. Callers depend only on this.

## Conventions

- The module is used by gym_env when `enable_deadlock_resolution=True`; see gym_env integration doc for how modes are driven and read.
