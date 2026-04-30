# Python PNR Utilities Docs

## Purpose

This directory contains documentation for the `python_pnr` module: Push and Rotate (PNR) solver, SubMap, ISearch (A*), and MAPF-related types used by deadlock_resolution and LongRangeNavi.

## Role

- **Internal**: Callers of `gym_env`, `deadlock_resolution`, or `LongRangeNavi` do not need to depend on python_pnr directly; they rely on those modules’ contracts.
- **Used by**:
  - **deadlock_resolution**: PARCoordinator uses `PushAndRotate`, `SubMap`, `MAPFSearchResult`, `Node`, `ActorSet`, `Actor`, `Point`, `ActorMove`, `MAPFConfig` for PAR solving.
  - **LongRangeNavi**: GlobalPathPlanner uses `SubMap` and `ISearch` for A* path planning.

## Files

- This README only. No public contract is required for external callers; if a stable public API for python_pnr is introduced later, add a `*.contract.md` and link it from doc-index.

## Conventions

- Changes to PNR solver interface or SubMap/ISearch affect deadlock_resolution and LongRangeNavi; update their design docs and tests when modifying python_pnr.
