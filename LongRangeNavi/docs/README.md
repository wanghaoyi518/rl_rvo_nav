# Long-Range Navigation Module Docs

## Purpose

This directory contains documentation for the `LongRangeNavi` module: global A* path planning and per-agent waypoint progression for long-range navigation.

## Files

- **long_range_navigation.contract.md**: Contract for LongRangeConfig, GlobalPathPlanner, and WaypointManager. Callers depend only on this.

## Conventions

- The module is used by gym_env when `enable_long_range_nav=True`; see gym_env integration doc for grid building and step flow.
