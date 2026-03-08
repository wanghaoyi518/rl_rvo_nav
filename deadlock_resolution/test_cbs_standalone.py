#!/usr/bin/env python3
"""
Standalone test for the CBS module and its interface.

Tests CBSCoordinator.prepare_cbs_execution(agent_states, participants) and
get_agent_path(agent_id) with a mock gym env (no real simulator). Use this to
see exactly what grid/start/goal are passed to the planner and whether CBS
returns a solution.

Usage:
  # From repo root (RL_RVO):
  python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py

  # With a cbs_debug JSON to reproduce a real run (starts/goals from file):
  python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py --json path/to/cbs_ep000_step050_n001.json

  # Optional: shorter timeout and debug on
  python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py --timeout 10 --debug

  # Run CBS with the exact grid + start/goal from a saved _grid.json (real run data):
  python rl_rvo_nav/deadlock_resolution/test_cbs_standalone.py --grid-json rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logs/20260226_171333/cbs_debug/cbs_ep000_step050_n001_grid.json
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import deadlock_resolution: the dir that contains the deadlock_resolution package
_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_RVO_NAV = _SCRIPT_DIR.parent
if str(_RL_RVO_NAV) not in sys.path:
    sys.path.insert(0, str(_RL_RVO_NAV))

try:
    from deadlock_resolution.cbs_coordinator import CBSCoordinator
except ImportError:
    CBSCoordinator = None


def grid_to_continuous(col, row, resolution, offset_x, offset_y):
    """Convert grid (col, row) to continuous (x, y) center of cell."""
    x = offset_x + (int(col) + 0.5) * resolution
    y = offset_y + (int(row) + 0.5) * resolution
    return [x, y]


class MockGymEnv:
    """Minimal gym env that provides only what CBSCoordinator needs for grid building."""

    def __init__(self, grid, resolution, world_w, world_h, offset_x=0.0, offset_y=0.0):
        """
        Args:
            grid: 2D list, grid[row][col], 0 = free, non-zero = obstacle.
            resolution: World units per cell (e.g. 0.5).
            world_w, world_h: World size in world units (used by some callers).
            offset_x, offset_y: World origin for continuous <-> grid mapping.
        """
        self._grid = grid
        self._resolution = float(resolution)
        self._world_w = world_w
        self._world_h = world_h
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)

    def _build_occupancy_grid_for_long_range(self):
        return self._grid, self._resolution, self._world_w, self._world_h


def make_simple_grid(rows=15, cols=50, resolution=0.5, add_wall=False):
    """Build a rectangular grid; optionally add a horizontal wall."""
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    if add_wall:
        wall_row = rows // 2
        for c in range(10, cols - 10):
            grid[wall_row][c] = 1
    world_w = cols * resolution
    world_h = rows * resolution
    return grid, world_w, world_h


def run_synthetic_test(config, timeout_sec, cbs_debug):
    """Test with synthetic two-agent scenario on a simple grid."""
    print("=" * 60)
    print("TEST 1: Synthetic (2 agents, simple grid)")
    print("=" * 60)
    resolution = 0.5
    grid, world_w, world_h = make_simple_grid(rows=15, cols=50, resolution=resolution, add_wall=False)
    rows, cols = len(grid), len(grid[0])
    obstacle_count = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    print(f"Grid: {rows} x {cols}  (rows x cols)")
    print(f"Resolution: {resolution}, World: {world_w} x {world_h}")
    print(f"Obstacle cells: {obstacle_count}")

    mock_env = MockGymEnv(grid, resolution, int(world_w), int(world_h), offset_x=0.0, offset_y=0.0)
    agent_states = {
        0: {"position": [2.5, 1.25], "goal": [22.5, 1.25]},
        1: {"position": [4.5, 1.25], "goal": [23.5, 1.25]},
    }
    participants = [0, 1]
    print(f"Participants: {participants}")
    for pid in participants:
        s = agent_states[pid]
        print(f"  Agent {pid}: start={s['position']}, goal={s['goal']}")

    cfg = dict(config)
    cfg["CBS_DEBUG"] = cbs_debug
    cfg["CBS_TIMEOUT_SEC"] = timeout_sec
    coordinator = CBSCoordinator(cfg, gym_env=mock_env)
    result = coordinator.prepare_cbs_execution(agent_states, participants)

    if result is None:
        print("Result: None (CBS failed)")
        return
    print("Result: self (CBS success)")
    for pid in participants:
        path = result.get_agent_path(pid)
        print(f"  Agent {pid} path length: {len(path)}")
        if path:
            print(f"    First: {path[0]}, Last: {path[-1]}")


def run_json_replay_test(json_path, config, timeout_sec, cbs_debug):
    """Replay starts/goals from a cbs_debug JSON (e.g. from a real run)."""
    print("=" * 60)
    print("TEST 2: Replay from JSON")
    print("=" * 60)
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return
    with open(path) as f:
        data = json.load(f)
    participants = data.get("participants", [])
    starts = data.get("starts", {})
    goals = data.get("goals", {})
    if not participants or not starts or not goals:
        print("JSON missing participants/starts/goals")
        return

    resolution = 0.5
    rows, cols = 20, 60
    grid, world_w, world_h = make_simple_grid(rows=rows, cols=cols, resolution=resolution, add_wall=False)
    obstacle_count = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] != 0)
    print(f"Grid: {rows} x {cols}")
    print(f"Resolution: {resolution}, World: {world_w} x {world_h}")
    print(f"Obstacle cells: {obstacle_count}")
    print(f"JSON: episode={data.get('episode')}, step={data.get('step')}, success={data.get('success')}")

    mock_env = MockGymEnv(grid, resolution, int(world_w), int(world_h), offset_x=0.0, offset_y=0.0)
    agent_states = {}
    for pid in participants:
        pid_str = str(pid)
        if pid_str in starts and pid_str in goals:
            agent_states[pid] = {
                "position": starts[pid_str],
                "goal": goals[pid_str],
            }
    if len(agent_states) != len(participants):
        print("Could not build agent_states for all participants")
        return
    print(f"Participants: {participants}")
    for pid in participants:
        s = agent_states[pid]
        print(f"  Agent {pid}: start={s['position']}, goal={s['goal']}")

    cfg = dict(config)
    cfg["CBS_DEBUG"] = cbs_debug
    cfg["CBS_TIMEOUT_SEC"] = timeout_sec
    coordinator = CBSCoordinator(cfg, gym_env=mock_env)
    result = coordinator.prepare_cbs_execution(agent_states, participants)

    if result is None:
        print("Result: None (CBS failed)")
        return
    print("Result: self (CBS success)")
    for pid in participants:
        path = result.get_agent_path(pid)
        print(f"  Agent {pid} path length: {len(path)}")
        if path:
            print(f"    First: {path[0]}, Last: {path[-1]}")


def run_grid_json_test(grid_json_path, config, timeout_sec, cbs_debug):
    """Run CBS using the exact grid and start/goal from a saved _grid.json (real run data)."""
    print("=" * 60)
    print("TEST 3: Real grid from _grid.json")
    print("=" * 60)
    path = Path(grid_json_path)
    if not path.exists():
        print(f"File not found: {path}")
        return
    with open(path) as f:
        data = json.load(f)
    grid = data.get("grid")
    resolution = float(data.get("resolution", 0.5))
    offset_x = float(data.get("offset_x", 0.0))
    offset_y = float(data.get("offset_y", 0.0))
    grid_height = int(data.get("grid_height", 0))
    grid_width = int(data.get("grid_width", 0))
    order_ids = data.get("order_ids", [])
    starts_list = data.get("starts_list", [])
    goals_list = data.get("goals_list", [])
    obstacle_count = int(data.get("obstacle_count", 0))
    if not grid or not order_ids or not starts_list or not goals_list:
        print("JSON missing grid, order_ids, starts_list, or goals_list")
        return
    if len(starts_list) != len(order_ids) or len(goals_list) != len(order_ids):
        print("Mismatch: starts_list/goals_list length != order_ids")
        return

    rows, cols = len(grid), len(grid[0]) if grid else 0
    world_w = cols * resolution
    world_h = rows * resolution
    print(f"Grid: {rows} x {cols}  (from file)")
    print(f"Resolution: {resolution}, offset: ({offset_x}, {offset_y})")
    print(f"Obstacle cells: {obstacle_count}")
    print(f"Episode: {data.get('episode')}, Step: {data.get('step')}, n: {data.get('n')}")

    mock_env = MockGymEnv(grid, resolution, int(world_w), int(world_h), offset_x=offset_x, offset_y=offset_y)
    agent_states = {}
    for i, pid in enumerate(order_ids):
        col_s, row_s = starts_list[i][0], starts_list[i][1]
        col_g, row_g = goals_list[i][0], goals_list[i][1]
        agent_states[pid] = {
            "position": grid_to_continuous(col_s, row_s, resolution, offset_x, offset_y),
            "goal": grid_to_continuous(col_g, row_g, resolution, offset_x, offset_y),
        }
    participants = order_ids
    print(f"Participants: {participants}")
    for pid in participants:
        s = agent_states[pid]
        print(f"  Agent {pid}: start={s['position']}, goal={s['goal']}")

    cfg = dict(config)
    cfg["CBS_DEBUG"] = cbs_debug
    cfg["CBS_TIMEOUT_SEC"] = timeout_sec
    coordinator = CBSCoordinator(cfg, gym_env=mock_env)
    result = coordinator.prepare_cbs_execution(agent_states, participants)

    if result is None:
        print("Result: None (CBS failed)")
        return
    print("Result: self (CBS success)")
    for pid in participants:
        path = result.get_agent_path(pid)
        print(f"  Agent {pid} path length: {len(path)}")
        if path:
            print(f"    First: {path[0]}, Last: {path[-1]}")


def main():
    parser = argparse.ArgumentParser(description="Standalone CBS module test")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to cbs_debug JSON (e.g. cbs_ep000_step050_n001.json) to replay starts/goals on empty grid",
    )
    parser.add_argument(
        "--grid-json",
        type=str,
        default=None,
        help="Path to cbs_*_grid.json to run CBS with the exact real grid and start/goal from a run",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="CBS_TIMEOUT_SEC")
    parser.add_argument("--debug", action="store_true", help="Set CBS_DEBUG=True")
    parser.add_argument("--max-iter", type=int, default=200, help="CBS_MAX_ITER")
    parser.add_argument("--low-level-iter", type=int, default=100, help="CBS_LOW_LEVEL_MAX_ITER")
    args = parser.parse_args()

    config = {
        "CBS_MAX_ITER": args.max_iter,
        "CBS_LOW_LEVEL_MAX_ITER": args.low_level_iter,
        "CBS_ROBOT_RADIUS_CELLS": 0,
        "CBS_TIMEOUT_SEC": args.timeout,
        "CBS_DEBUG": args.debug,
    }

    if CBSCoordinator is None:
        print("Import error: deadlock_resolution.cbs_coordinator not available (run from repo root or set PYTHONPATH)")
        sys.exit(1)

    if args.grid_json:
        run_grid_json_test(args.grid_json, config, args.timeout, args.debug)
        return

    run_synthetic_test(config, args.timeout, args.debug)
    print()

    if args.json:
        run_json_replay_test(args.json, config, args.timeout, args.debug)
    else:
        print("Tip: pass --json path/to/cbs_ep000_step050_n001.json to replay starts/goals on empty grid.")
        print("      pass --grid-json path/to/cbs_ep000_step050_n001_grid.json to use the real grid from a run.")


if __name__ == "__main__":
    main()
