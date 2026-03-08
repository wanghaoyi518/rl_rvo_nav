#!/usr/bin/env python3
"""
Standalone test for the rule-based sequential deadlock solver.

Tests RuleBasedSequentialCoordinator.prepare_rule_based_execution(agent_states, participants)
and get_agent_path(agent_id) with synthetic agent states (no gym env). Verifies:
- Priority order: left-to-right, bottom-to-top (x then y ascending).
- Relay paths: agent i goes to agent i+1 position; last agent goes to its goal.
- Wait segment: non-first agents have repeated start waypoints before moving.

Usage:
  # From repo root (RL_RVO):
  python rl_rvo_nav/deadlock_resolution/test_rule_based_standalone.py

  # With debug output (path lengths, wait counts):
  python rl_rvo_nav/deadlock_resolution/test_rule_based_standalone.py --debug
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_RVO_NAV = _SCRIPT_DIR.parent
if str(_RL_RVO_NAV) not in sys.path:
    sys.path.insert(0, str(_RL_RVO_NAV))

try:
    from deadlock_resolution.rule_based_coordinator import RuleBasedSequentialCoordinator
except ImportError:
    RuleBasedSequentialCoordinator = None


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def run_single_agent_test(config, debug):
    """Single participant: path from position to goal."""
    print("=" * 60)
    print("TEST 1: Single agent (path = start -> goal)")
    print("=" * 60)
    agent_states = {
        0: {"position": [1.0, 2.0], "goal": [5.0, 6.0]},
    }
    participants = [0]
    print(f"Participants: {participants}")
    print(f"  Agent 0: start={agent_states[0]['position']}, goal={agent_states[0]['goal']}")

    coordinator = RuleBasedSequentialCoordinator(config, gym_env=None)
    result = coordinator.prepare_rule_based_execution(agent_states, participants)

    if result is None:
        print("FAIL: prepare_rule_based_execution returned None")
        return False
    path = coordinator.get_agent_path(0)
    if not path:
        print("FAIL: get_agent_path(0) is empty")
        return False
    print(f"  Path length: {len(path)}")
    print(f"  First: {path[0]}, Last: {path[-1]}")
    if _dist(path[0], (1.0, 2.0)) > 0.01:
        print("FAIL: path start does not match agent position")
        return False
    if _dist(path[-1], (5.0, 6.0)) > 0.01:
        print("FAIL: path end does not match goal")
        return False
    print("PASS")
    return True


def run_two_agent_relay_test(config, debug):
    """Two agents: order by (x,y). Agent0 -> pos(agent1), Agent1 -> goal(agent1). Agent1 path has wait prefix."""
    print("=" * 60)
    print("TEST 2: Two agents (relay: 0->pos(1), 1->goal(1); agent1 has wait segment)")
    print("=" * 60)
    agent_states = {
        0: {"position": [3.0, 2.0], "goal": [10.0, 2.0]},
        1: {"position": [5.0, 1.0], "goal": [8.0, 4.0]},
    }
    participants = [0, 1]
    print(f"Participants: {participants}")
    for pid in participants:
        s = agent_states[pid]
        print(f"  Agent {pid}: start={s['position']}, goal={s['goal']}")
    print("  Expected order (left-to-right, bottom-to-top): agent 1 (5,1) then agent 0 (3,2) -> no, (3,2) has smaller x. So order: 0 then 1.")
    print("  So: path0 = (3,2) -> (5,1); path1 = wait at (5,1) + (5,1) -> (8,4).")

    coordinator = RuleBasedSequentialCoordinator(config, gym_env=None)
    result = coordinator.prepare_rule_based_execution(agent_states, participants)

    if result is None:
        print("FAIL: prepare_rule_based_execution returned None")
        return False

    path0 = coordinator.get_agent_path(0)
    path1 = coordinator.get_agent_path(1)
    if not path0 or not path1:
        print("FAIL: one or both paths empty")
        return False

    print(f"  Agent 0 path length: {len(path0)}, first={path0[0]}, last={path0[-1]}")
    print(f"  Agent 1 path length: {len(path1)}, first={path1[0]}, last={path1[-1]}")

    if _dist(path0[0], (3.0, 2.0)) > 0.01:
        print("FAIL: agent 0 path start != (3,2)")
        return False
    if _dist(path0[-1], (5.0, 1.0)) > 0.01:
        print("FAIL: agent 0 path end should be pos(agent1)=(5,1)")
        return False
    if _dist(path1[-1], (8.0, 4.0)) > 0.01:
        print("FAIL: agent 1 path end should be goal(1)=(8,4)")
        return False

    n_wait = 0
    for i, wp in enumerate(path1):
        if _dist(wp, (5.0, 1.0)) <= 0.01:
            n_wait += 1
        else:
            break
    if n_wait == 0:
        print("FAIL: agent 1 path should start with wait segment (repeated (5,1))")
        return False
    print(f"  Agent 1 wait waypoints: {n_wait} (expected len(path0) = {len(path0)})")
    if n_wait != len(path0):
        print("  (wait length may differ if RULE_BASED_WAIT_WAYPOINTS is set)")
    print("PASS")
    return True


def run_three_agent_order_test(config, debug):
    """Three agents: verify (x,y) order and relay endpoints."""
    print("=" * 60)
    print("TEST 3: Three agents (order by x then y; relay + wait)")
    print("=" * 60)
    agent_states = {
        0: {"position": [2.0, 3.0], "goal": [6.0, 5.0]},
        1: {"position": [1.0, 1.0], "goal": [7.0, 2.0]},
        2: {"position": [4.0, 2.0], "goal": [5.0, 6.0]},
    }
    participants = [0, 1, 2]
    for pid in participants:
        s = agent_states[pid]
        print(f"  Agent {pid}: start={s['position']}, goal={s['goal']}")
    print("  Order by (x,y): (1,1)=agent1, (2,3)=agent0, (4,2)=agent2 -> 1, 0, 2")
    print("  path(1): (1,1)->(2,3); path(0): wait + (2,3)->(4,2); path(2): wait + (4,2)->(5,6)")

    coordinator = RuleBasedSequentialCoordinator(config, gym_env=None)
    result = coordinator.prepare_rule_based_execution(agent_states, participants)

    if result is None:
        print("FAIL: prepare_rule_based_execution returned None")
        return False

    path1 = coordinator.get_agent_path(1)
    path0 = coordinator.get_agent_path(0)
    path2 = coordinator.get_agent_path(2)
    if not path1 or not path0 or not path2:
        print("FAIL: at least one path empty")
        return False

    if _dist(path1[0], (1.0, 1.0)) > 0.01 or _dist(path1[-1], (2.0, 3.0)) > 0.01:
        print("FAIL: agent 1 path should be (1,1)->(2,3)")
        return False
    if _dist(path0[-1], (4.0, 2.0)) > 0.01:
        print("FAIL: agent 0 path end should be (4,2)")
        return False
    if _dist(path2[-1], (5.0, 6.0)) > 0.01:
        print("FAIL: agent 2 path end should be (5,6)")
        return False
    print(f"  Agent 1 path len={len(path1)}, Agent 0 len={len(path0)}, Agent 2 len={len(path2)}")
    print("PASS")
    return True


def run_failure_cases_test(config, debug):
    """Empty participants or missing position -> None."""
    print("=" * 60)
    print("TEST 4: Failure cases (empty participants, missing state)")
    print("=" * 60)

    coordinator = RuleBasedSequentialCoordinator(config, gym_env=None)

    result = coordinator.prepare_rule_based_execution({}, [0])
    if result is not None:
        print("FAIL: empty agent_states should return None")
        return False
    print("  Empty agent_states -> None: OK")

    result = coordinator.prepare_rule_based_execution({0: {"position": [1, 1], "goal": [2, 2]}}, [])
    if result is not None:
        print("FAIL: empty participants should return None")
        return False
    print("  Empty participants -> None: OK")

    result = coordinator.prepare_rule_based_execution(
        {0: {"position": [1, 1], "goal": [2, 2]}, 1: {"goal": [3, 3]}},
        [0, 1],
    )
    if result is not None:
        print("FAIL: agent 1 missing position should return None")
        return False
    print("  Missing position -> None: OK")

    print("PASS")
    return True


def run_reset_test(config, debug):
    """After reset(), get_agent_path returns empty."""
    print("=" * 60)
    print("TEST 5: reset() clears solution")
    print("=" * 60)
    agent_states = {0: {"position": [1.0, 1.0], "goal": [2.0, 2.0]}}
    participants = [0]
    coordinator = RuleBasedSequentialCoordinator(config, gym_env=None)
    coordinator.prepare_rule_based_execution(agent_states, participants)
    if not coordinator.get_agent_path(0):
        print("FAIL: path should exist before reset")
        return False
    coordinator.reset()
    if coordinator.get_agent_path(0):
        print("FAIL: path should be empty after reset")
        return False
    print("PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="Standalone rule-based solver test")
    parser.add_argument("--debug", action="store_true", help="Set DEBUG_MODE=True")
    parser.add_argument("--step", type=float, default=0.0, help="RULE_BASED_INTERP_STEP (0 = use GRID_RESOLUTION)")
    parser.add_argument("--wait", type=int, default=0, help="RULE_BASED_WAIT_WAYPOINTS (0 = use prev path length)")
    args = parser.parse_args()

    config = {
        "GRID_RESOLUTION": 0.5,
        "DEBUG_MODE": args.debug,
        "RULE_BASED_INTERP_STEP": args.step if args.step > 0 else 0,
        "RULE_BASED_WAIT_WAYPOINTS": args.wait,
    }

    if RuleBasedSequentialCoordinator is None:
        print("Import error: deadlock_resolution.rule_based_coordinator not available")
        print("Run from repo root or set PYTHONPATH to rl_rvo_nav parent directory.")
        sys.exit(1)

    ok = True
    ok &= run_single_agent_test(config, args.debug)
    print()
    ok &= run_two_agent_relay_test(config, args.debug)
    print()
    ok &= run_three_agent_order_test(config, args.debug)
    print()
    ok &= run_failure_cases_test(config, args.debug)
    print()
    ok &= run_reset_test(config, args.debug)

    print()
    print("=" * 60)
    if ok:
        print("All tests PASSED. Rule-based solver module works.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
