"""
Standalone test: load mode8_long_range.yaml, run MAPF with all three solvers (PAR, CBS, rule_based),
save waypoint tuple output to txt and render GIFs.

Requires the same environment as policy_test (gym, gym_env, numpy, etc.). CBS needs cbs-mapf.

Run from rl_rvo_nav directory (parent of gym_env):
  cd /path/to/rl_rvo_nav && python -m rl_rvo_nav.policy_test_with_deadlock.test_mapf_waypoint_tuples

Output: mapf_tuples_output/mapf_waypoint_tuples.txt (all solvers),
  mapf_tuples_<solver>.txt and mapf_tuples_<solver>.gif per solver.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Ensure imports find gym_env, config, deadlock_resolution
script_dir = Path(__file__).resolve().parent
rl_rvo_nav_root = script_dir.parent.parent
if str(rl_rvo_nav_root) not in sys.path:
    sys.path.insert(0, str(rl_rvo_nav_root))
os.chdir(str(rl_rvo_nav_root))

# env_base opens world as sys.path[0] + '/' + world_name; ensure script_dir is first so
# world_name can be "mode8_long_range.yaml" (next to this script)
def _ensure_script_dir_first_for_world():
    """Put script_dir at sys.path[0] so env_base finds mode8_long_range.yaml."""
    d = str(script_dir)
    if sys.path[0] != d:
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)

import gym
import gym_env
from config.deadlock_config import DeadlockConfig
from deadlock_resolution.cbs_coordinator import CBSCoordinator
from deadlock_resolution.rule_based_coordinator import RuleBasedSequentialCoordinator

# Default: same world as policy test (path relative to rl_rvo_nav_root after chdir)
WORLD_NAME = "rl_rvo_nav/policy_test_with_deadlock/mode8_long_range.yaml"
OUTPUT_DIR = script_dir / "mapf_tuples_output"
# If running from script dir, world is next to script
WORLD_NAME_ALT = "mode8_long_range.yaml"
ROBOT_NUMBER = 4
DIS_MODE = 8
GRID_RESOLUTION = 0.5


def _format_tuple_line(timestep: int, tuple_row: tuple) -> str:
    """Format one timestep as 't: (x0,y0)(x1,y1)...' """
    parts = "".join(f"({x:.4f},{y:.4f})" for x, y in tuple_row)
    return f"{timestep}: {parts}"


def _write_tuples_txt(out_path: Path, participant_ids: list, waypoint_tuples: list, solver_name: str):
    with open(out_path, "w") as f:
        f.write(f"# solver={solver_name} participants={participant_ids} steps={len(waypoint_tuples)}\n")
        for t, row in enumerate(waypoint_tuples):
            f.write(_format_tuple_line(t, row) + "\n")


def _render_tuples_gif(env, participant_ids: list, waypoint_tuples: list, gif_path: Path, keep_last_frames: int = 30):
    """Set robot positions to each timestep, save frame, then build GIF (match full-test config)."""
    import imageio
    import shutil
    fig_dir = OUTPUT_DIR / "frames_temp"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ir_gym = env.ir_gym
    robot_list = ir_gym.robot_list
    wp = ir_gym.world_plot
    # participant_ids order matches waypoint_tuples[t][i] -> agent participant_ids[i]
    for t, row in enumerate(waypoint_tuples):
        for i, pid in enumerate(participant_ids):
            if pid < len(robot_list):
                x, y = row[i][0], row[i][1]
                robot_list[pid].state[0, 0] = x
                robot_list[pid].state[1, 0] = y
        for r in robot_list:
            if getattr(r, "goal", None) is not None and isinstance(r.goal, (list, tuple)):
                r.goal = np.array([[float(r.goal[0])], [float(r.goal[1])]])
        wp.ax.cla()
        wp.init_plot()
        wp.draw_robots(ir_gym.components["robots"], show_goal=False)
        wp.save_gif_figure(fig_dir, t, format="png")
    images = sorted(fig_dir.glob("*.png"), key=lambda p: p.name)
    image_list = [imageio.imread(p) for p in images]
    for _ in range(keep_last_frames):
        image_list.append(imageio.imread(images[-1]))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), image_list)
    shutil.rmtree(fig_dir, ignore_errors=True)
    print(f"  GIF saved: {gif_path}")


def _override_goals_to_final(ir_gym, agent_states: dict) -> None:
    """Set each agent's goal to final waypoint (last in list) so MAPF solvers plan start->destination."""
    managers = getattr(ir_gym, "_waypoint_managers", None)
    if not isinstance(managers, dict):
        return
    for aid, st in agent_states.items():
        if aid not in managers:
            continue
        mgr = managers[aid]
        waypoints = getattr(mgr, "_waypoints", None)
        if waypoints and len(waypoints) > 0:
            last = waypoints[-1]
            st["goal"] = [float(last[0]), float(last[1])]


def _ensure_agent_states_for_solvers(agent_states: dict) -> dict:
    """Use raw agent_states as-is (list/ndarray); only fill missing goal with list [x,y] so PAR/CBS accept."""
    out = {}
    for aid, state in agent_states.items():
        st = dict(state)
        p = st.get("position")
        if p is not None:
            try:
                if hasattr(p, "flat"):
                    px, py = float(p.flat[0]), float(p.flat[1])
                else:
                    px, py = float(p[0]), float(p[1])
            except (IndexError, TypeError):
                px, py = 0.0, 0.0
        else:
            px, py = 0.0, 0.0
        g = st.get("goal")
        if g is None:
            st["goal"] = [px + 0.5, py]
        else:
            if not isinstance(g, (list, np.ndarray)):
                try:
                    st["goal"] = [float(g[0]), float(g[1])]
                except (IndexError, TypeError):
                    st["goal"] = [px + 0.5, py]
        if "velocity" not in st or st["velocity"] is None:
            st["velocity"] = [0.0, 0.0]
        out[aid] = st
    return out


def _run_solver(env, solver_name: str, agent_states: dict, participants: list) -> tuple:
    """Run one solver; return (participant_ids, waypoint_tuples) or ([], [])."""
    ir_gym = env.ir_gym
    if solver_name == "par":
        if ir_gym.par_coordinator is None:
            print("  PAR coordinator not available, skip")
            return ([], [])
        ir_gym.deadlock_config.set("DEADLOCK_SOLVER", "par")
        prep = ir_gym.par_coordinator.prepare_par_execution(agent_states, participants)
        if not prep or not getattr(prep, "success", False):
            return ([], [])
        return ir_gym.par_coordinator.get_waypoint_tuples()
    if solver_name == "cbs":
        if ir_gym.cbs_coordinator is None:
            print("  CBS coordinator not available (e.g. cbs-mapf not installed), skip")
            return ([], [])
        ir_gym.deadlock_config.set("DEADLOCK_SOLVER", "cbs")
        prep = ir_gym.cbs_coordinator.prepare_cbs_execution(agent_states, participants)
        if prep is None:
            return ([], [])
        return ir_gym.cbs_coordinator.get_waypoint_tuples()
    if solver_name == "rule_based":
        if getattr(ir_gym, "rule_based_coordinator", None) is None:
            print("  Rule-based coordinator not available, skip")
            return ([], [])
        ir_gym.deadlock_config.set("DEADLOCK_SOLVER", "rule_based")
        prep = ir_gym.rule_based_coordinator.prepare_rule_based_execution(agent_states, participants)
        if prep is None:
            return ([], [])
        return ir_gym.rule_based_coordinator.get_waypoint_tuples()
    return ([], [])


def main():
    # env_base opens world as sys.path[0] + '/' + world_name
    world_path = script_dir / "mode8_long_range.yaml"
    if world_path.exists():
        _ensure_script_dir_first_for_world()
        world_name_for_env = "mode8_long_range.yaml"
    else:
        world_path = rl_rvo_nav_root / WORLD_NAME
        if not world_path.exists():
            print(f"World file not found. Tried: {script_dir / 'mode8_long_range.yaml'}, {rl_rvo_nav_root / WORLD_NAME}")
            return
        if sys.path[0] != str(rl_rvo_nav_root):
            if str(rl_rvo_nav_root) in sys.path:
                sys.path.remove(str(rl_rvo_nav_root))
            sys.path.insert(0, str(rl_rvo_nav_root))
        world_name_for_env = WORLD_NAME

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = OUTPUT_DIR / "mapf_waypoint_tuples.txt"
    with open(out_txt, "w") as f:
        f.write(f"# MAPF waypoint tuples test world={world_path}\n\n")

    # Use a dedicated debug dir for this script so we do not create deadlock_logs/<timestamp>.
    debug_dir = OUTPUT_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        from rl_rvo_nav.policy_test_with_deadlock.deadlock_logger import reset_deadlock_logger, get_deadlock_logger
        reset_deadlock_logger()
        get_deadlock_logger(log_dir=str(debug_dir), log_level="WARNING")
    except Exception:
        pass

    # Build env (same config as policy_test_long_range_with_rule)
    try:
        pickle_path = rl_rvo_nav_root / "policy_train" / "model_save" / "pre_train"
        if pickle_path.with_suffix("").exists():
            import pickle
            with open(str(pickle_path), "rb") as r:
                args = pickle.load(r)
            neighbors_region = getattr(args, "neighbors_region", 5)
            neighbors_num = getattr(args, "neighbors_num", 10)
            reward_parameter = getattr(args, "reward_parameter", (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20))
        else:
            args = None
            neighbors_region = 5
            neighbors_num = 10
            reward_parameter = (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)
    except Exception:
        neighbors_region = 5
        neighbors_num = 10
        reward_parameter = (0.2, 0.1, 0.1, 0.2, 0.2, 1, -20, 20)

    env = gym.make(
        "mrnav-v1",
        world_name=world_name_for_env,
        robot_number=ROBOT_NUMBER,
        neighbors_region=neighbors_region,
        neighbors_num=neighbors_num,
        robot_init_mode=DIS_MODE,
        env_train=False,
        reward_parameter=reward_parameter,
        goal_threshold=0.2,
        full=False,
        enable_long_range_nav=True,
        long_range_config={
            "grid_resolution": GRID_RESOLUTION,
            "waypoint_min_spacing": 2.0,
            "reach_threshold": 0.3,
            "waypoint_separation_manhattan": 2.0,
        },
        enable_deadlock_resolution=True,
    )
    deadlock_config = DeadlockConfig()
    deadlock_config.set("DEADLOCK_SOLVER", "par")
    deadlock_config.set("REQUIRED_NON_PROGRESS_NEIGHBORS", 2)
    deadlock_config.set("GRID_RESOLUTION", GRID_RESOLUTION)
    env.ir_gym.deadlock_config = deadlock_config
    env.enable_deadlock_resolution_mode()

    # For this test we need all three solvers; ir_gym only creates one by DEADLOCK_SOLVER. Create missing coordinators.
    ir_gym = env.ir_gym
    cfg = ir_gym.deadlock_config.config if hasattr(ir_gym.deadlock_config, "config") else {}
    if getattr(ir_gym, "cbs_coordinator", None) is None:
        ir_gym.cbs_coordinator = CBSCoordinator(cfg, gym_env=ir_gym)
    if getattr(ir_gym, "rule_based_coordinator", None) is None:
        ir_gym.rule_based_coordinator = RuleBasedSequentialCoordinator(cfg, gym_env=ir_gym)

    obs = env.reset(mode=DIS_MODE)
    ts = env.ir_gym.components["robots"].total_states()
    raw_states = env.ir_gym._get_agent_states_dict(ts[0])
    agent_states = _ensure_agent_states_for_solvers(raw_states)
    # Use final goal (last waypoint) per agent so MAPF plans start->destination; otherwise robot.goal is first waypoint and PAR gets start==goal.
    _override_goals_to_final(env.ir_gym, agent_states)
    participants = list(range(len(env.ir_gym.robot_list)))
    if not participants or not agent_states:
        print("No agents or agent_states after reset")
        return

    logger = getattr(env.ir_gym, "deadlock_logger", None)
    if logger is not None:
        logger.stats["episode"] = 0
        logger.stats["step"] = 0

    for solver_name in ("par", "cbs", "rule_based"):
        print(f"Running solver: {solver_name}")
        participant_ids, waypoint_tuples = _run_solver(env, solver_name, agent_states, participants)
        if logger is not None:
            if solver_name == "cbs":
                logger.stats["step"] = 1
                logger.save_cbs_trajectory_visualization(
                    agent_states,
                    env.ir_gym.cbs_coordinator if participant_ids and waypoint_tuples else None,
                    participant_ids if participant_ids else participants,
                )
                if participant_ids and waypoint_tuples and hasattr(logger, "save_cbs_grid_debug") and env.ir_gym.cbs_coordinator is not None:
                    try:
                        logger.save_cbs_grid_debug(env.ir_gym.cbs_coordinator)
                    except Exception:
                        pass
            elif solver_name == "rule_based":
                logger.stats["step"] = 2
                logger.save_rule_based_trajectory_visualization(
                    agent_states,
                    env.ir_gym.rule_based_coordinator if participant_ids and waypoint_tuples else None,
                    participant_ids if participant_ids else participants,
                )
                if participant_ids and waypoint_tuples and hasattr(logger, "save_rule_based_debug") and getattr(env.ir_gym, "rule_based_coordinator", None) is not None:
                    try:
                        logger.save_rule_based_debug(env.ir_gym.rule_based_coordinator)
                    except Exception:
                        pass
        if not participant_ids or not waypoint_tuples:
            print(f"  No output from {solver_name}")
            with open(out_txt, "a") as f:
                f.write(f"\n# --- {solver_name} ---\n# (no solution)\n")
            continue
        # Append to combined txt
        with open(out_txt, "a") as f:
            f.write(f"\n# --- {solver_name} ---\n")
            f.write(f"# participants={participant_ids} steps={len(waypoint_tuples)}\n")
            for t, row in enumerate(waypoint_tuples):
                f.write(_format_tuple_line(t, row) + "\n")
        # Per-solver txt
        solver_txt = OUTPUT_DIR / f"mapf_tuples_{solver_name}.txt"
        _write_tuples_txt(solver_txt, participant_ids, waypoint_tuples, solver_name)
        # GIF
        gif_path = OUTPUT_DIR / f"mapf_tuples_{solver_name}.gif"
        _render_tuples_gif(env, participant_ids, waypoint_tuples, gif_path, keep_last_frames=30)
    print(f"Output dir: {OUTPUT_DIR}")
    if logger is not None:
        print(f"CBS/rule debug: {debug_dir}")
    print(f"Combined txt: {out_txt}")


if __name__ == "__main__":
    main()
