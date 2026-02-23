import gym
import gym_env
from pathlib import Path
import pickle
import argparse
from os.path import dirname, abspath
import numpy as np
import random
import torch

from rl_rvo_nav.policy_test_with_deadlock.post_train_with_deadlock import post_train_with_deadlock

# 固定随机种子，保持与带 PAR 测试一致
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='policy test for long-range waypoint navigation with CBS-based deadlock resolution')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
parser.add_argument('--model_name', default='pre_train_check_point_1000.pt')
parser.add_argument('--arg_name', default='pre_train')
# parser.add_argument('--model_name', default='pre_train_obs_10_1/pre_train_obs_10_1_check_point_1000.pt')
# parser.add_argument('--arg_name', default='pre_train_obs_10_1/pre_train_obs_10_1')
parser.add_argument('--world_name', default='mode8_long_range.yaml')

parser.add_argument('--render', action='store_true')
parser.add_argument('--robot_number', type=int, default='8')
parser.add_argument('--num_episodes', type=int, default='10')
parser.add_argument('--dis_mode', type=int, default='8')
parser.add_argument('--save', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
parser.add_argument('--policy_dict', action='store_true', default=True)
parser.add_argument('--once', action='store_true')

# Long-range specific
parser.add_argument('--long_range', action='store_true', default=True)
parser.add_argument('--grid_resolution', type=float, default=0.5)
parser.add_argument('--waypoint_spacing', type=float, default=2.0)
parser.add_argument('--reach_threshold', type=float, default=0.3)
parser.add_argument('--waypoint_separation_manhattan', type=float, default=2.0)

# Deadlock resolution with CBS (use CBS instead of PAR when deadlock is detected)
parser.add_argument('--enable_deadlock_resolution', action='store_true', default=True)
parser.add_argument('--deadlock_config_file', default=None, help='JSON with USE_CBS_INSTEAD_OF_PAR; if None, uses deadlock_cbs.json in this dir')

policy_args = parser.parse_args()

cur_path = Path(__file__).parent.parent

model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

r = open(args_path, 'rb')
args = pickle.load(r)

if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name
    policy_name = 'drl_rvo_long_range'

env = gym.make(
    'mrnav-v1',
    world_name=policy_args.world_name,
    robot_number=policy_args.robot_number,
    neighbors_region=args.neighbors_region,
    neighbors_num=args.neighbors_num,
    robot_init_mode=policy_args.dis_mode,
    env_train=False,
    random_bear=args.random_bear,
    random_radius=args.random_radius,
    reward_parameter=args.reward_parameter,
    goal_threshold=0.2,
    full=policy_args.full,
    enable_long_range_nav=policy_args.long_range,
    long_range_config={
        'grid_resolution': policy_args.grid_resolution,
        'waypoint_min_spacing': policy_args.waypoint_spacing,
        'reach_threshold': policy_args.reach_threshold,
        'waypoint_separation_manhattan': policy_args.waypoint_separation_manhattan,
    },
    enable_deadlock_resolution=policy_args.enable_deadlock_resolution
)

if policy_args.enable_deadlock_resolution:
    config_file = policy_args.deadlock_config_file
    if config_file is None:
        script_dir = Path(__file__).parent
        default_cbs = script_dir / 'deadlock_cbs.json'
        if default_cbs.exists():
            config_file = str(default_cbs)
    if config_file:
        env.enable_deadlock_resolution_mode(config_file)
    else:
        env.enable_deadlock_resolution_mode()

policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode) + '_mode8_lr_with_cbs'

pt = post_train_with_deadlock(
    env,
    num_episodes=policy_args.num_episodes,
    reset_mode=policy_args.dis_mode,
    render=policy_args.render,
    std_factor=0.001,
    acceler_vel=1.0,
    max_ep_len=1000,
    neighbors_region=args.neighbors_region,
    neighbor_num=args.neighbors_num,
    args=args,
    save=policy_args.save,
    show_traj=policy_args.show_traj,
    figure_format='eps'
)

pt.policy_test(
    policy_args.policy_type,
    fname_model,
    policy_name,
    result_path=str(cur_path),
    result_name='/result_long_range_with_cbs.txt',
    figure_save_path=cur_path / 'figure_long_range_with_cbs',
    ani_save_path=cur_path / 'gif_long_range_with_cbs',
    policy_dict=policy_args.policy_dict,
    once=policy_args.once
)

