"""
Policy Test with Deadlock Resolution

This file provides testing functionality for RL_RVO navigation with deadlock resolution.
It allows comparison between pure RL algorithm and integrated deadlock resolution algorithm.
"""

import gym
import gym_env
from pathlib import Path
import pickle
import sys
from rl_rvo_nav.policy_test_with_deadlock.post_train_with_deadlock import post_train_with_deadlock
import argparse
import os
from os.path import dirname, abspath
import numpy as np
import random
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

# 设置固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='policy test with deadlock resolution for Stage 4 best model')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
# Stage 4 最佳模型
parser.add_argument('--model_name', default='r4_mode7_stage3_8_4/r4_mode7_stage3_8_4_900.pt')  #   policy_dict=True    
parser.add_argument('--arg_name', default='r4_mode7_stage3_8_4/r4_mode7_stage3_8_4')
# Stage 4 配置文件
parser.add_argument('--world_name', default='mode7_stage3_complex.yaml')  # Stage 4配置文件
parser.add_argument('--render', action='store_true')
# Stage 4 使用10个机器人
parser.add_argument('--robot_number', type=int, default='8')
parser.add_argument('--num_episodes', type=int, default='10')
# Mode 7: random with distance constraint + random polygons
parser.add_argument('--dis_mode', type=int, default='7')  # 7 for Mode 7
parser.add_argument('--save', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
# 不使用checkpoint格式，直接加载模型
parser.add_argument('--policy_dict', action='store_true', default=False)
parser.add_argument('--once', action='store_true')
# 死锁解决相关参数
parser.add_argument('--enable_deadlock_resolution', action='store_true', default=True)
parser.add_argument('--deadlock_config_file', default=None)

policy_args = parser.parse_args()

cur_path = Path(__file__).parent.parent 

# model_base_path = str(cur_path) + '/' + policy_args.model_path
model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

# args from train
r = open(args_path, 'rb')
args = pickle.load(r) 

if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name 
    policy_name = 'drl_rvo'
    
env = gym.make('mrnav-v1', world_name=policy_args.world_name, robot_number=policy_args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=policy_args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2, full=policy_args.full, enable_deadlock_resolution=policy_args.enable_deadlock_resolution)

# 如果启用了死锁解决，加载配置文件
if policy_args.enable_deadlock_resolution and policy_args.deadlock_config_file:
    env.enable_deadlock_resolution_mode(policy_args.deadlock_config_file)

policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode)
if policy_args.enable_deadlock_resolution:
    policy_name += '_with_deadlock'

pt = post_train_with_deadlock(env, num_episodes=policy_args.num_episodes, reset_mode=policy_args.dis_mode, render=policy_args.render, std_factor=0.00001, acceler_vel=1.0, max_ep_len=300, neighbors_region=args.neighbors_region, neighbor_num=args.neighbors_num, args=args, save=policy_args.save, show_traj=policy_args.show_traj, figure_format='eps')
pt.policy_test(policy_args.policy_type, fname_model, policy_name, result_path=str(cur_path), result_name='/result_with_deadlock.txt', figure_save_path=cur_path / 'figure_with_deadlock' , ani_save_path=cur_path / 'gif_with_deadlock', policy_dict=policy_args.policy_dict, once=policy_args.once)
