import gym
import gym_env
from pathlib import Path
import pickle
import sys
from rl_rvo_nav.policy_test.post_train import post_train
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

parser = argparse.ArgumentParser(description='policy test for Stage 4 best model')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
# Stage 4 最佳模型
# parser.add_argument('--model_name', default='r4_mode7_stage4_10_2/r4_mode7_stage4_10_2_900.pt')  #   policy_dict=True    
# parser.add_argument('--arg_name', default='r4_mode7_stage4_10_2/r4_mode7_stage4_10_2')
parser.add_argument('--model_name', default='pre_train_obs_10_1/pre_train_obs_10_1_check_point_1000.pt')  #   policy_dict=True    
parser.add_argument('--arg_name', default='pre_train_obs_10_1/pre_train_obs_10_1')
# Stage 4 配置文件
parser.add_argument('--world_name', default='mode7_stage4_complex+.yaml')  # Stage 4配置文件
# parser.add_argument('--world_name', default='mode8_long_range.yaml')  # Stage 4配置文件

parser.add_argument('--render', action='store_true')
# Stage 4 使用10个机器人
parser.add_argument('--robot_number', type=int, default='10')
parser.add_argument('--num_episodes', type=int, default='10')
# Mode 7: random with distance constraint + random polygons
parser.add_argument('--dis_mode', type=int, default='7')  # 7 for Mode 7
parser.add_argument('--save', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
# 不使用checkpoint格式，直接加载模型
parser.add_argument('--policy_dict', action='store_true', default=True)
parser.add_argument('--once', action='store_true')

policy_args = parser.parse_args()

# Validate num_episodes parameter
if policy_args.num_episodes <= 0:
    print(f"Error: num_episodes must be positive, got {policy_args.num_episodes}")
    print("Setting num_episodes to 1 for testing...")
    policy_args.num_episodes = 1

cur_path = Path(__file__).parent.parent 

# model_base_path = str(cur_path) + '/' + policy_args.model_path
model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

# args from train
r = open(args_path, 'rb')
args = pickle.load(r) 

if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name 
    policy_name = 'drl_rvo_pre_train'
    
env = gym.make('mrnav-v1', world_name=policy_args.world_name, robot_number=policy_args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=policy_args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2, full=policy_args.full)

policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode) + '_mode8'

pt = post_train(env, num_episodes=policy_args.num_episodes, reset_mode=policy_args.dis_mode, render=policy_args.render, std_factor=0.00001, acceler_vel=1.0, max_ep_len=300, neighbors_region=args.neighbors_region, neighbor_num=args.neighbors_num, args=args, save=policy_args.save, show_traj=policy_args.show_traj, figure_format='eps')
pt.policy_test(policy_args.policy_type, fname_model, policy_name, result_path=str(cur_path), result_name='/result.txt', figure_save_path=cur_path / 'figure' , ani_save_path=cur_path / 'gif', policy_dict=policy_args.policy_dict, once=policy_args.once)
