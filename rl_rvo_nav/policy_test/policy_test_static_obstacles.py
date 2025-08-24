#!/usr/bin/env python3
"""
Policy testing script with random static obstacles
"""

import gym
import gym_env
from pathlib import Path
import pickle
import sys
import argparse
import os
from os.path import dirname, abspath
import numpy as np
import torch
import random

# Add rl_rvo_nav to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.runtime_obstacle_manager import create_runtime_obstacle_env
from rl_rvo_nav.policy_test.post_train import post_train

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='policy test with random static obstacles')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
parser.add_argument('--model_name', default='r6_static_0/r6_static_0_check_point_1000.pt')
parser.add_argument('--arg_name', default='r6_static_0/r6_static_0')
parser.add_argument('--base_world_config', default='policy_test_world.yaml',
                   help='Base world configuration (without obstacles)')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--robot_number', type=int, default=6)
parser.add_argument('--num_episodes', type=int, default=100)
parser.add_argument('--dis_mode', type=int, default=1,
                   help='1 random - random start and goal points')

# Random static obstacle parameters
parser.add_argument('--obstacle_count', type=int, default=4,
                   help='Number of random obstacles per episode')
parser.add_argument('--obstacle_size', type=float, default=0.8,
                   help='Size of square obstacles (0.8x0.8)')
parser.add_argument('--min_obstacle_distance', type=float, default=1.5,
                   help='Minimum distance between obstacles')

parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--full', action='store_true')
parser.add_argument('--show_traj', action='store_true')
parser.add_argument('--policy_dict', action='store_true', default=True)
parser.add_argument('--once', action='store_true')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

policy_args = parser.parse_args()

# Set random seed
set_seed(policy_args.seed)

cur_path = Path(__file__).parent.parent 

# model_base_path = str(cur_path) + '/' + policy_args.model_path
model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

print("=== Random Static Obstacle Policy Testing ===")
print(f"Model path: {args_path}")
print(f"Robot number: {policy_args.robot_number}")
print(f"Episodes: {policy_args.num_episodes}")
print(f"Obstacle count per episode: {policy_args.obstacle_count}")
print(f"Obstacle size: {policy_args.obstacle_size}x{policy_args.obstacle_size}")
print(f"Minimum obstacle distance: {policy_args.min_obstacle_distance}")
print(f"Base world config: {policy_args.base_world_config}")
print("=" * 50)

# Load training args
try:
    r = open(args_path, 'rb')
    args = pickle.load(r) 
    print("Training arguments loaded successfully")
except Exception as e:
    print(f"Error loading training arguments: {e}")
    sys.exit(1)

# Model file path
if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name 
    policy_name = 'drl_rvo_static'
    
    # Check if model file exists
    if not os.path.exists(fname_model):
        print(f"Error: Model file not found: {fname_model}")
        sys.exit(1)
    print(f"Model file found: {fname_model}")

# Base world configuration path
base_world_path = str(Path(__file__).parent / policy_args.base_world_config)

if not os.path.exists(base_world_path):
    print(f"Error: Base world config file not found: {base_world_path}")
    sys.exit(1)

print(f"Base world config found: {base_world_path}")

# Create runtime obstacle environment (no yaml files!)
print("Creating runtime obstacle test environment...")
try:
    env = create_runtime_obstacle_env(
        base_world_config=base_world_path,
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
        seed=policy_args.seed
    )
    print("Runtime obstacle environment created successfully!")
except Exception as e:
    print(f"Error creating environment: {e}")
    sys.exit(1)

# Policy name for results
policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode)

# Create post-training tester
print("Initializing policy tester...")
try:
    pt = post_train(env, 
                   num_episodes=policy_args.num_episodes,
                   reset_mode=policy_args.dis_mode, 
                   render=policy_args.render,
                   std_factor=0.00001, 
                   acceler_vel=1.0, 
                   max_ep_len=300,
                   neighbors_region=args.neighbors_region, 
                   neighbor_num=args.neighbors_num, 
                   args=args,
                   save=policy_args.save, 
                   show_traj=policy_args.show_traj, 
                   figure_format='eps')
    print("Policy tester initialized successfully!")
except Exception as e:
    print(f"Error initializing policy tester: {e}")
    sys.exit(1)

# Run policy test
print(f"\nStarting policy test with {policy_args.num_episodes} episodes...")
print("Each episode will have runtime-injected random static obstacle layouts!")

try:
    pt.policy_test(policy_args.policy_type, 
                  fname_model, 
                  policy_name,
                  result_path=str(cur_path), 
                  result_name='/result_static_obstacles.txt',
                  figure_save_path=cur_path / 'figure', 
                  ani_save_path=cur_path / 'gif',
                  policy_dict=policy_args.policy_dict, 
                  once=policy_args.once)
    
    print("\nPolicy testing completed successfully!")
    print(f"Results saved to: {cur_path}/result_static_obstacles.txt")
    
except Exception as e:
    print(f"Error during policy testing: {e}")
    raise
finally:
    # Clean up environment
    if 'env' in locals():
        env.cleanup()
    print("Environment resources cleaned up!")

print("=" * 50)
print("Random static obstacle policy testing finished!")
