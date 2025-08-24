#!/usr/bin/env python3
"""
Training script with random static obstacles
"""

import os
import sys
import gym
import pickle
import shutil
import gym_env
import argparse
import random
import numpy as np
import yaml
from torch import nn
from pathlib import Path

# Add rl_rvo_nav to path
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.runtime_obstacle_manager import create_runtime_obstacle_env
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac

# path set
cur_path = Path(__file__).parent

# default
counter = 0

parser = argparse.ArgumentParser(description='drl rvo with random static obstacles parameters')

par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--base_world_config', default='train_world.yaml', 
                    help='Base world configuration (without obstacles)')
par_env.add_argument('--robot_number', type=int, default=6)
par_env.add_argument('--init_mode', default=2)
par_env.add_argument('--reset_mode', default=2)
par_env.add_argument('--mpi', default=False)

# Random static obstacle parameters
par_env.add_argument('--obstacle_count', type=int, default=4, 
                    help='Number of random obstacles per episode')
par_env.add_argument('--obstacle_size', type=float, default=0.8, 
                    help='Size of square obstacles (0.8x0.8)')
par_env.add_argument('--min_obstacle_distance', type=float, default=1.5, 
                    help='Minimum distance between obstacles')

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=5)
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), nargs='+')
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=True)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)

par_policy = parser.add_argument_group('par policy', 'policy parameters') 
par_policy.add_argument('--state_dim', default=6)
par_policy.add_argument('--rnn_input_dim', default=8)
par_policy.add_argument('--rnn_hidden_dim', default=256)
par_policy.add_argument('--trans_input_dim', default=8)
par_policy.add_argument('--trans_max_num', default=10)
par_policy.add_argument('--trans_nhead', default=1)
par_policy.add_argument('--trans_mode', default='attn')
par_policy.add_argument('--hidden_sizes_ac', default=(256, 256))
par_policy.add_argument('--drop_p', type=float, default=0)
par_policy.add_argument('--hidden_sizes_v', type=tuple, default=(256, 256))
par_policy.add_argument('--activation', default=nn.ReLU)
par_policy.add_argument('--output_activation', default=nn.Tanh)
par_policy.add_argument('--output_activation_v', default=nn.Identity)
par_policy.add_argument('--use_gpu', action='store_true')   
par_policy.add_argument('--rnn_mode', default='biGRU')

par_train = parser.add_argument_group('par train', 'train parameters') 
par_train.add_argument('--pi_lr', type=float, default=4e-6)
par_train.add_argument('--vf_lr', type=float, default=5e-5)
par_train.add_argument('--train_epoch', type=int, default=1000)
par_train.add_argument('--steps_per_epoch', type=int, default=500)
par_train.add_argument('--max_ep_len', default=150)
par_train.add_argument('--gamma', default=0.99)
par_train.add_argument('--lam', default=0.97)
par_train.add_argument('--clip_ratio', default=0.2)
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)
par_train.add_argument('--target_kl', default=0.05)
par_train.add_argument('--render', action='store_true')
par_train.add_argument('--render_freq', type=int, default=50)
par_train.add_argument('--con_train', action='store_true')
par_train.add_argument('--seed', type=int, default=7)
par_train.add_argument('--save_freq', type=int, default=50)
par_train.add_argument('--save_figure', action='store_true')
par_train.add_argument('--figure_save_path', default='figure')
par_train.add_argument('--save_path', default='model_save/')
par_train.add_argument('--save_name', default='r')
par_train.add_argument('--load_path', default='model_save/')
par_train.add_argument('--load_name', default='')
par_train.add_argument('--save_result', action='store_true')
par_train.add_argument('--lr_decay_epoch', type=int, default=1000)
par_train.add_argument('--max_update_num', type=int, default=10)

args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

print("=== Random Static Obstacle Training Configuration ===")
print(f"Robot number: {args.robot_number}")
print(f"Obstacle count per episode: {args.obstacle_count}")
print(f"Obstacle size: {args.obstacle_size}x{args.obstacle_size}")
print(f"Minimum obstacle distance: {args.min_obstacle_distance}")
print(f"Training epochs: {args.train_epoch}")
print(f"Base world config: {args.base_world_config}")
print("=" * 50)

# decide the model path and model name 
model_path_check = args.save_path + args.save_name + str(args.robot_number) + '_static_{}'
model_name_check = args.save_name + str(args.robot_number) + '_static_{}'
while os.path.isdir(model_path_check.format(counter)):
    counter += 1

model_abs_path = model_path_check.format(counter) + '/'
model_name = model_name_check.format(counter)

load_fname = args.load_path + args.load_name

print(f"Model will be saved to: {model_abs_path}")
print(f"Model name: {model_name}")

# Base world configuration path
base_world_path = str(cur_path / args.base_world_config)

# Create runtime obstacle environments (no yaml files!)
print("Creating runtime obstacle training environment...")
env = create_runtime_obstacle_env(
    base_world_config=base_world_path,
    robot_number=args.robot_number,
    neighbors_region=args.neighbors_region, 
    neighbors_num=args.neighbors_num,
    robot_init_mode=args.init_mode, 
    env_train=args.env_train,
    random_bear=args.random_bear, 
    random_radius=args.random_radius,
    reward_parameter=args.reward_parameter, 
    full=args.full
)

print("Creating runtime obstacle test environment...")
test_env = create_runtime_obstacle_env(
    base_world_config=base_world_path,
    robot_number=args.robot_number,
    neighbors_region=args.neighbors_region, 
    neighbors_num=args.neighbors_num,
    robot_init_mode=args.init_mode, 
    env_train=False,
    random_bear=args.random_bear, 
    random_radius=args.random_radius,
    reward_parameter=args.reward_parameter, 
    plot=False, 
    full=args.full
)

print("Environments created successfully!")

# Create policy network
policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, 
               args.rnn_input_dim, args.rnn_hidden_dim, args.hidden_sizes_ac, 
               args.hidden_sizes_v, args.activation, args.output_activation, 
               args.output_activation_v, args.use_gpu, args.rnn_mode, args.drop_p)

# Create PPO trainer
ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, 
               args.steps_per_epoch, args.max_ep_len, args.gamma, args.lam, 
               args.clip_ratio, args.train_pi_iters, args.train_v_iters, 
               args.target_kl, args.render, args.render_freq, args.con_train, 
               args.seed, args.save_freq, args.save_figure, model_abs_path, 
               model_name, load_fname, args.use_gpu, args.reset_mode, 
               args.save_result, counter, test_env, args.lr_decay_epoch, 
               args.max_update_num, args.mpi, args.figure_save_path)

# save hyperparameters
if not os.path.exists(model_abs_path):
    os.makedirs(model_abs_path)

f = open(model_abs_path + model_name, 'wb')
pickle.dump(args, f)
f.close()

with open(model_abs_path + model_name + '.txt', 'w') as p:
    print(vars(args), file=p)
p.close()

# Copy base world file
shutil.copyfile(base_world_path, model_abs_path + model_name + '_base_world.yaml')

# Note: No temporary files are generated with runtime obstacle injection!
print("Using runtime obstacle injection - no temporary files generated!")

print("Starting training with runtime-injected static obstacles...")
print(f"Each episode will have {args.obstacle_count} randomly placed 1x1 static obstacles (injected at runtime)")
print("Training loop started!")

# run the training loop
try:
    ppo.training_loop()
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nTraining failed with error: {e}")
    raise
finally:
    # Clean up environments
    env.cleanup()
    test_env.cleanup()
    print("Training completed and resources cleaned up!")
