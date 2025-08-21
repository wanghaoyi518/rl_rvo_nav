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
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac

# path set
cur_path = Path(__file__).parent
world_abs_path = str(cur_path/'train_world.yaml')

# default
counter = 0

parser = argparse.ArgumentParser(description='drl rvo parameters')

par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--world_path', default='train_world.yaml')
par_env.add_argument('--robot_number', type=int, default=2)  # Changed from 4 to 2
par_env.add_argument('--init_mode', default=2)
par_env.add_argument('--reset_mode', default=2)
par_env.add_argument('--mpi', default=False)

# Obstacle training parameters (only difference from original)
par_env.add_argument('--use_obstacles', action='store_true', help='Enable obstacle training')
par_env.add_argument('--obstacle_config', default='train_world_with_obstacles.yaml', help='Obstacle configuration file')
par_env.add_argument('--dynamic_obstacles', action='store_true', help='Use dynamically generated obstacles')
par_env.add_argument('--obstacle_count', type=int, default=2, help='Number of obstacles to generate')
par_env.add_argument('--min_obstacle_distance', type=float, default=5.0, help='Minimum distance between obstacles')

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=10)  # Changed from 5 to 10
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
par_train.add_argument('--train_epoch', type=int, default=250)
par_train.add_argument('--steps_per_epoch', type=int, default=500)
par_train.add_argument('--max_ep_len', default=150)
par_train.add_argument('--gamma', default=0.99)
par_train.add_argument('--lam', default=0.97)
par_train.add_argument('--clip_ratio', default=0.2)
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)
par_train.add_argument('--target_kl',type=float, default=0.05)
par_train.add_argument('--render', default=False)  # Changed from True to False
par_train.add_argument('--render_freq', default=50)
par_train.add_argument('--con_train', action='store_true')
par_train.add_argument('--seed', default=7)
par_train.add_argument('--save_freq', default=50)
par_train.add_argument('--save_figure', default=False)
par_train.add_argument('--figure_save_path', default='figure')
par_train.add_argument('--save_path', default=str(cur_path / 'model_save') + '/')
par_train.add_argument('--save_name', default= 'r')
par_train.add_argument('--load_path', default=str(cur_path / 'model_save')+ '/')
par_train.add_argument('--load_name', default='r4_0/r4_0_check_point_250.pt')
par_train.add_argument('--save_result', type=bool, default=True)
par_train.add_argument('--lr_decay_epoch', type=int, default=1000)
par_train.add_argument('--max_update_num', type=int, default=10)

# Use same hardcoded arguments as train_process_s1.py
args = parser.parse_args(['--train_epoch', '250', '--use_gpu'])

def generate_dynamic_obstacles(obstacle_count=2, min_distance=5.0, world_size=10, obstacle_size=2):
    """Generate dynamic obstacles with distance constraint."""
    obstacles = []
    obs_polygons = []
    obs_lines = []
    
    for i in range(obstacle_count):
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            # Generate obstacle position
            x = random.uniform(0, world_size - obstacle_size)
            y = random.uniform(0, world_size - obstacle_size)
            
            # Calculate obstacle center
            center_x = x + obstacle_size / 2
            center_y = y + obstacle_size / 2
            
            # Check distance constraint with existing obstacles
            valid_position = True
            for existing_obs in obstacles:
                existing_center_x = existing_obs[0] + obstacle_size / 2
                existing_center_y = existing_obs[1] + obstacle_size / 2
                
                distance = np.sqrt((center_x - existing_center_x)**2 + (center_y - existing_center_y)**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
            if valid_position:
                # Define obstacle corners
                bottom_left = (x, y)
                bottom_right = (x + obstacle_size, y)
                top_right = (x + obstacle_size, y + obstacle_size)
                top_left = (x, y + obstacle_size)
                
                obstacles.append((x, y))
                
                # Add to polygon list (clockwise order)
                obs_polygons.append([
                    [bottom_left[0], bottom_left[1]],
                    [bottom_right[0], bottom_right[1]],
                    [top_right[0], top_right[1]],
                    [top_left[0], top_left[1]]
                ])
                
                # Add line segments for obs_lines
                obs_lines.extend([
                    [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]],
                    [bottom_right[0], bottom_right[1], top_right[0], top_right[1]],
                    [top_right[0], top_right[1], top_left[0], top_left[1]],
                    [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]
                ])
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: Could not find valid position for obstacle {i+1}")
            # Fallback: place obstacle at a random position
            x = random.uniform(0, world_size - obstacle_size)
            y = random.uniform(0, world_size - obstacle_size)
            
            bottom_left = (x, y)
            bottom_right = (x + obstacle_size, y)
            top_right = (x + obstacle_size, y + obstacle_size)
            top_left = (x, y + obstacle_size)
            
            obstacles.append((x, y))
            
            obs_polygons.append([
                [bottom_left[0], bottom_left[1]],
                [bottom_right[0], bottom_right[1]],
                [top_right[0], top_right[1]],
                [top_left[0], top_left[1]]
            ])
            
            obs_lines.extend([
                [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]],
                [bottom_right[0], bottom_right[1], top_right[0], top_right[1]],
                [top_right[0], top_right[1], top_left[0], top_left[1]],
                [top_left[0], top_left[1], bottom_left[0], bottom_left[1]]
            ])
    
    return obs_polygons, obs_lines

def create_obstacle_world_config(obstacle_count=2, min_distance=5.0):
    """Create a world configuration with obstacles."""
    obs_polygons, obs_lines = generate_dynamic_obstacles(obstacle_count, min_distance)
    
    world_config = {
        'world': {
            'world_height': 10,
            'world_width': 10,
            'step_time': 0.1
        },
        'robots': {
            'robot_mode': 'diff',
            'radius_list': [0.2, 0.2],
            'vel_max': [1.5, 1.5],
            'radius_exp': 0.1,
            'interval': 1,
            'square': [0, 0, 10, 10],
            'circular': [5, 5, 4]
        },
        'obs_polygons': {
            'number': len(obs_polygons),
            'vertexes_list': obs_polygons
        },
        'obs_lines': {
            'number': len(obs_lines),
            'obs_line_states': obs_lines
        }
    }
    
    return world_config

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# Choose world configuration based on obstacle settings
if args.use_obstacles:
    if args.dynamic_obstacles:
        print(f"Using dynamic obstacles: {args.obstacle_count} obstacles with min distance {args.min_obstacle_distance}")
        world_config = create_obstacle_world_config(args.obstacle_count, args.min_obstacle_distance)
        # Save dynamic world config to temporary file
        temp_world_path = str(cur_path / f'temp_world_{args.seed}.yaml')
        with open(temp_world_path, 'w') as f:
            yaml.dump(world_config, f, default_flow_style=False, sort_keys=False)
        world_path = temp_world_path
    else:
        print(f"Using static obstacles from {args.obstacle_config}")
        world_path = str(cur_path / args.obstacle_config)
else:
    print("Using original world without obstacles")
    world_path = str(cur_path / args.world_path)

# decide the model path and model name 
model_path_check = args.save_path + args.save_name + str(args.robot_number) + '_{}'
model_name_check = args.save_name + str(args.robot_number) +  '_{}'
while os.path.isdir(model_path_check.format(counter)):
    counter+=1

model_abs_path = model_path_check.format(counter) + '/'
model_name = model_name_check.format(counter)

load_fname = args.load_path + args.load_name

# Create environments (same as original train_process_s1.py)
env = gym.make(args.env_name, world_name=world_path, robot_number=args.robot_number, 
               neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, 
               robot_init_mode=args.init_mode, env_train=args.env_train, 
               random_bear=args.random_bear, random_radius=args.random_radius, 
               reward_parameter=args.reward_parameter, full=args.full)

test_env = gym.make(args.env_name, world_name=world_path, robot_number=args.robot_number, 
                    neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, 
                    robot_init_mode=args.init_mode, env_train=False, 
                    random_bear=args.random_bear, random_radius=args.random_radius, 
                    reward_parameter=args.reward_parameter, plot=False, full=args.full)

# Create policy network (same as original)
policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, args.rnn_input_dim, 
                args.rnn_hidden_dim, args.hidden_sizes_ac, args.hidden_sizes_v, 
                args.activation, args.output_activation, args.output_activation_v, 
                args.use_gpu, args.rnn_mode, args.drop_p)

# Create PPO trainer (same as original)
ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, args.steps_per_epoch, 
                args.max_ep_len, args.gamma, args.lam, args.clip_ratio, args.train_pi_iters, 
                args.train_v_iters, args.target_kl, args.render, args.render_freq, 
                args.con_train, args.seed, args.save_freq, args.save_figure, model_abs_path, 
                model_name, load_fname, args.use_gpu, args.reset_mode, args.save_result, 
                counter, test_env, args.lr_decay_epoch, args.max_update_num, args.mpi, 
                args.figure_save_path)

# save hyperparameters (same as original)
if not os.path.exists(model_abs_path):
    os.makedirs(model_abs_path)

f = open(model_abs_path + model_name, 'wb')
pickle.dump(args, f)
f.close()

with open(model_abs_path+model_name+'.txt', 'w') as p:
    print(vars(args), file=p)
p.close()

# Copy world file (same as original, but use the actual world path)
shutil.copyfile(world_path, model_abs_path+model_name+'_world.yaml')

# run the training loop (same as original)
ppo.training_loop()

# Clean up temporary world file if created
if args.use_obstacles and args.dynamic_obstacles and os.path.exists(temp_world_path):
    os.remove(temp_world_path)
    print(f"Cleaned up temporary world file: {temp_world_path}")

print("Training completed!")
