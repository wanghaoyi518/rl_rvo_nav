#!/usr/bin/env python3
"""
Policy testing script with deadlock detection and random static obstacles
"""

import gym
import gym_env
from pathlib import Path
import pickle
import sys
from rl_rvo_nav.policy_test.post_train_with_deadlock import post_train
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

try:
    import yaml
except ImportError:
    print("警告：yaml模块未安装，将使用命令行参数作为默认配置")
    print("可以通过 pip install pyyaml 安装yaml模块")
    yaml = None

def print_usage_info():
    """打印使用信息"""
    print("=" * 60)
    print("多智能体导航策略测试 - 集成死锁检测功能 + 随机静态障碍物")
    print("=" * 60)
    print("基本用法:")
    print("  python policy_test_with_deadlock_static_obstacles.py --robot_number 6 --num_episodes 50")
    print("")
    print("静态障碍物参数:")
    print("  --obstacle_count               障碍物数量 (默认4)")
    print("  --obstacle_size                障碍物尺寸 (默认0.8)")
    print("  --min_obstacle_distance        障碍物最小距离 (默认1.5)")
    print("  --base_world_config            基础世界配置文件")
    print("")
    print("死锁检测参数:")
    print("  --enable_deadlock_detection    启用死锁检测 (默认启用)")
    print("  --deadlock_distance_threshold  距离阈值 (默认3.0)")
    print("  --deadlock_speed_threshold     速度阈值 (默认0.1)")
    print("  --min_agents_for_deadlock     最小agent数 (默认2)")
    print("  --deadlock_config             配置文件路径")
    print("")
    print("其他参数与原始policy_test.py相同")
    print("=" * 60)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

parser = argparse.ArgumentParser(description='policy test with deadlock detection and random static obstacles')
parser.add_argument('--policy_type', default='drl')
parser.add_argument('--model_path', default='policy_train/model_save')
parser.add_argument('--model_name', default='r6_static_optimized6_static_0/r6_static_optimized6_static_0_check_point_500.pt')
parser.add_argument('--arg_name', default='r6_static_optimized6_static_0/r6_static_optimized6_static_0')
parser.add_argument('--base_world_config', default='policy_test_world.yaml',
                   help='Base world configuration (without obstacles)')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--robot_number', type=int, default='6')
parser.add_argument('--num_episodes', type=int, default='50')
parser.add_argument('--dis_mode', type=int, default='1')  # 1 random - 随机生成起始点和目标点

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

# 分布式死锁检测相关参数
parser.add_argument('--deadlock_config', default='deadlock_config.yaml', help='deadlock detection config file')
parser.add_argument('--enable_deadlock_detection', action='store_true', default=True, help='enable deadlock detection')
parser.add_argument('--deadlock_speed_buffer_size', type=int, default=20, help='speed buffer size for deadlock detection')
parser.add_argument('--deadlock_speed_threshold', type=float, default=0.05, help='speed threshold for deadlock detection')
parser.add_argument('--deadlock_neighbor_speed_threshold', type=float, default=0.05, help='neighbor speed threshold for deadlock detection')
parser.add_argument('--min_agents_for_deadlock', type=int, default=1, help='minimum number of neighbors for deadlock detection')
parser.add_argument('--deadlock_distance_threshold', type=float, default=2.0, help='sight radius for deadlock detection')
parser.add_argument('--deadlock_detection_interval', type=int, default=10, help='detection interval for deadlock detection')

policy_args = parser.parse_args()

# 打印使用信息
print_usage_info()

# 设置随机种子
set_seed(policy_args.seed)

cur_path = Path(__file__).parent.parent 

# model_base_path = str(cur_path) + '/' + policy_args.model_path
model_base_path = dirname(dirname(abspath(__file__))) + '/' + policy_args.model_path
args_path = model_base_path + '/' + policy_args.arg_name

print("=== Deadlock Detection + Random Static Obstacle Policy Testing ===")
print(f"Model path: {args_path}")
print(f"Robot number: {policy_args.robot_number}")
print(f"Episodes: {policy_args.num_episodes}")
print(f"Obstacle count per episode: {policy_args.obstacle_count}")
print(f"Obstacle size: {policy_args.obstacle_size}x{policy_args.obstacle_size}")
print(f"Minimum obstacle distance: {policy_args.min_obstacle_distance}")
print(f"Base world config: {policy_args.base_world_config}")
print("=" * 70)

# args from train
try:
    r = open(args_path, 'rb')
    args = pickle.load(r)
    r.close()
    print("Training arguments loaded successfully")
except FileNotFoundError:
    print(f"错误：找不到参数文件 {args_path}")
    print("请确保模型路径和参数文件名正确")
    sys.exit(1)
except Exception as e:
    print(f"错误：加载参数文件失败 {e}")
    sys.exit(1) 

if policy_args.policy_type == 'drl':
    fname_model = model_base_path + '/' + policy_args.model_name 
    policy_name = 'drl_rvo_deadlock_static'
    
    # 检查模型文件是否存在
    if not Path(fname_model).exists():
        print(f"错误：找不到模型文件 {fname_model}")
        print("请确保模型路径和文件名正确")
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

# 对于随机生成的环境，waypoint序列为空，直接从起始点导航到目标点
waypoint_sequences = {}

policy_name = policy_name + '_' + str(policy_args.robot_number) + '_dis' + str(policy_args.dis_mode)

# 加载死锁检测配置
deadlock_config = {}
if policy_args.enable_deadlock_detection:
    # 尝试多个可能的配置文件路径
    possible_config_paths = [
        cur_path / policy_args.deadlock_config,
        Path(policy_args.deadlock_config),
        Path(__file__).parent.parent / policy_args.deadlock_config,
        Path(__file__).parent.parent.parent / policy_args.deadlock_config,  # 添加项目根目录
        Path.cwd() / policy_args.deadlock_config  # 添加当前工作目录
    ]
    
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path and yaml is not None:
        try:
            with open(config_path, 'r') as f:
                deadlock_config = yaml.safe_load(f)
            print(f"已加载死锁检测配置文件: {config_path}")
        except Exception as e:
            print(f"加载死锁检测配置文件失败: {e}")
            print("使用命令行参数作为默认配置")
    elif config_path and yaml is None:
        print("yaml模块未安装，无法加载配置文件")
        print("使用命令行参数作为默认配置")
    else:
        print(f"死锁检测配置文件不存在，尝试路径: {[str(p) for p in possible_config_paths]}")
        print("使用命令行参数作为默认配置")
    
    # 合并命令行参数和配置文件
    deadlock_kwargs = {
        'deadlock_speed_buffer_size': policy_args.deadlock_speed_buffer_size,
        'deadlock_speed_threshold': policy_args.deadlock_speed_threshold,
        'deadlock_neighbor_speed_threshold': policy_args.deadlock_neighbor_speed_threshold,
        'min_agents_for_deadlock': policy_args.min_agents_for_deadlock,
        'deadlock_distance_threshold': policy_args.deadlock_distance_threshold,
        'deadlock_detection_interval': policy_args.deadlock_detection_interval,
    }
    
    # 如果配置文件存在，用配置文件覆盖命令行参数
    if deadlock_config:
        deadlock_kwargs.update({
            'deadlock_speed_buffer_size': deadlock_config.get('speed_buffer_size', deadlock_kwargs['deadlock_speed_buffer_size']),
            'deadlock_speed_threshold': deadlock_config.get('small_speed_threshold', deadlock_kwargs['deadlock_speed_threshold']),
            'deadlock_neighbor_speed_threshold': deadlock_config.get('neighbor_speed_threshold', deadlock_kwargs['deadlock_neighbor_speed_threshold']),
            'min_agents_for_deadlock': deadlock_config.get('min_neighbors_for_deadlock', deadlock_kwargs['min_agents_for_deadlock']),
            'deadlock_distance_threshold': deadlock_config.get('sight_radius', deadlock_kwargs['deadlock_distance_threshold']),
            'deadlock_detection_interval': deadlock_config.get('detection_interval', deadlock_kwargs['deadlock_detection_interval']),
        })
    
    print("死锁检测参数:")
    for key, value in deadlock_kwargs.items():
        print(f"  {key}: {value}")
else:
    deadlock_kwargs = {}
    print("死锁检测已禁用")

# 创建post_train实例，集成死锁检测
print("Initializing policy tester with deadlock detection...")
try:
    pt = post_train(
        env, 
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
        figure_format='eps',
        waypoint_sequences=waypoint_sequences,
        waypoint_goal_threshold=0.2,
        **deadlock_kwargs  # 传入死锁检测参数
    )
    print("Policy tester with deadlock detection initialized successfully!")
except Exception as e:
    print(f"Error initializing policy tester: {e}")
    sys.exit(1)

# 运行策略测试
print(f"\nStarting policy test with {policy_args.num_episodes} episodes...")
print("Each episode will have runtime-injected random static obstacle layouts!")
print("Deadlock detection is enabled!")

try:
    pt.policy_test(
        policy_args.policy_type, 
        fname_model, 
        policy_name, 
        result_path=str(cur_path), 
        result_name='/result_with_deadlock_static_obstacles.txt', 
        figure_save_path=cur_path / 'figure_with_deadlock_static', 
        ani_save_path=cur_path / 'gif_with_deadlock_static', 
        policy_dict=policy_args.policy_dict, 
        once=policy_args.once
    )
    
    print("\nPolicy testing completed successfully!")
    print(f"Results saved to: {cur_path}/result_with_deadlock_static_obstacles.txt")
    
except Exception as e:
    print(f"Error during policy testing: {e}")
    raise
finally:
    # Clean up environment
    if 'env' in locals():
        env.cleanup()
    print("Environment resources cleaned up!")

print("策略测试完成，包含死锁检测功能和随机静态障碍物！")
print(f"结果文件保存在: {cur_path}")
print(f"死锁日志文件: {cur_path}/deadlock_log_YYYYMMDD_HHMMSS.txt")
print(f"测试结果文件: {cur_path}/result_with_deadlock_static_obstacles.txt")
print(f"碰撞日志文件: {cur_path}/collision_neighbor_YYYYMMDD_HHMMSS.txt")
print(f"VO标志日志文件: {cur_path}/vo_flag_log_YYYYMMDD_HHMMSS.txt")
print("注意：日志文件名包含时间戳，格式为 YYYYMMDD_HHMMSS")
print("=" * 70)
print("Deadlock detection + random static obstacle policy testing finished!")
