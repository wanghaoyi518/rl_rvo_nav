import os
import sys
import gym
import pickle
import shutil
import gym_env
import argparse
from torch import nn
from pathlib import Path
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac

# 路径设置
cur_path = Path(__file__).parent
world_abs_path = str(cur_path/'mode7_stage2_medium.yaml')

# 默认参数
counter = 0

parser = argparse.ArgumentParser(description='Mode 7 Stage 2 Curriculum Learning - Medium Complexity Random Polygons')

# 环境参数组
par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--world_path', default='mode7_stage2_medium.yaml')  # Stage 2配置
par_env.add_argument('--robot_number', type=int, default=6)
par_env.add_argument('--init_mode', default=7)  # Mode 7: random with distance constraint + random polygons
par_env.add_argument('--reset_mode', default=7)  # Mode 7 reset
par_env.add_argument('--mpi', default=False)

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=20)   
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), nargs='+')
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=True)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)

# 策略参数组
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

# 训练参数组
par_train = parser.add_argument_group('par train', 'train parameters') 
par_train.add_argument('--pi_lr', type=float, default=3e-6)  # Stage 2: 稍微降低学习率
par_train.add_argument('--vf_lr', type=float, default=4e-5)
par_train.add_argument('--train_epoch', type=int, default=400)  # Stage 2: 中等训练周期
par_train.add_argument('--steps_per_epoch', type=int, default=500)
par_train.add_argument('--max_ep_len', default=150)
par_train.add_argument('--gamma', default=0.99)
par_train.add_argument('--lam', default=0.97)
par_train.add_argument('--clip_ratio', default=0.2)
par_train.add_argument('--train_pi_iters', default=50)
par_train.add_argument('--train_v_iters', default=50)
par_train.add_argument('--target_kl',type=float, default=0.05)
par_train.add_argument('--render', default=True)
par_train.add_argument('--render_freq', default=50)
par_train.add_argument('--con_train', action='store_true')  # 启用连续训练
par_train.add_argument('--seed', default=7)
par_train.add_argument('--save_freq', default=50)
par_train.add_argument('--save_figure', default=False)
par_train.add_argument('--figure_save_path', default='figure')
par_train.add_argument('--save_path', default=str(cur_path / 'model_save') + '/')
par_train.add_argument('--save_name', default='r4_mode7_stage2_')  # Stage 2模型命名
par_train.add_argument('--load_path', default=str(cur_path / 'model_save')+ '/')
par_train.add_argument('--load_name', default='r4_mode7_stage1_4_0/r4_mode7_stage1_4_0_check_point_400.pt')  # 从stage1模型开始
par_train.add_argument('--save_result', type=bool, default=True)
par_train.add_argument('--lr_decay_epoch', type=int, default=1000)
par_train.add_argument('--max_update_num', type=int, default=10)

# 课程学习特定参数
par_curriculum = parser.add_argument_group('par curriculum', 'curriculum learning parameters')
par_curriculum.add_argument('--curriculum_stage', default=2, type=int, help='Current curriculum stage (1-3)')
par_curriculum.add_argument('--curriculum_auto', action='store_true', help='Enable automatic curriculum progression')
par_curriculum.add_argument('--performance_threshold', type=float, default=0.8, help='Performance threshold for stage advancement')

# 解析参数 (使用默认值，适合Stage 2)
args = parser.parse_args([
    '--train_epoch', '2000', 
    '--robot_number', '6', 
    '--load_name', 'r4_mode7_stage1_4_2/r4_mode7_stage1_4_2_check_point_350.pt', 
    '--con_train', 
    '--use_gpu'
])

# 决定模型路径和模型名称
model_path_check = args.save_path + args.save_name + str(args.robot_number) + '_{}'
model_name_check = args.save_name + str(args.robot_number) +  '_{}'
while os.path.isdir(model_path_check.format(counter)):
    counter+=1

model_abs_path = model_path_check.format(counter) + '/'
model_name = model_name_check.format(counter)

load_fname = args.load_path + args.load_name

print("🎯 Mode 7 Stage 2 课程学习训练")
print("=" * 60)
print(f"📁 加载Stage 1模型: {load_fname}")
print(f"📁 保存模型到: {model_abs_path}")
print(f"⚙️  使用配置: {args.world_path}")
print(f"🤖 机器人数量: {args.robot_number}")
print(f"📊 训练轮数: {args.train_epoch}")

# 创建环境
env = gym.make(args.env_name, 
               world_name=args.world_path, 
               robot_number=args.robot_number, 
               neighbors_region=args.neighbors_region, 
               neighbors_num=args.neighbors_num, 
               robot_init_mode=args.init_mode, 
               env_train=args.env_train, 
               random_bear=args.random_bear, 
               random_radius=args.random_radius, 
               reward_parameter=args.reward_parameter, 
               full=args.full)

test_env = gym.make(args.env_name, 
                    world_name=args.world_path, 
                    robot_number=args.robot_number, 
                    neighbors_region=args.neighbors_region, 
                    neighbors_num=args.neighbors_num, 
                    robot_init_mode=args.init_mode, 
                    env_train=False, 
                    random_bear=args.random_bear, 
                    random_radius=args.random_radius, 
                    reward_parameter=args.reward_parameter, 
                    plot=False, 
                    full=args.full)

# 创建策略网络
policy = rnn_ac(env.observation_space, env.action_space, args.state_dim, args.rnn_input_dim, args.rnn_hidden_dim, 
                args.hidden_sizes_ac, args.hidden_sizes_v, args.activation, args.output_activation, 
                args.output_activation_v, args.use_gpu, args.rnn_mode, args.drop_p)

# 创建PPO训练器
ppo = multi_ppo(env, policy, args.pi_lr, args.vf_lr, args.train_epoch, args.steps_per_epoch, args.max_ep_len, 
                args.gamma, args.lam, args.clip_ratio, args.train_pi_iters, args.train_v_iters, args.target_kl, 
                args.render, args.render_freq, args.con_train, args.seed, args.save_freq, args.save_figure, 
                model_abs_path, model_name, load_fname, args.use_gpu, args.reset_mode, args.save_result, 
                counter, test_env, args.lr_decay_epoch, args.max_update_num, args.mpi, args.figure_save_path)

# 保存超参数
if not os.path.exists(model_abs_path):
    os.makedirs(model_abs_path)

f = open(model_abs_path + model_name, 'wb')
pickle.dump(args, f)
f.close()

with open(model_abs_path+model_name+'.txt', 'w') as p:
    print(vars(args), file=p)
p.close()

# 保存配置文件副本
shutil.copyfile(str(cur_path/args.world_path), model_abs_path+model_name+'_world.yaml')

print(f"✅ 环境创建成功")
print(f"✅ 策略网络创建成功")
print(f"✅ PPO训练器创建成功")
print(f"📁 超参数已保存到: {model_abs_path + model_name}")
print(f"📁 配置文件已保存到: {model_abs_path + model_name + '_world.yaml'}")

# 运行训练循环
print("\n🚀 开始Stage 2训练...")
ppo.training_loop()

print(f"\n🎉 Stage 2训练完成！")
print(f"📁 最终模型保存在: {model_abs_path}")
print(f"📊 训练结果保存在: {model_abs_path + 'results.txt'}")
print(f"💡 下一步: 使用此模型作为Stage 3的输入")
