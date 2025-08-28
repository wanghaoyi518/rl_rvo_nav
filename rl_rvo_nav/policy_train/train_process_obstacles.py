import os
import sys
import gym
import pickle
import shutil
import gym_env
import argparse
import numpy as np
import torch
from torch import nn
from pathlib import Path
from rl_rvo_nav.policy_train.multi_ppo import multi_ppo
from rl_rvo_nav.policy.policy_rnn_ac import rnn_ac

# path set
cur_path = Path(__file__).parent
world_abs_path = str(cur_path/'train_world.yaml')

# Curriculum learning configuration
CURRICULUM_STAGES = [
    'curriculum_configs/stage0_basic.yaml',     # Stage 0: 无障碍物基础学习
    'curriculum_configs/stage1_simple.yaml',   # Stage 1: 简单障碍物
    'curriculum_configs/stage2_medium.yaml',   # Stage 2: 中等复杂度
    'curriculum_configs/stage3_complex.yaml',  # Stage 3: 复杂环境
    'curriculum_configs/stage4_advanced.yaml'  # Stage 4: 高级挑战
]

counter = 0

parser = argparse.ArgumentParser(description='DRL RVO with Curriculum Learning for Random Obstacles')

# Environment parameters
par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--world_path', default='train_world_random_obstacles.yaml')
par_env.add_argument('--robot_number', type=int, default=4)
par_env.add_argument('--init_mode', default=2)  # Default to random initialization
par_env.add_argument('--reset_mode', default=2)
par_env.add_argument('--mpi', default=False)

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=5)   
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), nargs='+')
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=True)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)

# Curriculum Learning parameters
par_curriculum = parser.add_argument_group('par curriculum', 'curriculum learning parameters')
par_curriculum.add_argument('--curriculum_enable', action='store_true', help='Enable curriculum learning')
par_curriculum.add_argument('--curriculum_auto', action='store_true', help='Automatic curriculum progression')
par_curriculum.add_argument('--curriculum_start_stage', type=int, default=0, help='Starting curriculum stage (0-3)')
par_curriculum.add_argument('--curriculum_success_threshold', type=float, default=0.6, help='Success rate to advance to next stage')  # 降低推进门槛
par_curriculum.add_argument('--curriculum_episodes_per_stage', type=int, default=200, help='Episodes per stage before auto-advance')  # 减少每阶段训练量
par_curriculum.add_argument('--curriculum_reward_threshold', type=float, default=15.0, help='Average reward threshold for advancement')
par_curriculum.add_argument('--curriculum_manual', action='store_true', help='Manual curriculum control (no auto advancement)')

# Random Obstacles parameters  
par_obstacles = parser.add_argument_group('par obstacles', 'random obstacles parameters')
par_obstacles.add_argument('--obs_curriculum_enable', action='store_true', help='Enable obstacle curriculum in gym env')
par_obstacles.add_argument('--obs_regenerate_freq', type=int, default=100, help='Regenerate obstacles every N episodes')
par_obstacles.add_argument('--obs_density_schedule', type=str, default='linear', choices=['linear', 'exponential', 'step'], 
                          help='Obstacle density increase schedule')

# Policy parameters
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

# Training parameters
par_train = parser.add_argument_group('par train', 'training parameters') 
par_train.add_argument('--train_epoch', type=int, default=2000)
par_train.add_argument('--episode_length', type=int, default=500)
par_train.add_argument('--buffer_size', type=int, default=2048)
par_train.add_argument('--batch_size', type=int, default=256)
par_train.add_argument('--lr_actor', type=float, default=3e-4)  # 恢复原始学习率
par_train.add_argument('--lr_critic', type=float, default=3e-4)
par_train.add_argument('--gamma', type=float, default=0.99)
par_train.add_argument('--lambd', type=float, default=0.95)
par_train.add_argument('--clip_param', type=float, default=0.2)
par_train.add_argument('--K_epochs', type=int, default=8)
par_train.add_argument('--use_gpu', action='store_true')

# Save and load parameters
par_save = parser.add_argument_group('par save', 'save parameters') 
par_save.add_argument('--save_path', default='./model_save/')
par_save.add_argument('--save_name', default='obs_curriculum_')
par_save.add_argument('--save_interval', type=int, default=50)
par_save.add_argument('--load_name', default='')
par_save.add_argument('--con_train', action='store_true')

# Test parameters
par_test = parser.add_argument_group('par test', 'test parameters')
par_test.add_argument('--test_interval', type=int, default=10)
par_test.add_argument('--test_episodes', type=int, default=50)

args = parser.parse_args()

class CurriculumManager:
    """Manages curriculum learning progression for obstacle-based training."""
    
    def __init__(self, args):
        self.args = args
        self.current_stage = args.curriculum_start_stage
        self.stages = CURRICULUM_STAGES
        self.episode_count = 0
        self.stage_episode_count = 0
        self.success_history = []
        self.reward_history = []
        self.advancement_log = []
        
        # Performance tracking
        self.stage_performance = {
            'success_rates': [],
            'average_rewards': [],
            'collision_rates': []
        }
        
        print(f"Curriculum Manager initialized:")
        print(f"  Starting stage: {self.current_stage}")
        print(f"  Total stages: {len(self.stages)}")
        print(f"  Auto progression: {args.curriculum_auto}")
        print(f"  Success threshold: {args.curriculum_success_threshold}")
    
    def get_current_config(self):
        """Get current stage configuration file path."""
        if self.current_stage >= len(self.stages):
            return self.stages[-1]  # Stay at final stage
        return self.stages[self.current_stage]
    
    def update_performance(self, episode_results):
        """Update performance tracking with episode results."""
        success_rate = episode_results.get('success_rate', 0.0)
        avg_reward = episode_results.get('avg_reward', 0.0)
        collision_rate = episode_results.get('collision_rate', 0.0)
        
        self.success_history.append(success_rate)
        self.reward_history.append(avg_reward)
        
        # Keep recent history for advancement decisions
        window_size = 50
        if len(self.success_history) > window_size:
            self.success_history = self.success_history[-window_size:]
            self.reward_history = self.reward_history[-window_size:]
    
    def should_advance(self):
        """Determine if curriculum should advance to next stage."""
        if not self.args.curriculum_auto or self.args.curriculum_manual:
            return False
            
        if self.current_stage >= len(self.stages) - 1:
            return False  # Already at final stage
        
        # Check episode count threshold
        if self.stage_episode_count < self.args.curriculum_episodes_per_stage:
            return False
        
        # Check performance thresholds
        if len(self.success_history) < 20:  # Need sufficient data
            return False
        
        recent_success = np.mean(self.success_history[-20:])
        recent_reward = np.mean(self.reward_history[-20:])
        
        success_met = recent_success >= self.args.curriculum_success_threshold
        reward_met = recent_reward >= self.args.curriculum_reward_threshold
        
        return success_met and reward_met
    
    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            old_stage = self.current_stage
            self.current_stage += 1
            self.stage_episode_count = 0
            self.success_history = []
            self.reward_history = []
            
            advancement_info = {
                'from_stage': old_stage,
                'to_stage': self.current_stage,
                'episode': self.episode_count,
                'config': self.get_current_config()
            }
            self.advancement_log.append(advancement_info)
            
            print(f"\n🎓 CURRICULUM ADVANCEMENT 🎓")
            print(f"  Advanced from Stage {old_stage} to Stage {self.current_stage}")
            print(f"  New config: {self.get_current_config()}")
            print(f"  Total episodes: {self.episode_count}")
            
            return True
        return False
    
    def get_status(self):
        """Get current curriculum status."""
        return {
            'current_stage': self.current_stage,
            'total_stages': len(self.stages),
            'episode_count': self.episode_count,
            'stage_episode_count': self.stage_episode_count,
            'config_file': self.get_current_config(),
            'recent_success_rate': np.mean(self.success_history[-10:]) if self.success_history else 0.0,
            'recent_avg_reward': np.mean(self.reward_history[-10:]) if self.reward_history else 0.0
        }

def create_environment(config_path, args, env_train=True):
    """Create gym environment with specified configuration."""
    env = gym.make(
        args.env_name, 
        world_name=config_path,
        robot_number=args.robot_number, 
        neighbors_region=args.neighbors_region, 
        neighbors_num=args.neighbors_num, 
        robot_init_mode=args.init_mode, 
        env_train=env_train, 
        random_bear=args.random_bear, 
        random_radius=args.random_radius, 
        reward_parameter=args.reward_parameter, 
        full=args.full,
        obs_curriculum_enable=args.obs_curriculum_enable,
        plot=not env_train  # No plot for training env, plot for test env
    )
    return env

def evaluate_performance(test_env, policy, args, num_episodes=50):
    """Evaluate policy performance on test environment."""
    success_count = 0
    collision_count = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs_list = test_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < args.episode_length:
            actions = []
            for obs in obs_list:
                action = policy.act(torch.as_tensor(obs, dtype=torch.float32))
                actions.append(action)
            
            obs_list, reward_list, done_list, info_list = test_env.step_ir(actions)
            episode_reward += sum(reward_list)
            episode_length += 1
            
            # Check if all robots reached goal
            if all(done_list):
                success_count += 1
                done = True
            
            # Check for collisions
            if test_env.ir_gym.collision_check():
                collision_count += 1
                done = True
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'std_reward': np.std(total_rewards)
    }

def main():
    print("Starting Curriculum Learning Training with Random Obstacles")
    print("=" * 60)
    
    # Initialize curriculum manager
    curriculum_manager = None
    if args.curriculum_enable:
        curriculum_manager = CurriculumManager(args)
        config_path = curriculum_manager.get_current_config()
        print(f"Using curriculum learning with config: {config_path}")
    else:
        config_path = args.world_path
        print(f"Using fixed config: {config_path}")
    
    # Create environments
    print("Creating training and test environments...")
    train_env = create_environment(config_path, args, env_train=True)
    test_env = create_environment(config_path, args, env_train=False)
    
    # Initialize policy
    print("Initializing policy...")
    policy = rnn_ac(
        observation_space=train_env.observation_space, 
        action_space=train_env.action_space, 
        state_dim=args.state_dim, 
        rnn_input_dim=args.rnn_input_dim, 
        rnn_hidden_dim=args.rnn_hidden_dim, 
        hidden_sizes_ac=args.hidden_sizes_ac, 
        drop_p=args.drop_p
    )
    
    # Create model save directory
    counter = 0  # Initialize counter
    model_path_check = args.save_path + args.save_name + str(args.robot_number) + '_{}'
    model_name_check = args.save_name + str(args.robot_number) + '_{}'
    while os.path.isdir(model_path_check.format(counter)):
        counter += 1
    
    model_path = model_path_check.format(counter)
    model_name = model_name_check.format(counter)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Model will be saved to: {model_path}")
    
    # Initialize PPO trainer
    ppo_trainer = multi_ppo(
        env=train_env, 
        ac_policy=policy,
        pi_lr=args.lr_actor,
        vf_lr=args.lr_critic,
        train_epoch=args.train_epoch,
        steps_per_epoch=args.episode_length,
        max_ep_len=args.buffer_size,
        gamma=args.gamma,
        lam=args.lambd,
        clip_ratio=args.clip_param,
        train_pi_iters=args.K_epochs,
        save_path=model_path,
        save_name=model_name,
        save_freq=args.save_interval,
        load_fname=args.load_name,
        con_train=args.con_train,
        use_gpu=args.use_gpu,
        reset_mode=args.reset_mode,
        test_env=test_env,
        seed=42  # Set explicit seed
    )
    
    # Start training using the built-in training loop with modification for curriculum learning
    print("\nStarting training...")
    print("=" * 60)
    
    # Use the modified training approach
    # Call training_loop directly since it handles data collection and update properly
    ppo_trainer.training_loop()
    
    # Final evaluation and save
    print("\n" + "=" * 60)
    print("Training completed!")
    
    if curriculum_manager:
        final_status = curriculum_manager.get_status()
        print(f"Final Curriculum Stage: {final_status['current_stage']}")
        print(f"Total Episodes: {final_status['episode_count']}")
        
        # Save curriculum log
        curriculum_log_path = os.path.join(model_path, 'curriculum_log.pkl')
        with open(curriculum_log_path, 'wb') as f:
            pickle.dump(curriculum_manager.advancement_log, f)
        print(f"Curriculum log saved to: {curriculum_log_path}")
    
    # Final evaluation
    print("\nFinal performance evaluation...")
    final_eval = evaluate_performance(test_env, policy, args, 100)
    print(f"Final Success Rate: {final_eval['success_rate']:.3f}")
    print(f"Final Collision Rate: {final_eval['collision_rate']:.3f}")
    print(f"Final Avg Reward: {final_eval['avg_reward']:.2f}")
    
    # Save final model
    ppo_trainer.save_model("final_model")
    print(f"Final model saved to: {model_path}")

if __name__ == '__main__':
    main()
