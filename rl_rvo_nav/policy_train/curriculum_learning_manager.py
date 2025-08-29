#!/usr/bin/env python3
"""
Mode 7 课程学习管理器
自动化执行三个阶段的课程学习流程
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

class CurriculumLearningManager:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.policy_train_dir = self.base_dir / 'rl_rvo_nav/rl_rvo_nav/policy_train'
        self.model_save_dir = self.policy_train_dir / 'model_save'
        
        # 课程学习配置
        self.stages = {
            1: {
                'script': 'train_process_obs_s1.py',
                'config': 'mode7_stage1_simple.yaml',
                'base_model': 'r4_1/r4_1_check_point_150.pt',
                'epochs': 300,
                'description': '简单随机多边形障碍物 (4个机器人)'
            },
            2: {
                'script': 'train_process_obs_s2.py', 
                'config': 'mode7_stage2_medium.yaml',
                'base_model': None,  # 将在运行时确定
                'epochs': 400,
                'description': '中等复杂度随机多边形障碍物 (6个机器人)'
            },
            3: {
                'script': 'train_process_obs_s3.py',
                'config': 'mode7_stage3_complex.yaml', 
                'base_model': None,  # 将在运行时确定
                'epochs': 500,
                'description': '复杂随机多边形障碍物 (8个机器人)'
            }
        }
        
        # 训练结果记录
        self.training_results = {}
        
    def check_prerequisites(self):
        """检查课程学习前置条件"""
        print("🔍 检查课程学习前置条件...")
        
        # 检查基础模型是否存在
        base_model_path = self.model_save_dir / self.stages[1]['base_model']
        if not base_model_path.exists():
            print(f"❌ 基础模型不存在: {base_model_path}")
            print("请确保已训练基础模型 r4_1_check_point_150.pt")
            return False
            
        # 检查配置文件是否存在
        for stage_id, stage_info in self.stages.items():
            config_path = self.policy_train_dir / stage_info['config']
            if not config_path.exists():
                print(f"❌ Stage {stage_id} 配置文件不存在: {config_path}")
                return False
                
        # 检查训练脚本是否存在
        for stage_id, stage_info in self.stages.items():
            script_path = self.policy_train_dir / stage_info['script']
            if not script_path.exists():
                print(f"❌ Stage {stage_id} 训练脚本不存在: {script_path}")
                return False
                
        print("✅ 所有前置条件检查通过")
        return True
        
    def find_latest_model(self, stage_prefix):
        """查找指定阶段的最新模型"""
        if not self.model_save_dir.exists():
            return None
            
        # 查找匹配的模型目录
        matching_dirs = []
        for item in self.model_save_dir.iterdir():
            if item.is_dir() and item.name.startswith(stage_prefix):
                matching_dirs.append(item)
                
        if not matching_dirs:
            return None
            
        # 按修改时间排序，返回最新的
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        
        # 查找最新的检查点文件
        checkpoint_files = list(latest_dir.glob('*_check_point_*.pt'))
        if not checkpoint_files:
            return None
            
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return latest_checkpoint.relative_to(self.model_save_dir)
        
    def update_stage_model_paths(self):
        """更新各阶段的模型路径"""
        print("🔄 更新各阶段模型路径...")
        
        # Stage 2 使用 Stage 1 的最新模型 (4个机器人)
        stage1_model = self.find_latest_model('r4_mode7_stage1_4_')
        if stage1_model:
            self.stages[2]['base_model'] = str(stage1_model)
            print(f"📁 Stage 2 将使用: {stage1_model}")
        else:
            print("⚠️  未找到 Stage 1 模型，将使用默认路径")
            
        # Stage 3 使用 Stage 2 的最新模型 (6个机器人)
        stage2_model = self.find_latest_model('r4_mode7_stage2_6_')
        if stage2_model:
            self.stages[3]['base_model'] = str(stage2_model)
            print(f"📁 Stage 3 将使用: {stage2_model}")
        else:
            print("⚠️  未找到 Stage 2 模型，将使用默认路径")
            
    def run_stage_training(self, stage_id):
        """运行指定阶段的训练"""
        stage_info = self.stages[stage_id]
        
        print(f"\n🎯 开始 Stage {stage_id} 训练")
        print("=" * 60)
        print(f"📝 描述: {stage_info['description']}")
        print(f"📁 脚本: {stage_info['script']}")
        print(f"⚙️  配置: {stage_info['config']}")
        print(f"📊 训练轮数: {stage_info['epochs']}")
        if stage_info['base_model']:
            print(f"📁 基础模型: {stage_info['base_model']}")
            
        # 构建训练命令
        script_path = self.policy_train_dir / stage_info['script']
        cmd = [
            sys.executable, str(script_path),
            '--train_epoch', str(stage_info['epochs']),
            '--robot_number', '4',
            '--con_train',
            '--use_gpu'
        ]
        
        if stage_info['base_model']:
            cmd.extend(['--load_name', stage_info['base_model']])
            
        print(f"🚀 执行命令: {' '.join(cmd)}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 执行训练
            result = subprocess.run(
                cmd,
                cwd=self.policy_train_dir,
                capture_output=True,
                text=True,
                timeout=3600 * 6  # 6小时超时
            )
            
            # 记录结束时间
            end_time = time.time()
            training_duration = end_time - start_time
            
            # 保存训练结果
            self.training_results[stage_id] = {
                'success': result.returncode == 0,
                'duration': training_duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if result.returncode == 0:
                print(f"✅ Stage {stage_id} 训练成功完成")
                print(f"⏱️  训练耗时: {training_duration/3600:.2f} 小时")
            else:
                print(f"❌ Stage {stage_id} 训练失败")
                print(f"错误信息: {result.stderr}")
                
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Stage {stage_id} 训练超时")
            self.training_results[stage_id] = {
                'success': False,
                'duration': 3600 * 6,
                'stdout': '',
                'stderr': 'Training timeout',
                'return_code': -1
            }
            return False
            
        except Exception as e:
            print(f"❌ Stage {stage_id} 训练异常: {e}")
            self.training_results[stage_id] = {
                'success': False,
                'duration': time.time() - start_time,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1
            }
            return False
            
    def run_full_curriculum(self, start_stage=1, end_stage=3):
        """运行完整的课程学习流程"""
        print("🎓 Mode 7 课程学习管理器")
        print("=" * 60)
        print(f"📊 训练范围: Stage {start_stage} - Stage {end_stage}")
        
        # 检查前置条件
        if not self.check_prerequisites():
            return False
            
        # 更新模型路径
        self.update_stage_model_paths()
        
        # 执行各阶段训练
        success_count = 0
        for stage_id in range(start_stage, end_stage + 1):
            if stage_id not in self.stages:
                print(f"⚠️  Stage {stage_id} 不存在，跳过")
                continue
                
            print(f"\n{'='*20} Stage {stage_id} {'='*20}")
            
            # 运行训练
            success = self.run_stage_training(stage_id)
            
            if success:
                success_count += 1
                print(f"✅ Stage {stage_id} 完成")
                
                # 更新下一阶段的模型路径
                if stage_id < end_stage:
                    self.update_stage_model_paths()
            else:
                print(f"❌ Stage {stage_id} 失败，停止课程学习")
                break
                
        # 生成训练报告
        self.generate_training_report()
        
        print(f"\n🎉 课程学习完成！")
        print(f"📊 成功完成: {success_count}/{end_stage - start_stage + 1} 个阶段")
        
        return success_count == (end_stage - start_stage + 1)
        
    def generate_training_report(self):
        """生成训练报告"""
        report_path = self.policy_train_dir / 'curriculum_learning_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Mode 7 课程学习训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            for stage_id, result in self.training_results.items():
                f.write(f"Stage {stage_id}: {self.stages[stage_id]['description']}\n")
                f.write(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}\n")
                f.write(f"  耗时: {result['duration']/3600:.2f} 小时\n")
                f.write(f"  返回码: {result['return_code']}\n")
                
                if result['stderr']:
                    f.write(f"  错误信息: {result['stderr']}\n")
                    
                f.write("\n")
                
        print(f"📄 训练报告已保存到: {report_path}")
        
    def list_available_models(self):
        """列出可用的模型"""
        print("📁 可用模型列表:")
        print("=" * 40)
        
        if not self.model_save_dir.exists():
            print("❌ 模型保存目录不存在")
            return
            
        for item in sorted(self.model_save_dir.iterdir()):
            if item.is_dir():
                # 查找检查点文件
                checkpoints = list(item.glob('*_check_point_*.pt'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    mod_time = latest_checkpoint.stat().st_mtime
                    print(f"📁 {item.name}/")
                    print(f"  最新检查点: {latest_checkpoint.name}")
                    print(f"  修改时间: {time.ctime(mod_time)}")
                    print()

def main():
    parser = argparse.ArgumentParser(description='Mode 7 课程学习管理器')
    parser.add_argument('--base_dir', default='/home/haoyiwang/Desktop/RL_RVO', 
                       help='项目基础目录')
    parser.add_argument('--start_stage', type=int, default=1, 
                       help='开始阶段 (1-3)')
    parser.add_argument('--end_stage', type=int, default=3, 
                       help='结束阶段 (1-3)')
    parser.add_argument('--list_models', action='store_true',
                       help='列出可用模型')
    parser.add_argument('--check_only', action='store_true',
                       help='仅检查前置条件，不开始训练')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = CurriculumLearningManager(args.base_dir)
    
    if args.list_models:
        manager.list_available_models()
        return
        
    if args.check_only:
        manager.check_prerequisites()
        return
        
    # 运行课程学习
    success = manager.run_full_curriculum(args.start_stage, args.end_stage)
    
    if success:
        print("\n🎉 所有阶段训练成功完成！")
        print("📁 最终模型可用于实际应用")
    else:
        print("\n⚠️  部分阶段训练失败，请检查错误信息")

if __name__ == "__main__":
    main()
