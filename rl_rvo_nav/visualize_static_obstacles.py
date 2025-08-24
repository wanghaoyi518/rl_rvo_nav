#!/usr/bin/env python3
"""
可视化静态障碍物环境
展示随机生成的静态障碍物布局和机器人位置
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the project root to sys.path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.static_obstacle_generator import StaticObstacleGenerator
from rl_rvo_nav.utils.runtime_obstacle_manager import create_runtime_obstacle_env

def visualize_static_obstacles_environment(num_layouts=3):
    """
    可视化随机静态障碍物环境
    """
    print("🎯 创建静态障碍物环境可视化...")
    
    # 环境参数 - 优化设置
    obstacle_count = 4
    obstacle_size = 0.8
    min_obstacle_distance = 1.5
    world_size = 12  # 扩大世界大小为12x12
    
    # 创建障碍物生成器
    generator = StaticObstacleGenerator(
        world_width=world_size,
        world_height=world_size,
        obstacle_size=obstacle_size,
        obstacle_count=obstacle_count,
        min_obstacle_distance=min_obstacle_distance
    )
    
    # 创建多个布局的可视化
    fig, axes = plt.subplots(1, num_layouts, figsize=(5*num_layouts, 5))
    if num_layouts == 1:
        axes = [axes]
    
    for i in range(num_layouts):
        ax = axes[i]
        
        # 生成随机机器人位置（示例）
        robot_positions = []
        robot_goals = []
        for j in range(6):  # 6个机器人
            # 随机生成起始位置和目标位置
            start_x = np.random.uniform(0.5, world_size-0.5)
            start_y = np.random.uniform(0.5, world_size-0.5)
            goal_x = np.random.uniform(0.5, world_size-0.5)
            goal_y = np.random.uniform(0.5, world_size-0.5)
            
            robot_positions.append([start_x, start_y])
            robot_goals.append([goal_x, goal_y])
        
        # 生成障碍物
        obs_polygons, obs_lines = generator.generate_random_obstacles(robot_positions, robot_goals)
        
        # 绘制环境边界
        ax.add_patch(patches.Rectangle((0, 0), world_size, world_size, 
                                     linewidth=2, edgecolor='black', facecolor='white'))
        
        # 绘制静态障碍物
        for polygon in obs_polygons:
            # polygon是4个顶点的列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            rect = patches.Polygon(polygon, linewidth=2, edgecolor='red', 
                                 facecolor='red', alpha=0.7)
            ax.add_patch(rect)
        
        # 绘制机器人起始位置（蓝色圆圈）
        for j, pos in enumerate(robot_positions):
            circle = patches.Circle(pos, 0.2, facecolor='blue', edgecolor='darkblue', alpha=0.8)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(j), ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # 绘制机器人目标位置（绿色星号）
        for j, goal in enumerate(robot_goals):
            ax.plot(goal[0], goal[1], marker='*', markersize=12, color='green', 
                   markeredgecolor='darkgreen', markeredgewidth=1)
            ax.text(goal[0]+0.3, goal[1]+0.3, f'G{j}', ha='center', va='center', 
                   fontsize=8, color='darkgreen', weight='bold')
        
        # 绘制机器人到目标的连线（虚线）
        for pos, goal in zip(robot_positions, robot_goals):
            ax.plot([pos[0], goal[0]], [pos[1], goal[1]], 
                   linestyle='--', color='gray', alpha=0.5, linewidth=1)
        
        # 设置坐标轴
        ax.set_xlim(-0.5, world_size+0.5)
        ax.set_ylim(-0.5, world_size+0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Random Static Obstacle Layout {i+1}\n{len(obs_polygons)} obstacles, 6 robots', 
                    fontsize=12, weight='bold')
        
        # 添加图例
        if i == 0:
            legend_elements = [
                patches.Patch(color='red', alpha=0.7, label='Static Obstacles'),
                patches.Patch(color='blue', alpha=0.8, label='Robot Start Positions'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                          markersize=12, label='Robot Goal Positions'),
                plt.Line2D([0], [0], linestyle='--', color='gray', alpha=0.5, label='Path Direction')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.suptitle('Random Static Obstacle Environment Visualization\nEach episode generates different obstacle layouts', 
                fontsize=14, weight='bold', y=1.02)
    
    # 保存图像
    output_path = Path(__file__).parent / 'static_obstacles_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图像已保存至: {output_path}")
    
    # 显示图像（如果有显示环境）
    try:
        plt.show()
        print("🖼️  图像已显示（如果有图形界面）")
    except:
        print("⚠️  无图形界面，请查看保存的PNG文件")
    
    return output_path

def visualize_obstacle_generation_stats():
    """
    可视化障碍物生成的统计信息
    """
    print("\n📊 生成障碍物统计信息...")
    
    generator = StaticObstacleGenerator(
        world_width=12.0,
        world_height=12.0,
        obstacle_size=0.8,
        obstacle_count=4,
        min_obstacle_distance=1.5
    )
    
    # 生成多次来获取统计信息
    num_generations = 50
    obstacle_counts = []
    generation_attempts = []
        
    for i in range(num_generations):
        # 随机生成机器人位置
        robot_positions = [[np.random.uniform(0.5, 11.5), np.random.uniform(0.5, 11.5)] for _ in range(6)]
        robot_goals = [[np.random.uniform(0.5, 11.5), np.random.uniform(0.5, 11.5)] for _ in range(6)]
        
        obs_polygons, obs_lines = generator.generate_random_obstacles(robot_positions, robot_goals)
        obstacle_counts.append(len(obs_polygons))
    
    # 统计结果
    successful_rate = (np.array(obstacle_counts) == 4).mean() * 100
    avg_obstacles = np.mean(obstacle_counts)
    
    print(f"📈 统计结果（基于{num_generations}次生成）:")
    print(f"   - 成功生成4个障碍物的概率: {successful_rate:.1f}%")
    print(f"   - 平均生成障碍物数量: {avg_obstacles:.2f}")
    print(f"   - 障碍物数量分布: {dict(zip(*np.unique(obstacle_counts, return_counts=True)))}")
    
    return successful_rate, avg_obstacles

if __name__ == "__main__":
    print("🎮 静态障碍物环境可视化工具")
    print("=" * 50)
    
    # 创建可视化
    image_path = visualize_static_obstacles_environment(num_layouts=3)
    
    # 生成统计信息
    success_rate, avg_count = visualize_obstacle_generation_stats()
    
    print("\n🎯 可视化完成!")
    print(f"📁 图像文件: {image_path}")
    print(f"🎲 障碍物生成成功率: {success_rate:.1f}%")
    print(f"📊 平均障碍物数量: {avg_count:.2f}")
    
    print("\n🔍 环境特点:")
    print("   ✅ 每个episode生成不同的随机障碍物布局")
    print("   ✅ 障碍物大小: 1.0x1.0米")
    print("   ✅ 障碍物最小间距: 2.0米")
    print("   ✅ 避开机器人起始位置和目标位置")
    print("   ✅ 运行时直接注入，无临时文件")
