#!/usr/bin/env python3
"""
测试不同障碍物参数的成功率
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from rl_rvo_nav.utils.static_obstacle_generator import StaticObstacleGenerator

def test_obstacle_parameters():
    """测试不同参数组合的障碍物生成成功率"""
    
    print("🧪 Testing different obstacle parameters...")
    print("=" * 60)
    
    # 测试参数组合
    test_configs = [
        {
            "name": "Current Settings",
            "min_distance": 2.0,
            "obstacle_size": 1.0,
            "world_size": 10.0,
            "margin": 1.0
        },
        {
            "name": "Reduced Distance",
            "min_distance": 1.5,
            "obstacle_size": 1.0,
            "world_size": 10.0,
            "margin": 1.0
        },
        {
            "name": "Smaller Obstacles",
            "min_distance": 2.0,
            "obstacle_size": 0.8,
            "world_size": 10.0,
            "margin": 1.0
        },
        {
            "name": "Larger World",
            "min_distance": 2.0,
            "obstacle_size": 1.0,
            "world_size": 12.0,
            "margin": 1.0
        },
        {
            "name": "Optimized Settings",
            "min_distance": 1.5,
            "obstacle_size": 0.8,
            "world_size": 12.0,
            "margin": 1.0
        }
    ]
    
    num_tests = 100
    results = []
    
    for config in test_configs:
        print(f"\n🔧 Testing: {config['name']}")
        print(f"   Min distance: {config['min_distance']}m")
        print(f"   Obstacle size: {config['obstacle_size']}×{config['obstacle_size']}m")
        print(f"   World size: {config['world_size']}×{config['world_size']}m")
        
        generator = StaticObstacleGenerator(
            world_width=config['world_size'],
            world_height=config['world_size'],
            obstacle_size=config['obstacle_size'],
            obstacle_count=4,
            min_obstacle_distance=config['min_distance'],
            margin=config['margin']
        )
        
        obstacle_counts = []
        success_count = 0
        
        for i in range(num_tests):
            # 生成随机机器人位置
            robot_positions = []
            robot_goals = []
            for j in range(6):
                start_x = np.random.uniform(0.5, config['world_size']-0.5)
                start_y = np.random.uniform(0.5, config['world_size']-0.5)
                goal_x = np.random.uniform(0.5, config['world_size']-0.5)
                goal_y = np.random.uniform(0.5, config['world_size']-0.5)
                
                robot_positions.append([start_x, start_y])
                robot_goals.append([goal_x, goal_y])
            
            # 生成障碍物
            obs_polygons, obs_lines = generator.generate_random_obstacles(robot_positions, robot_goals)
            obstacle_count = len(obs_polygons)
            obstacle_counts.append(obstacle_count)
            
            if obstacle_count == 4:
                success_count += 1
        
        success_rate = success_count / num_tests * 100
        avg_obstacles = np.mean(obstacle_counts)
        
        print(f"   ✅ Success rate (4 obstacles): {success_rate:.1f}%")
        print(f"   📊 Average obstacles: {avg_obstacles:.2f}")
        print(f"   📈 Distribution: {dict(zip(*np.unique(obstacle_counts, return_counts=True)))}")
        
        results.append({
            'config': config['name'],
            'success_rate': success_rate,
            'avg_obstacles': avg_obstacles,
            'distribution': dict(zip(*np.unique(obstacle_counts, return_counts=True)))
        })
    
    # 显示总结
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    
    for result in sorted(results, key=lambda x: x['success_rate'], reverse=True):
        print(f"{result['config']:<20}: {result['success_rate']:6.1f}% success, {result['avg_obstacles']:.2f} avg obstacles")
    
    # 推荐最佳配置
    best_config = max(results, key=lambda x: x['success_rate'])
    print(f"\n🏆 BEST CONFIGURATION: {best_config['config']}")
    print(f"   Success Rate: {best_config['success_rate']:.1f}%")
    print(f"   Average Obstacles: {best_config['avg_obstacles']:.2f}")
    
    return results

if __name__ == "__main__":
    results = test_obstacle_parameters()
