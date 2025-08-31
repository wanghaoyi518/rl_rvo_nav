import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from map_config_2 import MAP_CONFIG
# from map_config_cplx import MAP_CONFIG
from python_pnr.sub_map import SubMap
from python_pnr.actor import Actor
from python_pnr.actor_set import ActorSet
from python_pnr.push_and_rotate import PushAndRotate
from python_pnr.mapf_config import MAPFConfig
from python_pnr.node import Point
from python_pnr.visualization import visualize_push_and_rotate_result

def run_push_and_rotate(map_config):
    grid = map_config["grid"]
    agents = map_config["agents"]
    sub_map = SubMap(grid)
    actor_set = ActorSet()
    
    print("=== 地图配置分析 ===")
    print(f"地图大小: {len(grid)}x{len(grid[0])}")
    print("地图网格:")
    for i, row in enumerate(grid):
        print(f"行{i}: {row}")
    
    print("\n=== Agent配置分析 ===")
    for agent in agents:
        # 颠倒x和y轴：地图配置中 (row, col) 对应 Point(x=row, y=col)
        start = Point(agent["start"][0], agent["start"][1])  # (row, col) -> (x=row, y=col)
        goal = Point(agent["goal"][0], agent["goal"][1])     # (row, col) -> (x=row, y=col)
        
        print(f"Agent {agent['id']}:")
        print(f"  配置: start={agent['start']}, goal={agent['goal']}")
        print(f"  转换: start=Point({start.x},{start.y}), goal=Point({goal.x},{goal.y})")
        print(f"  起始位置可通行: {sub_map.is_traversable(start.x, start.y)}")
        print(f"  目标位置可通行: {sub_map.is_traversable(goal.x, goal.y)}")
        
        actor_set.add_actor(Actor(agent["id"], start, goal))
    
    config = MAPFConfig(max_steps=100)
    solver = PushAndRotate()
    result = solver.start_search(sub_map, config, actor_set)
    print(f"\nSuccess: {result.success}")
    print(f"Agents moves count: {len(solver.agents_moves)}")
    print(f"Agents moves: {[(move.id, (move.di, move.dj)) for move in solver.agents_moves[:10]]}...")  # 显示前10个移动
    for aid, path in result.paths.items():
        print(f"Agent {aid} path: {[ (n.x, n.y) for n in path ]}")
    print(f"Total steps: {result.steps}")
    print(f"Runtime: {result.runtime:.4f}s")
    
    # 检查碰撞
    print("\nChecking for collisions...")
    if result.paths:
        max_path_length = max(len(path) for path in result.paths.values())
        for step in range(max_path_length):
            positions = set()
            collisions = []
            for aid, path in result.paths.items():
                if step < len(path):
                    pos = (path[step].x, path[step].y)
                    if pos in positions:
                        collisions.append((step, pos))
                    positions.add(pos)
            if collisions:
                print(f"COLLISION at step {step}: {collisions}")
                # 显示碰撞的agent
                for aid, path in result.paths.items():
                    if step < len(path):
                        pos = (path[step].x, path[step].y)
                        for collision in collisions:
                            if collision[1] == pos:
                                print(f"  Agent {aid} at {pos}")
            else:
                print(f"Step {step}: No collisions")
    else:
        print("No paths generated - algorithm failed")
    
    # 生成可视化
    if result.success:
        print("\nGenerating visualizations...")
        vis_dir = os.path.join(os.path.dirname(__file__), "vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 生成时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # 确保路径按照agent的ID顺序排列
        ordered_paths = []
        for agent in agents:
            if agent['id'] in result.paths:
                ordered_paths.append(result.paths[agent['id']])
            else:
                # 如果某个agent没有路径，创建一个空路径
                ordered_paths.append([])
        
        visualize_push_and_rotate_result(
            grid, agents, ordered_paths,
            png_filename=os.path.join(vis_dir, f"test_result_{timestamp}.png"),
            gif_filename=os.path.join(vis_dir, f"test_animation_{timestamp}.gif")
        )
    else:
        print("No visualization generated due to algorithm failure.")

if __name__ == "__main__":
    run_push_and_rotate(MAP_CONFIG) 