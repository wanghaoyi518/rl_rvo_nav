import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from map_config_cplx import MAP_CONFIG
from python_pnr.sub_map import SubMap
from python_pnr.actor import Actor
from python_pnr.actor_set import ActorSet
from python_pnr.isearch import ISearch
from python_pnr.node import Point

def debug_agent12():
    # 创建地图和agent
    grid = MAP_CONFIG['grid']
    agents = MAP_CONFIG['agents']
    sub_map = SubMap(grid)
    actor_set = ActorSet()

    # 添加前12个agent（模拟算法执行到第13步的状态）
    for i in range(12):
        agent = agents[i]
        start = Point(agent['start'][0], agent['start'][1])
        goal = Point(agent['goal'][0], agent['goal'][1])
        actor_set.add_actor(Actor(agent['id'], start, goal))

    # 检查Agent 12的情况
    agent12 = agents[12]
    start = Point(agent12['start'][0], agent12['start'][1])
    goal = Point(agent12['goal'][0], agent12['goal'][1])
    print(f'Agent 12: start={agent12["start"]}, goal={agent12["goal"]}')
    print(f'Start traversable: {sub_map.is_traversable(start.x, start.y)}')
    print(f'Goal traversable: {sub_map.is_traversable(goal.x, goal.y)}')

    # 尝试搜索路径
    search = ISearch(sub_map)
    occupied_nodes = set((a.current.x, a.current.y) for a in actor_set)
    print(f'Occupied nodes: {occupied_nodes}')
    print(f'Number of occupied nodes: {len(occupied_nodes)}')

    # 检查地图状态
    print("\n=== 地图状态分析 ===")
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:  # 可通行位置
                is_occupied = (i, j) in occupied_nodes
                print(f"({i},{j}): {'OCCUPIED' if is_occupied else 'FREE'}")

    result = search.startSearch(sub_map, actor_set, start.x, start.y, goal.x, goal.y, 
                              occupied_nodes=occupied_nodes)
    print(f'\nPath found: {result.pathfound}')
    if result.pathfound:
        print(f'Path: {[(n.i, n.j) for n in result.lppath]}')
    else:
        print('No path found')
        print(f'Search steps: {result.numberofsteps}')
        print(f'Nodes created: {result.nodescreated}')

if __name__ == "__main__":
    debug_agent12() 