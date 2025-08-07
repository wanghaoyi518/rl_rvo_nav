import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from .node import Node, Point

class PushAndRotateVisualizer:
    def __init__(self, grid, agents_config, paths_result):
        self.grid = grid
        self.agents_config = agents_config
        self.paths_result = paths_result
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        
        # 为每个agent分配不同颜色
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
    def create_figure(self, figsize=(10, 10)):
        """创建基础图形"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 使y轴从上到下递增
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴标签
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # 设置坐标轴刻度
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))
        
        return fig, ax
    
    def draw_grid(self, ax):
        """绘制网格和障碍物"""
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 1:  # 障碍物
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                           facecolor='black', edgecolor='black')
                    ax.add_patch(rect)
                else:  # 空闲格子
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                           facecolor='white', edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
    
    def draw_agents_and_goals(self, ax, show_paths=False):
        # 只显示起点、终点和当前位置，不显示轨迹
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for idx, agent in enumerate(self.agents_config):
            color = colors[idx % len(colors)]
            # 起点 - 使用统一的(i,j)坐标系统
            start_i = agent['start'][0]  # row (i)
            start_j = agent['start'][1]  # column (j)
            ax.scatter(start_j, start_i, marker='o', color=color, s=100, 
                      label=f"Agent {idx+1} Start" if idx==0 else None, 
                      edgecolors='black', zorder=3)
            # 终点 - 使用统一的(i,j)坐标系统
            goal_i = agent['goal'][0]  # row (i)
            goal_j = agent['goal'][1]  # column (j)
            ax.scatter(goal_j, goal_i, marker='*', color=color, s=150, 
                      label=f"Agent {idx+1} Goal" if idx==0 else None, 
                      edgecolors='black', zorder=3)
        # 当前帧agent位置（由外部传入）
        # 由create_animation动态绘制

    def save_png(self, filename="push_and_rotate_result.png"):
        """保存PNG图片"""
        fig, ax = self.create_figure()
        self.draw_grid(ax)
        self.draw_agents_and_goals(ax, show_paths=True)
        
        plt.title("Push and Rotate Algorithm Result", fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PNG saved as: {filename}")
    
    def create_animation(self, filename="push_and_rotate_animation.gif", frames=50):
        import matplotlib.pyplot as plt
        from matplotlib import animation
        fig, ax = self.create_figure()
        self.draw_grid(ax)
        self.draw_agents_and_goals(ax, show_paths=False)
        # 计算最大路径长度
        max_len = max(len(path) for path in self.paths_result)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        def animate(frame):
            ax.clear()
            # 重新设置坐标轴属性，确保与静态图一致
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(-0.5, self.height - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # 使y轴从上到下递增，与静态图一致
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            ax.set_xticks(range(self.width))
            ax.set_yticks(range(self.height))
            ax.set_xticklabels(range(self.width))
            ax.set_yticklabels(range(self.height))
            
            self.draw_grid(ax)
            # 只显示起点、终点
            self.draw_agents_and_goals(ax, show_paths=False)
            # 只显示当前位置
            for idx, path in enumerate(self.paths_result):
                color = colors[idx % len(colors)]
                if frame < len(path):
                    pos = path[frame]
                else:
                    pos = path[-1]
                # 使用统一的(i,j)坐标系统 - Point.x=row(i), Point.y=col(j)
                # 在matplotlib中：x轴=column(j), y轴=row(i)
                # 与静态位置保持一致：ax.scatter(start_j, start_i)
                ax.scatter(pos.y, pos.x, marker='s', color=color, s=120, edgecolors='black', zorder=4)
            ax.set_title(f"Push-and-Rotate Step {frame+1}")
        anim = animation.FuncAnimation(fig, animate, frames=max_len, interval=2000)
        anim.save(filename, writer='pillow')
        plt.close(fig)
        print(f"GIF saved as: {filename}")

def visualize_push_and_rotate_result(grid, agents_config, paths_result, 
                                   png_filename="push_and_rotate_result.png",
                                   gif_filename="push_and_rotate_animation.gif"):
    """便捷函数：生成PNG和GIF可视化"""
    visualizer = PushAndRotateVisualizer(grid, agents_config, paths_result)
    visualizer.save_png(png_filename)
    visualizer.create_animation(gif_filename) 