"""
可视化工具：绘制 Q-Table 热力图、训练曲线等
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_q_table_heatmap(q_table: np.ndarray, env, save_path: str = None):
    """
    绘制 Q-Table 热力图
    
    Args:
        q_table: Q-Table (n_states x n_actions)
        env: Gymnasium 环境，用于获取网格尺寸
        save_path: 保存路径，若为 None 则显示图像
    """
    # 获取状态价值（每个状态的最大 Q 值）
    state_values = np.max(q_table, axis=1)
    
    # 尝试推断网格大小（适用于 FrozenLake 等网格环境）
    n_states = q_table.shape[0]
    grid_size = int(np.sqrt(n_states))
    
    if grid_size * grid_size == n_states:
        # 方形网格环境
        grid_values = state_values.reshape((grid_size, grid_size))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(grid_values, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'State Value (max Q)'}, ax=ax)
        ax.set_title('Q-Table State Values Heatmap')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    else:
        # 非网格环境，直接显示柱状图
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(n_states)
        ax.bar(x, state_values, color='steelblue')
        ax.set_xlabel('State')
        ax.set_ylabel('State Value (max Q)')
        ax.set_title('State Values from Q-Table')
        ax.set_xticks(x)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_training_curves(rewards: list, save_path: str = None, title: str = "Training Curve"):
    """
    绘制训练奖励曲线
    
    Args:
        rewards: 每轮的奖励列表
        save_path: 保存路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.6, color='steelblue')
    
    # 添加移动平均线
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax.plot(range(99, len(rewards)), moving_avg, color='red', linewidth=2, label='MA(100)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curve to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_metrics(metrics_dict: dict, save_path: str = None):
    """
    绘制多个指标曲线
    
    Args:
        metrics_dict: 指标字典，如 {'success_rate': [...], 'avg_steps': [...]}
        save_path: 保存路径
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics_dict.items()):
        ax.plot(values, color='steelblue')
        ax.set_xlabel('Episode')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
        plt.close()
    else:
        plt.show()
