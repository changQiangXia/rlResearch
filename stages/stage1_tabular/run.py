"""
Q-Learning 训练脚本 (Stage 1)
环境: FrozenLake-v1
"""
import os
import sys
import numpy as np
import gym

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage1_tabular.agent import QLearningAgent
from common.logger import Logger
from utils.visualization import plot_q_table_heatmap, plot_metrics


def train(
    # 环境参数
    env_name: str = 'FrozenLake-v1',
    env_kwargs: dict = None,
    
    # 训练参数
    total_episodes: int = 5000,
    max_steps_per_episode: int = 100,
    
    # Q-Learning 参数
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    
    # 日志参数
    log_interval: int = 100,
    save_interval: int = 500,
    results_dir: str = None,
):
    """
    Q-Learning 训练主函数
    
    Args:
        env_name: 环境名称
        env_kwargs: 环境额外参数，如 {'map_name': '4x4', 'is_slippery': False}
        total_episodes: 总训练轮数
        max_steps_per_episode: 每轮最大步数
        learning_rate: 学习率 α
        gamma: 折扣因子 γ
        epsilon_start: 初始探索率
        epsilon_end: 最小探索率
        epsilon_decay: 探索率衰减系数
        log_interval: 打印日志间隔（轮数）
        save_interval: 保存热力图间隔（轮数）
        results_dir: 结果保存目录
    """
    # 默认环境参数
    if env_kwargs is None:
        env_kwargs = {'map_name': '4x4', 'is_slippery': False}
    
    # 默认结果目录
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage1')
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建环境
    env = gym.make(env_name, **env_kwargs)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State space size: {n_states}")
    print(f"Action space size: {n_actions}")
    print(f"Environment kwargs: {env_kwargs}")
    print("-" * 50)
    
    # 创建 Agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
    )
    
    # 创建 Logger
    logger = Logger(log_dir=os.path.join(results_dir, 'logs'))
    
    # 训练统计
    epsilon = epsilon_start
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # 记录每 log_interval 轮的指标
    metrics_history = {
        'success_rate': [],
        'avg_reward': [],
        'avg_steps': [],
        'epsilon': [],
    }
    
    print("Starting training...")
    print(f"{'Episode':>8} | {'Success':>7} | {'AvgSteps':>8} | {'AvgReward':>9} | {'Epsilon':>7}")
    print("-" * 60)
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state, epsilon)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # Q-Learning 更新
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # 记录本轮统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 0:  # FrozenLake 中成功到达目标才有正奖励
            success_count += 1
        
        # 衰减探索率（线性衰减）
        epsilon = max(epsilon_end, epsilon_start - episode * (epsilon_start - epsilon_end) / total_episodes)
        
        # 定期打印日志
        if episode % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_lengths = episode_lengths[-log_interval:]
            
            success_rate = sum([1 for r in recent_rewards if r > 0]) / log_interval
            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_lengths)
            
            metrics_history['success_rate'].append(success_rate)
            metrics_history['avg_reward'].append(avg_reward)
            metrics_history['avg_steps'].append(avg_steps)
            metrics_history['epsilon'].append(epsilon)
            
            print(f"{episode:8d} | {success_rate:7.2%} | {avg_steps:8.2f} | {avg_reward:9.4f} | {epsilon:7.4f}")
            
            # 记录到 TensorBoard
            logger.log_scalar('train/success_rate', success_rate, episode)
            logger.log_scalar('train/avg_reward', avg_reward, episode)
            logger.log_scalar('train/avg_steps', avg_steps, episode)
            logger.log_scalar('train/epsilon', epsilon, episode)
        
        # 定期保存 Q-Table 热力图
        if episode % save_interval == 0:
            q_table = agent.get_q_table()
            save_path = os.path.join(results_dir, f'qtable_heatmap_ep{episode}.png')
            plot_q_table_heatmap(q_table, env, save_path)
    
    print("-" * 50)
    print("Training completed!")
    
    # 保存最终模型
    model_path = os.path.join(results_dir, 'q_learning_model.pkl')
    agent.save(model_path)
    
    # 保存最终热力图
    final_heatmap_path = os.path.join(results_dir, 'qtable_heatmap_final.png')
    plot_q_table_heatmap(agent.get_q_table(), env, final_heatmap_path)
    
    # 保存训练曲线
    metrics_plot_path = os.path.join(results_dir, 'training_metrics.png')
    plot_metrics(metrics_history, metrics_plot_path)
    
    # 打印最终统计
    final_success_rate = sum([1 for r in episode_rewards[-1000:] if r > 0]) / min(1000, len(episode_rewards))
    print(f"\nFinal Statistics (last 1000 episodes):")
    print(f"  Success Rate: {final_success_rate:.2%}")
    print(f"  Avg Steps: {np.mean(episode_lengths[-1000:]):.2f}")
    print(f"  Final Epsilon: {epsilon:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    logger.close()
    env.close()
    
    return agent, episode_rewards, episode_lengths


def evaluate(agent: QLearningAgent, env_name: str = 'FrozenLake-v1', 
             env_kwargs: dict = None, n_episodes: int = 100):
    """
    评估训练好的 Agent（贪婪策略，无探索）
    
    Args:
        agent: 训练好的 QLearningAgent
        env_name: 环境名称
        env_kwargs: 环境参数
        n_episodes: 评估轮数
    """
    if env_kwargs is None:
        env_kwargs = {'map_name': '4x4', 'is_slippery': False}
    
    env = gym.make(env_name, **env_kwargs)
    
    success_count = 0
    total_steps = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_length = 0
        
        for _ in range(100):
            action = agent.select_action(state, epsilon=0)  # 纯贪婪
            state, reward, terminated, truncated, _ = env.step(action)
            episode_length += 1
            
            if terminated or truncated:
                if reward > 0:
                    success_count += 1
                break
        
        total_steps += episode_length
    
    env.close()
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Success Rate: {success_count / n_episodes:.2%}")
    print(f"  Avg Steps: {total_steps / n_episodes:.2f}")


if __name__ == '__main__':
    # 训练配置
    agent, rewards, lengths = train(
        env_name='FrozenLake-v1',
        env_kwargs={'map_name': '4x4', 'is_slippery': False},
        total_episodes=5000,
        max_steps_per_episode=100,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        log_interval=100,
        save_interval=500,
    )
    
    # 评估
    evaluate(agent, env_name='FrozenLake-v1', 
             env_kwargs={'map_name': '4x4', 'is_slippery': False}, 
             n_episodes=100)
