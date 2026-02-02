"""
DQN 训练脚本 (Stage 2)
环境: CartPole-v1
"""
import os
import sys
import numpy as np
import gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage2_dqn.agent import DQNAgent, DoubleDQNAgent
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


def train(
    # 环境参数
    env_name: str = 'CartPole-v1',
    
    # 训练参数
    total_episodes: int = 500,
    max_steps_per_episode: int = 500,
    warmup_steps: int = 1000,  # 开始训练前的预热步数
    
    # DQN 参数
    hidden_dims: list = [128, 128],
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    use_double_dqn: bool = False,
    
    # 日志参数
    log_interval: int = 10,
    save_interval: int = 100,
    eval_interval: int = 50,
    results_dir: str = None,
):
    """
    DQN 训练主函数
    
    Args:
        env_name: 环境名称
        total_episodes: 总训练轮数
        max_steps_per_episode: 每轮最大步数
        warmup_steps: 开始训练前的预热步数
        hidden_dims: 网络隐藏层维度
        learning_rate: 学习率
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最小探索率
        epsilon_decay: 探索率衰减系数
        buffer_capacity: 回放缓冲区容量
        batch_size: 训练批次大小
        target_update_freq: Target Network 更新频率（步数）
        use_double_dqn: 是否使用 Double DQN
        log_interval: 打印日志间隔
        save_interval: 保存模型间隔
        eval_interval: 评估间隔
        results_dir: 结果保存目录
    """
    # 默认结果目录
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage2')
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Use Double DQN: {use_double_dqn}")
    print("-" * 50)
    
    # 创建 Agent
    AgentClass = DoubleDQNAgent if use_double_dqn else DQNAgent
    agent = AgentClass(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device,
    )
    
    # 创建 Logger
    logger = Logger(
        log_dir=os.path.join(results_dir, 'logs'),
        use_tensorboard=True,
    )
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    total_steps = 0
    
    print("Starting training...")
    print(f"{'Episode':>8} | {'Reward':>8} | {'Steps':>6} | {'Loss':>8} | {'Epsilon':>8} | {'Buffer':>8}")
    print("-" * 65)
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新（warmup 后开始）
            if total_steps >= warmup_steps:
                loss = agent.update()
                if loss > 0:
                    episode_loss.append(loss)
            
            episode_reward += reward
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        # 记录统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 打印日志
        if episode % log_interval == 0:
            recent_reward = np.mean(episode_rewards[-log_interval:])
            recent_steps = np.mean(episode_lengths[-log_interval:])
            buffer_size = len(agent.replay_buffer)
            
            print(f"{episode:8d} | {recent_reward:8.2f} | {recent_steps:6.1f} | {avg_loss:8.4f} | {agent.epsilon:8.4f} | {buffer_size:8d}")
            
            # 记录到 TensorBoard
            logger.log_scalar('train/reward', recent_reward, episode)
            logger.log_scalar('train/steps', recent_steps, episode)
            logger.log_scalar('train/loss', avg_loss, episode)
            logger.log_scalar('train/epsilon', agent.epsilon, episode)
            logger.log_scalar('train/buffer_size', buffer_size, episode)
            
            # 记录 Q 值分布（可选）
            if episode % (log_interval * 5) == 0:
                with torch.no_grad():
                    sample_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = agent.policy_net(sample_state).cpu().numpy()[0]
                    logger.log_histogram('train/q_values', q_values, episode)
        
        # 评估
        if episode % eval_interval == 0:
            eval_reward = evaluate(agent, env_name, n_episodes=10)
            eval_rewards.append(eval_reward)
            print(f"  [Eval] Episode {episode}: Avg Reward = {eval_reward:.2f}")
            logger.log_scalar('eval/reward', eval_reward, episode)
        
        # 保存模型
        if episode % save_interval == 0:
            model_path = os.path.join(results_dir, f'dqn_model_ep{episode}.pth')
            agent.save(model_path)
    
    print("-" * 65)
    print("Training completed!")
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'dqn_model_final.pth')
    agent.save(final_model_path)
    
    # 保存训练曲线
    plot_training_curves(episode_rewards, 
                        save_path=os.path.join(results_dir, 'training_rewards.png'),
                        title='DQN Training Rewards')
    
    metrics_dict = {
        'loss': agent.losses if agent.losses else [0],
    }
    plot_metrics(metrics_dict, save_path=os.path.join(results_dir, 'training_loss.png'))
    
    # 打印最终统计
    final_avg_reward = np.mean(episode_rewards[-100:])
    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Avg Reward: {final_avg_reward:.2f}")
    print(f"  Best Reward: {max(episode_rewards):.2f}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    print(f"  Total Steps: {total_steps}")
    print(f"\nResults saved to: {results_dir}")
    
    logger.close()
    env.close()
    
    return agent, episode_rewards


def evaluate(agent, env_name: str = 'CartPole-v1', n_episodes: int = 10):
    """
    评估 Agent（贪婪策略，无探索）
    
    Args:
        agent: DQNAgent
        env_name: 环境名称
        n_episodes: 评估轮数
    
    Returns:
        平均奖励
    """
    env = gym.make(env_name)
    total_reward = 0
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(500):
            action = agent.select_action(state, epsilon=0)  # 纯贪婪
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    env.close()
    return total_reward / n_episodes


if __name__ == '__main__':
    # 训练配置
    agent, rewards = train(
        env_name='CartPole-v1',
        total_episodes=500,
        hidden_dims=[128, 128],
        learning_rate=1e-3,
        use_double_dqn=False,  # 设为 True 使用 Double DQN
    )
    
    # 最终评估
    final_eval = evaluate(agent, n_episodes=100)
    print(f"\nFinal Evaluation (100 episodes): Avg Reward = {final_eval:.2f}")
