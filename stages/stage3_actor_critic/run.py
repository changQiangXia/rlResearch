"""
A2C 训练脚本 (Stage 3)
环境: LunarLander-v2
"""
import os
import sys
import numpy as np
import gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage3_actor_critic.agent import A2CAgent
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


def train(
    # 环境参数
    env_name: str = 'LunarLander-v2',
    
    # 训练参数
    total_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    
    # A2C 参数
    hidden_dims: list = [256, 256],
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    
    # 日志参数
    log_interval: int = 10,
    save_interval: int = 100,
    eval_interval: int = 50,
    results_dir: str = None,
):
    """
    A2C 训练主函数
    
    Args:
        env_name: 环境名称
        total_episodes: 总训练轮数
        max_steps_per_episode: 每轮最大步数
        hidden_dims: 网络隐藏层维度
        learning_rate: 学习率
        gamma: 折扣因子
        gae_lambda: GAE 参数
        value_coef: Value Loss 系数
        entropy_coef: Entropy Loss 系数
        max_grad_norm: 梯度裁剪阈值
        log_interval: 打印日志间隔
        save_interval: 保存模型间隔
        eval_interval: 评估间隔
        results_dir: 结果保存目录
    """
    # 默认结果目录
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage3')
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
    print(f"Entropy coef: {entropy_coef}")
    print("-" * 70)
    
    # 创建 Agent
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
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
    
    print("Starting training...")
    print(f"{'Episode':>8} | {'Reward':>8} | {'Steps':>6} | {'Policy':>10} | {'Value':>10} | {'Entropy':>8}")
    print("-" * 70)
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        
        # 收集一条轨迹
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }
        
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action, log_prob, entropy, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储数据
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['dones'].append(float(done))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_length = step + 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 计算 GAE
        next_value = 0 if done else agent.network.get_value(
            torch.FloatTensor(state).unsqueeze(0).to(device)
        ).item()
        
        advantages, returns = agent.compute_gae(
            trajectory['rewards'],
            trajectory['values'],
            trajectory['dones'],
            next_value,
        )
        
        # 转换为张量
        states = torch.FloatTensor(np.array(trajectory['states'])).to(device)
        actions = torch.LongTensor(np.array(trajectory['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(trajectory['log_probs'])).to(device)
        advantages = torch.FloatTensor(np.array(advantages)).to(device)
        returns = torch.FloatTensor(np.array(returns)).to(device)
        
        # 更新网络
        loss_dict = agent.update(states, actions, old_log_probs, advantages, returns)
        
        # 打印日志
        if episode % log_interval == 0:
            recent_reward = np.mean(episode_rewards[-log_interval:])
            recent_steps = np.mean(episode_lengths[-log_interval:])
            
            print(f"{episode:8d} | {recent_reward:8.2f} | {recent_steps:6.1f} | "
                  f"{loss_dict['policy_loss']:10.4f} | {loss_dict['value_loss']:10.4f} | "
                  f"{loss_dict['entropy']:8.4f}")
            
            # 记录到 TensorBoard
            logger.log_scalar('train/reward', recent_reward, episode)
            logger.log_scalar('train/steps', recent_steps, episode)
            logger.log_scalar('train/policy_loss', loss_dict['policy_loss'], episode)
            logger.log_scalar('train/value_loss', loss_dict['value_loss'], episode)
            logger.log_scalar('train/entropy', loss_dict['entropy'], episode)
        
        # 评估
        if episode % eval_interval == 0:
            eval_reward = evaluate(agent, env_name, n_episodes=5)
            eval_rewards.append(eval_reward)
            print(f"  [Eval] Episode {episode}: Avg Reward = {eval_reward:.2f}")
            logger.log_scalar('eval/reward', eval_reward, episode)
        
        # 保存模型
        if episode % save_interval == 0:
            model_path = os.path.join(results_dir, f'a2c_model_ep{episode}.pth')
            agent.save(model_path)
    
    print("-" * 70)
    print("Training completed!")
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'a2c_model_final.pth')
    agent.save(final_model_path)
    
    # 保存训练曲线
    plot_training_curves(
        episode_rewards,
        save_path=os.path.join(results_dir, 'training_rewards.png'),
        title='A2C Training Rewards'
    )
    
    metrics_dict = {
        'policy_loss': agent.policy_losses,
        'value_loss': agent.value_losses,
        'entropy': agent.entropy_losses,
    }
    plot_metrics(metrics_dict, save_path=os.path.join(results_dir, 'training_losses.png'))
    
    # 打印最终统计
    final_avg_reward = np.mean(episode_rewards[-100:])
    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Avg Reward: {final_avg_reward:.2f}")
    print(f"  Best Reward: {max(episode_rewards):.2f}")
    print(f"  Final Entropy: {loss_dict['entropy']:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    logger.close()
    env.close()
    
    return agent, episode_rewards


def evaluate(agent, env_name: str = 'Acrobot-v1', n_episodes: int = 5):
    """
    评估 Agent（贪婪策略）
    
    Args:
        agent: A2CAgent
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
        
        for _ in range(1000):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_logits, _ = agent.network(state_tensor)
                action = torch.distributions.Categorical(logits=action_logits).probs.argmax().item()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    env.close()
    return total_reward / n_episodes


if __name__ == '__main__':
    # 训练配置
    # 可选环境: 'LunarLander-v2' (需要 Box2D), 'Acrobot-v1', 'MountainCar-v0'
    agent, rewards = train(
        env_name='Acrobot-v1',  # 不需要 Box2D
        total_episodes=1000,
        hidden_dims=[256, 256],
        learning_rate=3e-4,
        entropy_coef=0.01,
    )
    
    # 最终评估
    final_eval = evaluate(agent, env_name='Acrobot-v1', n_episodes=10)
    print(f"\nFinal Evaluation (10 episodes): Avg Reward = {final_eval:.2f}")
