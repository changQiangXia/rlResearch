"""
PPO Training Script (Stage 4)
使用并行环境充分利用 GPU
"""
import os
import sys
import numpy as np
import gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage4_ppo.agent import PPOAgent
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


def train(
    env_name: str = 'Acrobot-v1',
    num_envs: int = 16,          # 并行环境数
    num_steps: int = 256,        # 每轮收集步数
    total_updates: int = 500,    # 总更新次数
    num_epochs: int = 10,        # 每次数据训练轮数
    batch_size: int = 256,       # Mini-batch 大小
    
    hidden_dims: list = [512, 512],
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    
    log_interval: int = 10,
    save_interval: int = 100,
    results_dir: str = None,
):
    """
    PPO 训练主函数
    
    Args:
        num_envs: 并行环境数量（推荐 8-16）
        num_steps: 每次收集的步数（推荐 128-512）
        total_updates: 总更新次数
        num_epochs: 每个数据集的训练轮数（PPO 特有，推荐 3-10）
        batch_size: Mini-batch 大小
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage4')
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 PPO Training - Stage 4")
    print(f"Device: {device}")
    print(f"Parallel envs: {num_envs}")
    print(f"Steps per update: {num_steps * num_envs}")
    print(f"PPO epochs: {num_epochs}")
    print("-" * 70)
    
    # 创建并行环境
    envs = [gym.make(env_name) for _ in range(num_envs)]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 70)
    
    # 创建 PPO Agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
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
    
    # 初始化
    states = np.array([env.reset() for env in envs])
    episode_rewards = [0.0] * num_envs
    episode_counts = [0] * num_envs
    all_episode_rewards = []
    
    print(f"{'Update':>8} | {'Episodes':>10} | {'AvgReward':>10} | {'Policy':>10} | {'Value':>10} | {'ClipFrac':>8} | {'ExpVar':>8}")
    print("-" * 85)
    
    for update in range(1, total_updates + 1):
        # 存储数据
        batch_states = np.zeros((num_steps, num_envs, state_dim), dtype=np.float32)
        batch_actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        batch_log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        batch_rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        batch_values = np.zeros((num_steps, num_envs), dtype=np.float32)
        batch_dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        # 收集数据
        for step in range(num_steps):
            batch_states[step] = states
            
            # 并行选择动作
            actions, log_probs, _, values = zip(*[agent.select_action(s) for s in states])
            batch_actions[step] = actions
            batch_log_probs[step] = log_probs
            batch_values[step] = values
            
            # 执行动作
            for i, env in enumerate(envs):
                next_state, reward, done, _ = env.step(actions[i])
                batch_rewards[step, i] = reward
                batch_dones[step, i] = float(done)
                
                episode_rewards[i] += reward
                
                if done:
                    all_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    episode_counts[i] += 1
                    states[i] = env.reset()
                else:
                    states[i] = next_state
        
        # 计算下一个状态的价值
        with torch.no_grad():
            next_states_tensor = torch.FloatTensor(states).to(device)
            _, next_values = agent.network(next_states_tensor)
            next_values = next_values.squeeze().cpu().numpy()
        
        # 计算 GAE
        advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        for env_id in range(num_envs):
            adv, ret = agent.compute_gae(
                batch_rewards[:, env_id],
                batch_values[:, env_id],
                batch_dones[:, env_id],
                next_values[env_id]
            )
            advantages[:, env_id] = adv
            returns[:, env_id] = ret
        
        # 展平数据 (num_steps, num_envs) -> (num_steps * num_envs,)
        batch_states = batch_states.reshape(-1, state_dim)
        batch_actions = batch_actions.reshape(-1)
        batch_log_probs = batch_log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        
        # PPO 更新
        loss_dict = agent.update(
            batch_states, batch_actions, batch_log_probs,
            advantages, returns,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        # 日志
        if update % log_interval == 0:
            avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(all_episode_rewards) if all_episode_rewards else 0.0
            total_episodes = sum(episode_counts)
            
            print(f"{update:8d} | {total_episodes:10d} | {avg_reward:10.2f} | "
                  f"{loss_dict['policy_loss']:10.4f} | {loss_dict['value_loss']:10.4f} | "
                  f"{loss_dict['clip_fraction']:8.4f} | {loss_dict['explained_variance']:8.4f}")
            
            # 记录到 TensorBoard
            logger.log_scalar('train/avg_reward', avg_reward, update)
            logger.log_scalar('train/policy_loss', loss_dict['policy_loss'], update)
            logger.log_scalar('train/value_loss', loss_dict['value_loss'], update)
            logger.log_scalar('train/entropy', loss_dict['entropy'], update)
            logger.log_scalar('train/clip_fraction', loss_dict['clip_fraction'], update)
            logger.log_scalar('train/explained_variance', loss_dict['explained_variance'], update)
        
        # 保存模型
        if update % save_interval == 0:
            model_path = os.path.join(results_dir, f'ppo_model_update{update}.pth')
            agent.save(model_path)
    
    print("-" * 85)
    print("Training completed!")
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'ppo_model_final.pth')
    agent.save(final_model_path)
    
    # 保存训练曲线
    plot_training_curves(
        all_episode_rewards,
        save_path=os.path.join(results_dir, 'training_rewards.png'),
        title='PPO Training Rewards'
    )
    
    # 保存 PPO 特有指标
    metrics_dict = {
        'policy_loss': agent.policy_losses,
        'value_loss': agent.value_losses,
        'entropy': agent.entropy_losses,
        'clip_fraction': agent.clip_fractions,
        'explained_variance': agent.explained_variances,
    }
    plot_metrics(metrics_dict, save_path=os.path.join(results_dir, 'training_metrics.png'))
    
    # 关闭环境
    for env in envs:
        env.close()
    logger.close()
    
    return agent, all_episode_rewards


if __name__ == '__main__':
    # PPO 训练配置
    agent, rewards = train(
        env_name='Acrobot-v1',
        num_envs=16,        # 16 并行环境
        num_steps=256,      # 每轮 256 步
        total_updates=500,  # 共 500 次更新
        num_epochs=10,      # 每批数据训练 10 轮（PPO 核心）
        batch_size=256,     # Mini-batch 大小
        
        hidden_dims=[512, 512],
        learning_rate=3e-4,
        clip_epsilon=0.2,   # PPO clipping
        entropy_coef=0.01,
    )
    
    print(f"\nFinal Stats:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Avg last 100: {np.mean(rewards[-100:]):.2f}")
    print(f"  Best: {max(rewards):.2f}")
