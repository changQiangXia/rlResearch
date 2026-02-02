"""
A2C 优化版 - 充分利用 GPU
使用 8 个并行环境 + 更大的 Batch Size
"""
import os
import sys
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.networks import ActorCriticNetwork
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


class VectorizedA2CAgent:
    """支持多环境并行的 A2C Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_envs: int = 8,  # 并行环境数
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.device = device
        
        # 创建网络
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation='relu',
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 训练统计
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def select_actions(self, states: np.ndarray):
        """
        批量选择动作 - 一次处理所有环境
        
        Args:
            states: shape (num_envs, state_dim)
        
        Returns:
            actions: shape (num_envs,)
            log_probs: shape (num_envs,)
            entropies: shape (num_envs,)
            values: shape (num_envs,)
        """
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            action_logits, values = self.network(states_tensor)
            probs = torch.distributions.Categorical(logits=action_logits)
            
            actions = probs.sample()
            log_probs = probs.log_prob(actions)
            entropies = probs.entropy()
            
            return (
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                entropies.cpu().numpy(),
                values.squeeze().cpu().numpy(),
            )
    
    def compute_gae(self, rewards, values, dones, next_values):
        """
        向量化 GAE 计算 - 同时处理所有环境
        
        Args:
            rewards: shape (num_steps, num_envs)
            values: shape (num_steps, num_envs)
            dones: shape (num_steps, num_envs)
            next_values: shape (num_envs,)
        """
        advantages = np.zeros_like(rewards)
        
        for env_id in range(self.num_envs):
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_v = next_values[env_id]
                else:
                    next_v = values[t + 1, env_id]
                
                delta = rewards[t, env_id] + self.gamma * next_v * (1 - dones[t, env_id]) - values[t, env_id]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t, env_id]) * gae
                advantages[t, env_id] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        批量更新 - 一次处理 num_steps * num_envs 个样本
        """
        # 展平数据 (num_steps, num_envs) -> (num_steps * num_envs,)
        batch_size = states.shape[0] * states.shape[1]
        states = states.reshape(batch_size, -1)
        actions = actions.reshape(batch_size)
        old_log_probs = old_log_probs.reshape(batch_size)
        advantages = advantages.reshape(batch_size)
        returns = returns.reshape(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 前向传播
        action_logits, values = self.network(states)
        probs = torch.distributions.Categorical(logits=action_logits)
        
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        
        # 计算 Loss
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
        }
        
        self.policy_losses.append(loss_dict['policy_loss'])
        self.value_losses.append(loss_dict['value_loss'])
        self.entropy_losses.append(loss_dict['entropy_loss'])
        
        return loss_dict
    
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Saved model to {filepath}")


def train(
    env_name: str = 'Acrobot-v1',
    num_envs: int = 8,  # 并行环境数
    num_steps: int = 128,  # 每次收集的步数
    total_updates: int = 1000,  # 总更新次数
    
    hidden_dims: list = [256, 256],
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    
    log_interval: int = 10,
    save_interval: int = 100,
    results_dir: str = None,
):
    """
    向量化 A2C 训练
    
    Args:
        num_envs: 并行环境数量（建议 4-8 个）
        num_steps: 每次收集的步数（建议 128-512）
        total_updates: 总更新次数
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage3_optimized')
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Parallel envs: {num_envs}")
    print(f"Batch size per update: {num_steps * num_envs}")
    
    # 创建多个环境
    envs = [gym.make(env_name) for _ in range(num_envs)]
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 70)
    
    # 创建 Agent
    agent = VectorizedA2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        num_envs=num_envs,
        device=device,
    )
    
    # 创建 Logger
    logger = Logger(
        log_dir=os.path.join(results_dir, 'logs'),
        use_tensorboard=True,
    )
    
    # 初始化环境
    states = np.array([env.reset() for env in envs])
    episode_rewards = [0.0] * num_envs
    episode_counts = [0] * num_envs
    all_episode_rewards = []
    
    print("Starting training...")
    print(f"{'Update':>8} | {'Episodes':>10} | {'AvgReward':>10} | {'Policy':>10} | {'Value':>10} | {'Entropy':>8}")
    print("-" * 75)
    
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
            
            actions, log_probs, _, values = agent.select_actions(states)
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
            states_tensor = torch.FloatTensor(states).to(device)
            _, next_values = agent.network(states_tensor)
            next_values = next_values.squeeze().cpu().numpy()
        
        # 计算 GAE
        advantages, returns = agent.compute_gae(
            batch_rewards, batch_values, batch_dones, next_values
        )
        
        # 更新网络
        loss_dict = agent.update(
            batch_states, batch_actions, batch_log_probs, advantages, returns
        )
        
        # 打印日志
        if update % log_interval == 0:
            avg_reward = np.mean(all_episode_rewards[-100:]) if len(all_episode_rewards) >= 100 else np.mean(all_episode_rewards) if all_episode_rewards else 0.0
            total_episodes = sum(episode_counts)
            
            print(f"{update:8d} | {total_episodes:10d} | {avg_reward:10.2f} | "
                  f"{loss_dict['policy_loss']:10.4f} | {loss_dict['value_loss']:10.4f} | "
                  f"{loss_dict['entropy']:8.4f}")
            
            # 记录到 TensorBoard
            logger.log_scalar('train/avg_reward', avg_reward, update)
            logger.log_scalar('train/policy_loss', loss_dict['policy_loss'], update)
            logger.log_scalar('train/value_loss', loss_dict['value_loss'], update)
            logger.log_scalar('train/entropy', loss_dict['entropy'], update)
        
        # 保存模型
        if update % save_interval == 0:
            model_path = os.path.join(results_dir, f'a2c_model_update{update}.pth')
            agent.save(model_path)
    
    print("-" * 75)
    print("Training completed!")
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'a2c_model_final.pth')
    agent.save(final_model_path)
    
    # 保存训练曲线
    plot_training_curves(
        all_episode_rewards,
        save_path=os.path.join(results_dir, 'training_rewards.png'),
        title='A2C Vectorized Training Rewards'
    )
    
    # 关闭环境
    for env in envs:
        env.close()
    logger.close()
    
    return agent, all_episode_rewards


if __name__ == '__main__':
    # 优化版训练配置 - 最大化利用 3050Ti 4GB
    agent, rewards = train(
        env_name='Acrobot-v1',
        num_envs=16,       # 16 个并行环境（最大化 GPU 利用率）
        num_steps=256,     # 每次收集 256 步（大批量）
        total_updates=500, # 共 500 次更新 = 204.8万 步
        
        hidden_dims=[512, 512],  # 更大的网络
        learning_rate=3e-4,
        entropy_coef=0.01,
    )
    
    print(f"\nTotal episodes completed: {len(rewards)}")
    print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
