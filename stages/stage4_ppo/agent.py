"""
PPO Agent (Proximal Policy Optimization)
使用 Clipped Surrogate Objective 和 GAE
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from common.networks import ActorCriticNetwork


class PPOAgent:
    """
    PPO Agent - 目前最稳定的策略梯度算法
    
    核心公式：
    L^{CLIP} = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    其中 r_t = π_new(a|s) / π_old(a|s)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,  # PPO clipping parameter
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Actor-Critic 网络
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation='relu',
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
        # 统计
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.clip_fractions = []
        self.explained_variances = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.network(state_tensor)
            probs = torch.distributions.Categorical(logits=action_logits)
            
            if deterministic:
                action = probs.probs.argmax(dim=-1)
            else:
                action = probs.sample()
            
            return (
                action.item(),
                probs.log_prob(action).item(),
                probs.entropy().item(),
                value.item(),
            )
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算 GAE (Generalized Advantage Estimation)"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, num_epochs=10, batch_size=64):
        """
        PPO 更新 - 多 epoch + mini-batch + clipping
        
        Args:
            states: (N, state_dim)
            actions: (N,)
            old_log_probs: (N,) - 旧策略的 log 概率
            advantages: (N,) - GAE 优势
            returns: (N,) - 回报
            num_epochs: 每个数据集的更新次数
            batch_size: mini-batch 大小
        """
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 记录更新前的价值，用于计算 explained variance
        with torch.no_grad():
            _, old_values = self.network(states)
            old_values = old_values.squeeze()
        
        # PPO 多 epoch 更新
        total_loss_dict = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'clip_fraction': 0.0,
        }
        
        for epoch in range(num_epochs):
            # 生成 mini-batch
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Mini-batch 数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_logits, values = self.network(batch_states)
                probs = torch.distributions.Categorical(logits=action_logits)
                log_probs = probs.log_prob(batch_actions)
                entropy = probs.entropy()
                
                # 计算比率 r_t
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped Surrogate Objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (MSE)
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy Loss (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 总 Loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 统计
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                
                total_loss_dict['policy_loss'] += policy_loss.item()
                total_loss_dict['value_loss'] += value_loss.item()
                total_loss_dict['entropy_loss'] += entropy_loss.item()
                total_loss_dict['clip_fraction'] += clip_fraction
        
        # 计算 explained variance (评估 Critic 质量)
        with torch.no_grad():
            _, new_values = self.network(states)
            new_values = new_values.squeeze()
            explained_variance = 1 - torch.var(returns - new_values) / torch.var(returns)
            explained_variance = explained_variance.item()
        
        # 平均统计
        num_updates = num_epochs * (len(states) // batch_size + 1)
        for key in total_loss_dict:
            total_loss_dict[key] /= num_updates
        
        # 记录
        self.policy_losses.append(total_loss_dict['policy_loss'])
        self.value_losses.append(total_loss_dict['value_loss'])
        self.entropy_losses.append(total_loss_dict['entropy_loss'])
        self.clip_fractions.append(total_loss_dict['clip_fraction'])
        self.explained_variances.append(explained_variance)
        
        return {
            'policy_loss': total_loss_dict['policy_loss'],
            'value_loss': total_loss_dict['value_loss'],
            'entropy': -total_loss_dict['entropy_loss'],
            'clip_fraction': total_loss_dict['clip_fraction'],
            'explained_variance': explained_variance,
        }
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Saved PPO model to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded PPO model from {filepath}")
