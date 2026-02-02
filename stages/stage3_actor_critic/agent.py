"""
A2C Agent (Advantage Actor-Critic)
同步的 Actor-Critic 算法
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from common.networks import ActorCriticNetwork
from common.buffer import RolloutBuffer


class A2CAgent:
    """
    A2C Agent (Advantage Actor-Critic)
    
    核心思想：
    - Actor (策略网络): 输出动作概率分布 π(a|s)
    - Critic (价值网络): 估计状态价值 V(s)
    - 使用优势函数 A(s,a) = Q(s,a) - V(s) 来减少方差
    
    Loss 组成：
    1. Policy Loss: -log(π(a|s)) * Advantage
    2. Value Loss: (V(s) - Return)^2
    3. Entropy Bonus: -H(π) (鼓励探索)
    """
    
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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度（离散）
            hidden_dims: 网络隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            value_coef: Value Loss 系数
            entropy_coef: Entropy Loss 系数
            max_grad_norm: 梯度裁剪阈值
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # 创建 Actor-Critic 网络
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation='relu',
        ).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 训练统计
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
    
    def select_action(self, state: np.ndarray):
        """
        选择动作
        
        Args:
            state: 当前状态
        
        Returns:
            action: 动作索引
            log_prob: 动作的 log 概率
            entropy: 策略熵
            value: 状态价值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            action_logits, value = self.network(state_tensor)
            probs = torch.distributions.Categorical(logits=action_logits)
            
            action = probs.sample()
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()
            
            return (
                action.item(),
                log_prob.item(),
                entropy.item(),
                value.item(),
            )
    
    def compute_gae(
        self,
        rewards: list,
        values: list,
        dones: list,
        next_value: float,
    ) -> tuple:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励列表
            values: 价值估计列表
            dones: 终止标志列表
            next_value: 最后状态的估计价值
        
        Returns:
            advantages: 优势函数列表
            returns: 回报列表
        """
        advantages = []
        gae = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            # TD Error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Returns = Advantages + Values
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """
        更新网络
        
        Args:
            states: 状态张量
            actions: 动作张量
            old_log_probs: 旧策略的 log 概率
            advantages: 优势函数
            returns: 回报
        
        Returns:
            损失字典
        """
        # 标准化优势（有助于稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 重新计算 log_probs 和 values
        action_logits, values = self.network(states)
        probs = torch.distributions.Categorical(logits=action_logits)
        
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        
        # Policy Loss: -log(π(a|s)) * Advantage
        policy_loss = -(log_probs * advantages).mean()
        
        # Value Loss: MSE(V(s), Return)
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy Loss: -H(π) (我们希望最大化熵，所以最小化 -H)
        entropy_loss = -entropy.mean()
        
        # 总 Loss
        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # 记录损失
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': loss.item(),
        }
        
        self.policy_losses.append(loss_dict['policy_loss'])
        self.value_losses.append(loss_dict['value_loss'])
        self.entropy_losses.append(loss_dict['entropy_loss'])
        self.total_losses.append(loss_dict['total_loss'])
        
        return loss_dict
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from {filepath}")
