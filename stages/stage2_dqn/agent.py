"""
DQN Agent (Deep Q-Network)
基于深度网络的值函数近似
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from common.networks import MLP
from common.buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent
    
    核心组件：
    1. Policy Network: 用于选择动作和计算当前 Q 值
    2. Target Network: 用于计算目标 Q 值，稳定训练
    3. Replay Buffer: 存储和采样经验
    
    更新公式：
        L = (Q(s,a) - [r + γ * max(Q_target(s',a')) * (1-done)])^2
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 网络隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减系数
            buffer_capacity: 回放缓冲区容量
            batch_size: 训练批次大小
            target_update_freq: Target Network 更新频率（步数）
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # 创建 Q-Networks
        self.policy_net = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation='relu',
        ).to(device)
        
        self.target_net = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation='relu',
        ).to(device)
        
        # 初始化 Target Network 与 Policy Network 相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target Network 只用于推理
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            device=device,
        )
        
        # 训练统计
        self.update_count = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        ε-贪婪策略选择动作
        
        Args:
            state: 当前状态，shape (state_dim,)
            epsilon: 探索率，若为 None 则使用当前的 self.epsilon
        
        Returns:
            选择的动作索引
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(self.action_dim)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return int(q_values.argmax(dim=1).item())
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """
        更新 Policy Network
        
        Returns:
            当前 batch 的 loss 值
        """
        # 缓冲区数据不足时不训练
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 当前 Q 值：Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # 目标 Q 值：r + γ * max(Q_target(s', a')) * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算 Loss (MSE)
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新 Target Network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._update_target_network()
        
        # 记录 loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def _update_target_network(self):
        """硬更新：将 Policy Network 的权重复制到 Target Network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
        }, filepath)
        print(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        print(f"Loaded model from {filepath}")


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent
    解决 DQN 的 Q 值过高估计问题
    
    区别：
    - DQN: target = r + γ * max(Q_target(s', a'))
    - Double DQN: target = r + γ * Q_target(s', argmax(Q_policy(s', a)))
    """
    
    def update(self) -> float:
        """使用 Double DQN 的更新方式"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: 用 Policy Network 选择动作，用 Target Network 评估
        with torch.no_grad():
            # Policy Network 选择最佳动作
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target Network 评估该动作的 Q 值
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算 Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self._update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
