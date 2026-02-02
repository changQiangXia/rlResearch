"""
神经网络定义 (PyTorch)
包括 MLP 和 CNN 网络
"""
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)
    适用于状态为连续向量的环境（如 CartPole, LunarLander）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu',
        output_activation: str = None,
    ):
        """
        Args:
            input_dim: 输入维度（状态空间维度）
            hidden_dims: 隐藏层维度列表，如 [128, 128]
            output_dim: 输出维度（动作空间维度或 1）
            activation: 隐藏层激活函数 ('relu', 'tanh')
            output_activation: 输出层激活函数 (None, 'tanh', 'softmax')
        """
        super(MLP, self).__init__()
        
        # 构建层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier 初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态，shape (batch_size, input_dim) 或 (input_dim,)
        
        Returns:
            输出，shape (batch_size, output_dim) 或 (output_dim,)
        """
        return self.network(x)


class DuelingMLP(nn.Module):
    """
    Dueling DQN 网络
    将 Q 值分解为状态价值 V(s) 和优势函数 A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'relu',
    ):
        super(DuelingMLP, self).__init__()
        
        # 共享的特征提取层
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature = nn.Sequential(*feature_layers)
        self.feature_dim = hidden_dims[-1] if len(hidden_dims) > 0 else prev_dim
        
        # 价值分支 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)
        )
        
        # 优势分支 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 共享网络
    同时输出策略 (动作概率) 和价值估计
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = 'relu',
    ):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享的特征提取层
        feature_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature = nn.Sequential(*feature_layers)
        
        # Actor 头：输出动作概率
        self.actor_head = nn.Linear(prev_dim, action_dim)
        
        # Critic 头：输出状态价值
        self.critic_head = nn.Linear(prev_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor):
        """
        前向传播
        
        Returns:
            action_logits: 动作 logits (用于 softmax)
            value: 状态价值
        """
        features = self.feature(state)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        features = self.feature(state)
        return self.critic_head(features)
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """
        获取动作分布、价值，以及可选的动作 log_prob 和熵
        
        Args:
            state: 状态
            action: 如果提供，计算该动作的 log_prob 和熵
        
        Returns:
            action: 采样的动作
            log_prob: 动作的 log 概率
            entropy: 策略熵
            value: 状态价值
        """
        action_logits, value = self.forward(state)
        probs = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化 (Orthogonal Initialization)
    在 PPO 中常用，有助于稳定训练
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
