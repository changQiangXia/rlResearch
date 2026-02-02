"""
各种缓冲区定义
包括 ReplayBuffer (DQN) 和 RolloutBuffer (A2C/PPO)
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    经验回放缓冲区 (Experience Replay Buffer)
    用于 DQN 等 Off-Policy 算法
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1, device: str = 'cpu'):
        """
        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            action_dim: 动作维度（通常为 1，表示离散动作索引）
            device: PyTorch 设备
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.position = 0
        self.size = 0
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一个状态
            done: 是否终止
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """
        随机采样一个 batch
        
        Args:
            batch_size: 批量大小
        
        Returns:
            states, actions, rewards, next_states, dones (Tensor)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """清空缓冲区"""
        self.position = 0
        self.size = 0


class RolloutBuffer:
    """
    轨迹缓冲区 (Rollout Buffer)
    用于 A2C, PPO 等 On-Policy 算法
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1, device: str = 'cpu'):
        """
        Args:
            capacity: 缓冲区容量（最大步数）
            state_dim: 状态维度
            action_dim: 动作维度
            device: PyTorch 设备
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def push(self, state, action, log_prob, reward, value, done):
        """
        存储一步数据
        
        Args:
            state: 状态
            action: 动作
            log_prob: 动作的 log 概率
            reward: 奖励
            value: 状态价值估计
            done: 是否终止
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        计算 Generalized Advantage Estimation (GAE)
        
        Args:
            next_value: 最后状态的估计价值
            gamma: 折扣因子
            gae_lambda: GAE 参数
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        # 从后向前计算 GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        self.advantages = advantages.tolist()
        self.returns = (advantages + np.array(self.values)).tolist()
    
    def get(self):
        """
        获取所有数据
        
        Returns:
            states, actions, log_probs, advantages, returns (Tensor)
        """
        return (
            torch.FloatTensor(np.array(self.states)).to(self.device),
            torch.LongTensor(np.array(self.actions)).to(self.device),
            torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            torch.FloatTensor(np.array(self.advantages)).to(self.device),
            torch.FloatTensor(np.array(self.returns)).to(self.device),
        )
    
    def get_batch(self, batch_size: int, shuffle: bool = True):
        """
        分批获取数据（用于 PPO 的多 epoch 训练）
        
        Args:
            batch_size: 批量大小
            shuffle: 是否打乱数据
        
        Yields:
            每个 batch 的数据
        """
        indices = np.arange(len(self.states))
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(np.array([self.states[i] for i in batch_indices])).to(self.device),
                torch.LongTensor(np.array([self.actions[i] for i in batch_indices])).to(self.device),
                torch.FloatTensor(np.array([self.log_probs[i] for i in batch_indices])).to(self.device),
                torch.FloatTensor(np.array([self.advantages[i] for i in batch_indices])).to(self.device),
                torch.FloatTensor(np.array([self.returns[i] for i in batch_indices])).to(self.device),
            )
    
    def __len__(self):
        return len(self.states)


class VectorizedRolloutBuffer:
    """
    向量化轨迹缓冲区
    用于多个并行环境的 PPO 训练
    """
    
    def __init__(self, num_steps: int, num_envs: int, state_dim: int, device: str = 'cpu'):
        """
        Args:
            num_steps: 每次收集的步数
            num_envs: 并行环境数量
            state_dim: 状态维度
            device: PyTorch 设备
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.device = device
        
        # 预分配内存
        self.states = np.zeros((num_steps, num_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        self.step = 0
    
    def push(self, state, action, log_prob, reward, value, done):
        """存储一步数据"""
        self.states[self.step] = state
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        
        self.step += 1
    
    def compute_gae(self, next_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        计算 GAE（向量化版本）
        
        Args:
            next_values: shape (num_envs,)
        """
        advantages = np.zeros_like(self.rewards)
        
        for env_id in range(self.num_envs):
            gae = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_value = next_values[env_id]
                else:
                    next_value = self.values[t + 1, env_id]
                
                delta = self.rewards[t, env_id] + gamma * next_value * (1 - self.dones[t, env_id]) - self.values[t, env_id]
                gae = delta + gamma * gae_lambda * (1 - self.dones[t, env_id]) * gae
                advantages[t, env_id] = gae
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get(self):
        """获取所有数据并展平"""
        # 展平 (num_steps, num_envs) -> (num_steps * num_envs, ...)
        return (
            torch.FloatTensor(self.states.reshape(-1, self.state_dim)).to(self.device),
            torch.LongTensor(self.actions.reshape(-1)).to(self.device),
            torch.FloatTensor(self.log_probs.reshape(-1)).to(self.device),
            torch.FloatTensor(self.advantages.reshape(-1)).to(self.device),
            torch.FloatTensor(self.returns.reshape(-1)).to(self.device),
        )
    
    def clear(self):
        """清空并重置"""
        self.step = 0
