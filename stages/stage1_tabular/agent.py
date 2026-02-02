"""
Q-Learning Agent (表格型)
适用于离散状态、离散动作环境，如 FrozenLake
"""
import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Q-Learning Agent (Off-Policy TD Control)
    
    更新公式：
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        """
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率 α
            gamma: 折扣因子 γ
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        # 初始化 Q-Table 为零
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    
    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """
        ε-贪婪策略选择动作
        
        Args:
            state: 当前状态
            epsilon: 探索率
            
        Returns:
            选择的动作
        """
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.randint(self.n_actions)
        else:
            # 贪婪选择
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Q-Learning 更新
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            
        Returns:
            TD-Error
        """
        # 当前 Q 值
        current_q = self.q_table[state, action]
        
        # 计算目标 Q 值
        if done:
            # 终止状态，使用实际奖励
            target_q = reward
        else:
            # 非终止状态，使用下一个状态的最大 Q 值
            # 即使当前所有 Q 值为 0，这个公式也能正确传播奖励
            next_max_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * next_max_q
        
        # TD-Error
        td_error = target_q - current_q
        
        # 更新 Q-Table（即使 td_error 很小也会更新）
        new_q = current_q + self.lr * td_error
        self.q_table[state, action] = new_q
        
        return td_error
    
    def get_q_table(self) -> np.ndarray:
        """获取当前 Q-Table"""
        return self.q_table.copy()
    
    def save(self, filepath: str):
        """保存 Q-Table 到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'lr': self.lr,
                'gamma': self.gamma
            }, f)
        print(f"Saved Q-Table to {filepath}")
    
    def load(self, filepath: str):
        """从文件加载 Q-Table"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.n_states = data['n_states']
            self.n_actions = data['n_actions']
            self.lr = data['lr']
            self.gamma = data['gamma']
        print(f"Loaded Q-Table from {filepath}")


class SarsaAgent:
    """
    Sarsa Agent (On-Policy TD Control)
    
    更新公式：
        Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    
    与 Q-Learning 的区别：Sarsa 使用实际采取的下一个动作 a'，而非最大 Q 值
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
    
    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """ε-贪婪策略选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool
    ) -> float:
        """
        Sarsa 更新
        
        Args:
            next_action: 在下一个状态实际采取的动作（On-Policy）
        """
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.q_table[next_state, next_action]
        
        td_error = target_q - current_q
        self.q_table[state, action] += self.lr * td_error
        
        return td_error
    
    def get_q_table(self) -> np.ndarray:
        return self.q_table.copy()
