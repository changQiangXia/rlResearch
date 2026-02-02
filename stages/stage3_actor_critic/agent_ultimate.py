"""Ultimate A2C - Maximum Optimization"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class UltimateActorCritic(nn.Module):
    """Large network with orthogonal init"""
    def __init__(self, state_dim, action_dim, hidden_dims=[1024, 1024]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        self.actor = nn.Linear(prev_dim, action_dim)
        self.critic = nn.Linear(prev_dim, 1)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        features = self.feature(x)
        return self.actor(features), self.critic(features)


class UltimateA2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=[1024, 1024],
                 lr=3e-4, gamma=0.99, value_coef=0.5, entropy_coef=0.1,
                 max_grad_norm=0.5, device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.network = UltimateActorCritic(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Normalizers
        self.obs_rms = RunningMeanStd((state_dim,))
        
        # Stats
        self.policy_losses = []
        self.value_losses = []
    
    def normalize_obs(self, obs):
        return self.obs_rms.normalize(obs)
    
    def update_obs_rms(self, obs):
        self.obs_rms.update(obs)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = self.normalize_obs(state.reshape(1, -1))
            state_t = torch.FloatTensor(state).to(self.device)
            logits, value = self.network(state_t)
            probs = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = probs.probs.argmax(dim=-1)
            else:
                action = probs.sample()
            
            return action.item(), probs.log_prob(action).item(), probs.entropy().item(), value.item()
    
    def compute_returns(self, rewards, values, dones, next_value):
        """Monte Carlo returns - most accurate for sparse rewards"""
        returns = np.zeros_like(rewards)
        running_return = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        advantages = returns - values
        return returns, advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Forward
        logits, values = self.network(states)
        probs = torch.distributions.Categorical(logits=logits)
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        
        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
        }
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_mean': self.obs_rms.mean,
            'obs_var': self.obs_rms.var,
        }, filepath)
