"""A2C Agent with Adaptive Parameters"""
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
        self.epsilon = epsilon
    
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
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


class AdaptiveA2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 value_coef=0.5, entropy_coef=0.1, entropy_coef_min=0.01,
                 entropy_coef_max=0.5, target_entropy=0.5, max_grad_norm=0.5,
                 use_obs_norm=True, lr_decay=True, device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_max = entropy_coef_max
        self.target_entropy = target_entropy
        self.max_grad_norm = max_grad_norm
        self.use_obs_norm = use_obs_norm
        self.lr_decay = lr_decay
        self.device = device
        self.initial_lr = learning_rate
        
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd((state_dim,))
        
        self.network = self._create_network(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        
        self.policy_losses = []
        self.value_losses = []
        self.entropy_history = []
        self.entropy_coef_history = [entropy_coef]
        self.lr_history = [learning_rate]
        self.update_count = 0
    
    def _create_network(self, state_dim, action_dim, hidden_dims):
        class LayerNormActorCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super().__init__()
                layers = []
                prev_dim = state_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    prev_dim = hidden_dim
                self.feature = nn.Sequential(*layers)
                self.actor_head = nn.Linear(prev_dim, action_dim)
                self.critic_head = nn.Linear(prev_dim, 1)
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
            
            def forward(self, state):
                features = self.feature(state)
                return self.actor_head(features), self.critic_head(features)
        
        return LayerNormActorCritic(state_dim, action_dim, hidden_dims)
    
    def normalize_obs(self, obs):
        if self.use_obs_norm:
            return self.obs_rms.normalize(obs)
        return obs
    
    def update_obs_rms(self, obs):
        if self.use_obs_norm:
            self.obs_rms.update(obs)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = self.normalize_obs(state.reshape(1, -1))
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_logits, value = self.network(state_tensor)
            probs = torch.distributions.Categorical(logits=action_logits)
            
            if deterministic:
                action = probs.probs.argmax(dim=-1)
            else:
                action = probs.sample()
            
            return action.item(), probs.log_prob(action).item(), probs.entropy().item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def adapt_entropy_coef(self, current_entropy):
        # If entropy is too low, increase coef
        if current_entropy < self.target_entropy:
            self.entropy_coef = min(self.entropy_coef * 1.02, self.entropy_coef_max)
        else:
            self.entropy_coef = max(self.entropy_coef * 0.995, self.entropy_coef_min)
    
    def adapt_learning_rate(self, progress):
        if self.lr_decay:
            lr = self.initial_lr * (1.0 - progress)  # Linear decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return self.initial_lr
    
    def update(self, states, actions, old_log_probs, advantages, returns, progress=0.0):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        action_logits, values = self.network(states)
        probs = torch.distributions.Categorical(logits=action_logits)
        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()
        
        current_entropy = entropy.mean().item()
        self.entropy_history.append(current_entropy)
        
        # Adapt parameters
        self.adapt_entropy_coef(current_entropy)
        self.entropy_coef_history.append(self.entropy_coef)
        current_lr = self.adapt_learning_rate(progress)
        self.lr_history.append(current_lr)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.update_count += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': current_entropy,
            'entropy_coef': self.entropy_coef,
            'lr': current_lr,
        }
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_rms_mean': self.obs_rms.mean if self.use_obs_norm else None,
            'obs_rms_var': self.obs_rms.var if self.use_obs_norm else None,
        }
        torch.save(data, filepath)
        print(f"Saved to {filepath}")
