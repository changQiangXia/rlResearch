"""A2C Adaptive Training Script"""
import os
import sys
import numpy as np
import gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage3_actor_critic.agent_adaptive import AdaptiveA2CAgent
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


def train(
    env_name='Acrobot-v1',
    total_episodes=1000,
    max_steps=500,
    hidden_dims=[256, 256],
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    value_coef=0.5,
    entropy_coef=0.1,  # Higher initial entropy
    target_entropy=0.5,
    max_grad_norm=0.5,
    log_interval=10,
    save_interval=100,
    results_dir=None,
):
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage3_adaptive')
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Environment: {env_name}")
    print(f"Adaptive entropy: initial={entropy_coef}, target={target_entropy}")
    print("-" * 70)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = AdaptiveA2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        target_entropy=target_entropy,
        max_grad_norm=max_grad_norm,
        use_obs_norm=True,
        lr_decay=True,
        device=device,
    )
    
    logger = Logger(log_dir=os.path.join(results_dir, 'logs'), use_tensorboard=True)
    
    episode_rewards = []
    episode_lengths = []
    
    print(f"{'Episode':>8} | {'Reward':>8} | {'Steps':>6} | {'Entropy':>8} | {'EntCoef':>8} | {'LR':>10}")
    print("-" * 70)
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        trajectory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}
        
        for step in range(max_steps):
            # Collect trajectory
            action, log_prob, entropy, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['log_probs'].append(log_prob)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['dones'].append(float(done))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Update observation normalization
        agent.update_obs_rms(np.array(trajectory['states']))
        
        # Compute GAE
        with torch.no_grad():
            next_state_normalized = agent.normalize_obs(state.reshape(1, -1))
            next_value = agent.network(torch.FloatTensor(next_state_normalized).to(device))[1].item()
        
        advantages, returns = agent.compute_gae(
            trajectory['rewards'], trajectory['values'], trajectory['dones'], next_value
        )
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([agent.normalize_obs(s) for s in trajectory['states']])).to(device)
        actions = torch.LongTensor(trajectory['actions']).to(device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Update with progress for adaptive LR
        progress = episode / total_episodes
        loss_dict = agent.update(states, actions, old_log_probs, advantages, returns, progress)
        
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"{episode:8d} | {avg_reward:8.2f} | {step+1:6d} | {loss_dict['entropy']:8.4f} | {loss_dict['entropy_coef']:8.4f} | {loss_dict['lr']:10.6f}")
            
            logger.log_scalar('train/reward', avg_reward, episode)
            logger.log_scalar('train/entropy', loss_dict['entropy'], episode)
            logger.log_scalar('train/entropy_coef', loss_dict['entropy_coef'], episode)
            logger.log_scalar('train/lr', loss_dict['lr'], episode)
        
        if episode % save_interval == 0:
            agent.save(os.path.join(results_dir, f'a2c_adaptive_ep{episode}.pth'))
    
    print("-" * 70)
    print("Training completed!")
    
    agent.save(os.path.join(results_dir, 'a2c_adaptive_final.pth'))
    
    # Save plots
    plot_training_curves(episode_rewards, save_path=os.path.join(results_dir, 'training_rewards.png'))
    
    metrics = {
        'policy_loss': agent.policy_losses,
        'value_loss': agent.value_losses,
        'entropy': agent.entropy_history,
        'entropy_coef': agent.entropy_coef_history,
        'learning_rate': agent.lr_history,
    }
    plot_metrics(metrics, save_path=os.path.join(results_dir, 'training_metrics.png'))
    
    env.close()
    logger.close()
    
    return agent, episode_rewards


if __name__ == '__main__':
    agent, rewards = train(
        env_name='Acrobot-v1',
        total_episodes=1000,
        entropy_coef=0.2,  # Higher initial
        target_entropy=0.8,  # Target entropy for Acrobot
        learning_rate=3e-4,
    )
    print(f"\nFinal avg reward (last 100): {np.mean(rewards[-100:]):.2f}")
