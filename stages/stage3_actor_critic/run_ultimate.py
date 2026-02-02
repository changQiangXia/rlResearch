"""Ultimate A2C Training - Maximum Effort"""
import os
import sys
import numpy as np
import gym
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stages.stage3_actor_critic.agent_ultimate import UltimateA2CAgent
from common.logger import Logger
from utils.visualization import plot_training_curves, plot_metrics


def train(
    env_name='Acrobot-v1',
    total_episodes=5000,  # 5x more episodes
    max_steps=1000,       # Longer episodes
    hidden_dims=[1024, 1024],  # Bigger network
    learning_rate=3e-4,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.2,     # Higher entropy
    max_grad_norm=0.5,
    log_interval=50,
    save_interval=500,
    results_dir=None,
):
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'stage3_ultimate')
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"ULTIMATE A2C - Maximum Optimization")
    print(f"Device: {device}")
    print(f"Network: {hidden_dims}")
    print(f"Episodes: {total_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print("-" * 70)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = UltimateA2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        lr=learning_rate,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        device=device,
    )
    
    logger = Logger(log_dir=os.path.join(results_dir, 'logs'), use_tensorboard=True)
    
    episode_rewards = []
    best_reward = -np.inf
    
    print(f"{'Episode':>8} | {'AvgR100':>10} | {'Best':>10} | {'Entropy':>8} | {'VLoss':>10}")
    print("-" * 70)
    
    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        # Collect full episode trajectory
        trajectory = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'values': [], 'dones': []
        }
        
        for step in range(max_steps):
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
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Update observation normalization
        agent.update_obs_rms(np.array(trajectory['states']))
        
        # Compute returns using Monte Carlo (best for sparse rewards)
        with torch.no_grad():
            next_state_norm = agent.normalize_obs(state.reshape(1, -1))
            next_value = agent.network(torch.FloatTensor(next_state_norm).to(device))[1].item()
        
        returns, advantages = agent.compute_returns(
            np.array(trajectory['rewards']),
            np.array(trajectory['values']),
            np.array(trajectory['dones']),
            next_value
        )
        
        # Update network
        states = np.array([agent.normalize_obs(s) for s in trajectory['states']])
        loss_dict = agent.update(
            states,
            np.array(trajectory['actions']),
            np.array(trajectory['log_probs']),
            returns,
            advantages
        )
        
        # Logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"{episode:8d} | {avg_reward:10.2f} | {best_reward:10.2f} | {loss_dict['entropy']:8.4f} | {loss_dict['value_loss']:10.2f}")
            
            logger.log_scalar('train/avg_reward', avg_reward, episode)
            logger.log_scalar('train/best_reward', best_reward, episode)
            logger.log_scalar('train/entropy', loss_dict['entropy'], episode)
            logger.log_scalar('train/value_loss', loss_dict['value_loss'], episode)
        
        # Save checkpoints
        if episode % save_interval == 0:
            agent.save(os.path.join(results_dir, f'a2c_ultimate_ep{episode}.pth'))
    
    print("-" * 70)
    print("Training completed!")
    
    agent.save(os.path.join(results_dir, 'a2c_ultimate_final.pth'))
    
    # Plots
    plot_training_curves(episode_rewards, save_path=os.path.join(results_dir, 'training_rewards.png'))
    
    metrics = {
        'policy_loss': agent.policy_losses,
        'value_loss': agent.value_losses,
    }
    plot_metrics(metrics, save_path=os.path.join(results_dir, 'training_losses.png'))
    
    env.close()
    logger.close()
    
    return agent, episode_rewards


if __name__ == '__main__':
    agent, rewards = train(
        env_name='Acrobot-v1',
        total_episodes=5000,      # 5x more training
        max_steps=1000,           # Longer episodes
        hidden_dims=[1024, 1024], # 2x larger network
        learning_rate=3e-4,
        entropy_coef=0.2,         # High entropy for exploration
    )
    print(f"\nFinal Stats:")
    print(f"  Avg last 100: {np.mean(rewards[-100:]):.2f}")
    print(f"  Best: {max(rewards):.2f}")
