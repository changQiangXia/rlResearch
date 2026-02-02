# RL_Evolution: From Q-Learning to PPO

<h3 align="center">
  <a href="#chinese">🇨🇳 中文</a> | 
  <a href="#english">🇺🇸 English</a>
</h3>

---

<a name="chinese"></a>
## 🇨🇳 中文版

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Gym](https://img.shields.io/badge/Gym-0.21+-green.svg)](https://gym.openai.com/)

本项目是一个**从零开始实现**的强化学习教程，完整展示了从简单的 Q-Learning 到先进的 PPO 算法的演进过程。

### 🎯 项目特色

- **模块化设计**：环境交互、网络模型、缓冲区、算法逻辑完全解耦
- **完整演进**：4 个阶段，从表格型到深度强化学习
- **GPU 优化**：充分利用显卡进行并行训练
- **丰富的可视化**：每种算法都有专门的可视化方案

### 📁 项目结构

```
RL_Evolution/
├── common/                     # 通用组件
│   ├── networks.py             # PyTorch 网络
│   ├── buffer.py               # 缓冲区
│   └── logger.py               # 日志系统
├── stages/                     # 四个演进阶段
│   ├── stage1_tabular/         # Q-Learning
│   ├── stage2_dqn/             # DQN
│   ├── stage3_actor_critic/    # A2C
│   └── stage4_ppo/             # PPO
├── utils/                      # 工具函数
├── results/                    # 训练结果
├── requirements.txt            # 依赖列表
└── README.md
```

### 🔧 环境要求

- **Python**: 3.7+
- **PyTorch**: 1.13.1+
- **CUDA**: 11.7+ (可选，用于 GPU 加速)
- **操作系统**: Windows 10/11, Linux, macOS

**推荐配置**:
- GPU: NVIDIA GTX 1050Ti 或更高 (4GB+ 显存)
- RAM: 8GB+

### 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Stage 1: Q-Learning (FrozenLake-v1)
python stages/stage1_tabular/run.py

# Stage 2: DQN (CartPole-v1)
python stages/stage2_dqn/run.py

# Stage 3: A2C - 终极版 (Acrobot-v1)
python stages/stage3_actor_critic/run_ultimate.py

# Stage 4: PPO (Acrobot-v1) - 推荐
python stages/stage4_ppo/run.py
```

### 📊 算法演进对比

| 阶段 | 算法 | 环境 | 平均奖励 | 训练时间 |
|------|------|------|---------|---------|
| 1 | Q-Learning | FrozenLake | 100% 成功率 | <1分钟 |
| 2 | DQN | CartPole | ~250 | ~5分钟 |
| 3 | A2C | Acrobot | ~-220 | ~30分钟 |
| 4 | **PPO** | Acrobot | **~-80** | ~20分钟 |

### 🎨 可视化

每个阶段都会生成训练曲线和指标图：
- `training_rewards.png`: 奖励曲线
- `training_metrics.png`: 损失和指标曲线

查看 TensorBoard:
```bash
tensorboard --logdir=results/stage4/logs
```

### 🐛 常见问题

**PyTorch DLL 错误**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Box2D 安装失败**:
```bash
conda install -c conda-forge box2d-py
```

---

<a name="english"></a>
## 🇺🇸 English Version

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![Gym](https://img.shields.io/badge/Gym-0.21+-green.svg)](https://gym.openai.com/)

This project is a **from-scratch** reinforcement learning tutorial demonstrating the complete evolution from simple Q-Learning to the advanced PPO algorithm.

### 🎯 Features

- **Modular Design**: Environment interaction, network models, buffers, and algorithm logic are completely decoupled
- **Complete Evolution**: 4 stages, from tabular to deep reinforcement learning
- **GPU Optimized**: Fully utilizes GPU for parallel training
- **Rich Visualization**: Each algorithm has dedicated visualization

### 📁 Project Structure

```
RL_Evolution/
├── common/                     # Common components
│   ├── networks.py             # PyTorch networks
│   ├── buffer.py               # Replay and rollout buffers
│   └── logger.py               # Logging system
├── stages/                     # Four evolution stages
│   ├── stage1_tabular/         # Q-Learning
│   ├── stage2_dqn/             # DQN
│   ├── stage3_actor_critic/    # A2C
│   └── stage4_ppo/             # PPO
├── utils/                      # Utilities
├── results/                    # Training results
├── requirements.txt            # Dependencies
└── README.md
```

### 🔧 Requirements

- **Python**: 3.7+
- **PyTorch**: 1.13.1+
- **CUDA**: 11.7+ (optional, for GPU acceleration)
- **OS**: Windows 10/11, Linux, macOS

**Recommended**:
- GPU: NVIDIA GTX 1050Ti+ (4GB+ VRAM)
- RAM: 8GB+

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Stage 1: Q-Learning (FrozenLake-v1)
python stages/stage1_tabular/run.py

# Stage 2: DQN (CartPole-v1)
python stages/stage2_dqn/run.py

# Stage 3: A2C - Ultimate (Acrobot-v1)
python stages/stage3_actor_critic/run_ultimate.py

# Stage 4: PPO (Acrobot-v1) - Recommended
python stages/stage4_ppo/run.py
```

### 📊 Algorithm Comparison

| Stage | Algorithm | Environment | Avg Reward | Training Time |
|-------|-----------|-------------|------------|---------------|
| 1 | Q-Learning | FrozenLake | 100% success | <1 min |
| 2 | DQN | CartPole | ~250 | ~5 min |
| 3 | A2C | Acrobot | ~-220 | ~30 min |
| 4 | **PPO** | Acrobot | **~-80** | ~20 min |

### 🎨 Visualization

Each stage generates training curves and metric plots:
- `training_rewards.png`: Reward curves
- `training_metrics.png`: Loss and metric curves

View TensorBoard:
```bash
tensorboard --logdir=results/stage4/logs
```

### 🐛 Troubleshooting

**PyTorch DLL Error**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Box2D Installation Failed**:
```bash
conda install -c conda-forge box2d-py
```

---

<div align="center">
  <b>Happy Reinforcement Learning! 🚀</b><br>
  <b>强化学习快乐！🚀</b>
</div>
