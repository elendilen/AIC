import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from collections import deque
import random
from typing import Tuple, List
import os


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward):
        """添加经验"""
        self.buffer.append((state, action, next_state, reward))
    
    def sample(self, batch_size: int):
        """随机采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(next_states),
            torch.FloatTensor(rewards)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """策略网络（Actor）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """价值网络（Critic）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1


class CQL:
    """Conservative Q-Learning 算法"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,  # CQL loss权重
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2
    ):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.total_it = 0
        
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=256):
        """训练网络"""
        self.total_it += 1
        
        # 采样批次数据
        state, action, next_state, reward = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # 选择下一步动作（加噪声进行正则化）
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # 计算目标Q值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.gamma * target_Q
        
        # 获取当前Q估计
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Conservative Q-Learning损失
        # 1. 常规TD损失
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 2. CQL保守性损失
        # 随机动作
        random_actions = torch.FloatTensor(
            batch_size, action.shape[-1]
        ).uniform_(-self.max_action, self.max_action).to(self.device)
        
        # 当前策略动作
        current_pi_actions = self.actor(state)
        
        # 下一状态策略动作
        next_pi_actions = self.actor(next_state)
        
        # 计算Q值
        q1_rand = self.critic.Q1(state, random_actions)
        q1_curr_actions = self.critic.Q1(state, current_pi_actions)
        q1_next_actions = self.critic.Q1(state, next_pi_actions)
        
        # CQL损失项
        cat_q1 = torch.cat([q1_rand, q1_curr_actions, q1_next_actions], 1)
        cql1_loss = torch.logsumexp(cat_q1, dim=1).mean() - current_Q1.mean()
        
        q2_rand = self.critic(state, random_actions)[1]
        q2_curr_actions = self.critic(state, current_pi_actions)[1]
        q2_next_actions = self.critic(state, next_pi_actions)[1]
        
        cat_q2 = torch.cat([q2_rand, q2_curr_actions, q2_next_actions], 1)
        cql2_loss = torch.logsumexp(cat_q2, dim=1).mean() - current_Q2.mean()
        
        # 总的Critic损失
        critic_loss = critic_loss + self.alpha * (cql1_loss + cql2_loss)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 计算Actor损失
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def load_offline_data(data_path: str) -> ReplayBuffer:
    """从CSV文件加载离线数据到回放缓冲区"""
    print(f"Loading offline data from {data_path}...")
    
    # 读取CSV数据
    df = pd.read_csv(data_path)
    
    # 数据预处理
    obs_cols = ['obs_1', 'obs_2', 'obs_3', 'obs_4', 'obs_5']
    action_cols = ['action_1', 'action_2', 'action_3']
    
    # 按轨迹索引分组
    replay_buffer = ReplayBuffer(capacity=len(df))
    
    # 按index分组处理每条轨迹
    for index in df['index'].unique():
        trajectory = df[df['index'] == index].reset_index(drop=True)
        
        for i in range(len(trajectory) - 1):
            state = trajectory.loc[i, obs_cols].values.astype(np.float32)
            action = trajectory.loc[i, action_cols].values.astype(np.float32)
            next_state = trajectory.loc[i + 1, obs_cols].values.astype(np.float32)
            reward = trajectory.loc[i, 'reward'].astype(np.float32)
            
            replay_buffer.push(state, action, next_state, reward)
    
    print(f"Loaded {len(replay_buffer)} transitions from {len(df['index'].unique())} trajectories")
    return replay_buffer


def train_cql_agent(
    data_path: str = "data/data.csv",
    save_path: str = "agent/cql_model.pth",
    num_epochs: int = 1000,
    batch_size: int = 256,
    eval_freq: int = 100
):
    """训练CQL智能体"""
    print("Training CQL Agent...")
    
    # 根据赛题规则设置参数
    state_dim = 5  # 观测维度
    action_dim = 3  # 动作维度
    max_action = 1.0  # 动作范围[-1, 1]
    
    # 初始化CQL算法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cql = CQL(state_dim, action_dim, max_action, device)
    
    # 加载离线数据
    replay_buffer = load_offline_data(data_path)
    
    # 训练循环
    print("Starting training...")
    for epoch in range(num_epochs):
        # 训练一步
        if len(replay_buffer) > batch_size:
            cql.train(replay_buffer, batch_size)
        
        # 定期输出训练进度
        if (epoch + 1) % eval_freq == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cql.save(save_path)
    print(f"Model saved to {save_path}")
    
    return cql


if __name__ == "__main__":
    # 训练CQL智能体
    train_cql_agent()
