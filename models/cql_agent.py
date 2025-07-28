import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, Dict, Any
from collections import deque


class QNetwork(nn.Module):
    """Q网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """
        前向传播
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
        Returns:
            q_value: [batch_size, 1]
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class CQLAgent:
    """Conservative Q-Learning 代理"""
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        cql_alpha: float = 1.0,
        device: str = 'cuda'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化网络
        from agent.agent import PolicyNetwork
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_policy = copy.deepcopy(self.policy)
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        
        # 学习率调度器
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=100, gamma=0.95
        )
        self.q1_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q1_optimizer, step_size=100, gamma=0.95
        )
        self.q2_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q2_optimizer, step_size=100, gamma=0.95
        )
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'cql_loss': []
        }
    
    def select_action(self, state_sequence: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy(state_tensor)
            
            # 添加探索噪声
            if noise_scale > 0:
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.clamp(action, -1, 1)
        
        return action.cpu().numpy()[0]
    
    def compute_q_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算Q网络损失"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # 当前Q值
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        # 目标Q值
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Q损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        return q1_loss, q2_loss
    
    def compute_cql_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算CQL正则化损失"""
        states = batch['states']
        actions = batch['actions']
        
        # 随机动作Q值
        batch_size = states.shape[0]
        random_actions = torch.uniform(-1, 1, (batch_size, 10, self.action_dim)).to(self.device)
        
        # 计算随机动作的Q值
        q1_random = []
        q2_random = []
        for i in range(10):
            q1_random.append(self.q1(states, random_actions[:, i, :]))
            q2_random.append(self.q2(states, random_actions[:, i, :]))
        
        q1_random = torch.cat(q1_random, dim=1)
        q2_random = torch.cat(q2_random, dim=1)
        
        # 策略动作Q值
        policy_actions = self.policy(states)
        q1_policy = self.q1(states, policy_actions)
        q2_policy = self.q2(states, policy_actions)
        
        # 数据动作Q值
        q1_data = self.q1(states, actions)
        q2_data = self.q2(states, actions)
        
        # CQL损失
        cql1_loss = torch.logsumexp(q1_random, dim=1).mean() - q1_data.mean()
        cql2_loss = torch.logsumexp(q2_random, dim=1).mean() - q2_data.mean()
        
        cql_loss = (cql1_loss + cql2_loss) / 2
        
        return cql_loss
    
    def compute_policy_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算策略损失"""
        states = batch['states']
        
        # 策略动作
        policy_actions = self.policy(states)
        
        # Q值
        q1_policy = self.q1(states, policy_actions)
        q2_policy = self.q2(states, policy_actions)
        q_policy = torch.min(q1_policy, q2_policy)
        
        # 策略损失（最大化Q值）
        policy_loss = -q_policy.mean()
        
        return policy_loss
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络参数"""
        # 计算损失
        q1_loss, q2_loss = self.compute_q_loss(batch)
        cql_loss = self.compute_cql_loss(batch)
        policy_loss = self.compute_policy_loss(batch)
        
        # 更新Q网络
        total_q1_loss = q1_loss + self.cql_alpha * cql_loss
        total_q2_loss = q2_loss + self.cql_alpha * cql_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 软更新目标网络
        self.soft_update_target_networks()
        
        # 更新学习率
        self.policy_scheduler.step()
        self.q1_scheduler.step()
        self.q2_scheduler.step()
        
        # 记录统计信息
        stats = {
            'policy_loss': policy_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql_loss': cql_loss.item()
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def soft_update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        
        # 单独保存策略网络权重（用于提交）
        policy_path = filepath.replace('.pth', '_policy.pth')
        torch.save(self.policy.state_dict(), policy_path)
        print(f"模型保存到: {filepath}")
        print(f"策略网络保存到: {policy_path}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        
        # 更新目标网络
        self.target_policy = copy.deepcopy(self.policy)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        print(f"模型加载成功: {filepath}")
