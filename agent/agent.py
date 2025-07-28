import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional
import os


class BaseAgent:
    """基础代理类"""
    def __init__(self):
        pass
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """获取动作的基础接口"""
        raise NotImplementedError


class PolicyNetwork(nn.Module):
    """策略网络，处理时间延迟、噪声和部分可观测性"""
    
    def __init__(self, obs_dim=5, action_dim=3, hidden_dim=256, seq_len=10):
        super(PolicyNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        # LSTM编码器处理时序信息和部分可观测性
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 噪声过滤层
        self.noise_filter = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 状态重构层
        self.state_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
    
    def forward(self, obs_sequence):
        """
        前向传播
        Args:
            obs_sequence: [batch_size, seq_len, obs_dim] 或 [seq_len, obs_dim]
        Returns:
            action: [batch_size, action_dim] 或 [action_dim]
        """
        if obs_sequence.dim() == 2:
            obs_sequence = obs_sequence.unsqueeze(0)  # 添加batch维度
            squeeze_output = True
        else:
            squeeze_output = False
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(obs_sequence)
        
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        
        # 噪声过滤
        filtered_state = self.noise_filter(last_hidden)
        
        # 状态重构
        reconstructed_state = self.state_reconstructor(filtered_state)
        
        # 生成动作
        action = self.policy_head(reconstructed_state)
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


class NoiseFilter:
    """噪声过滤器"""
    
    def __init__(self, obs_dim=5, alpha=0.3):
        self.obs_dim = obs_dim
        self.alpha = alpha  # 平滑因子
        self.history = deque(maxlen=5)
        self.filtered_obs = None
    
    def filter(self, obs):
        """应用指数移动平均滤波"""
        self.history.append(obs.copy())
        
        if self.filtered_obs is None:
            self.filtered_obs = obs.copy()
        else:
            # 指数移动平均
            self.filtered_obs = self.alpha * obs + (1 - self.alpha) * self.filtered_obs
        
        # 如果有足够历史数据，应用额外的平滑
        if len(self.history) >= 3:
            moving_avg = np.mean(list(self.history), axis=0)
            self.filtered_obs = 0.7 * self.filtered_obs + 0.3 * moving_avg
        
        return self.filtered_obs.copy()


class DelayCompensator:
    """时间延迟补偿器"""
    
    def __init__(self, max_delay=5):
        self.max_delay = max_delay
        self.action_history = deque(maxlen=max_delay)
        self.obs_history = deque(maxlen=max_delay)
    
    def compensate(self, obs, raw_action):
        """延迟补偿"""
        # 记录历史
        self.obs_history.append(obs.copy())
        
        if len(self.action_history) == 0:
            compensated_action = raw_action.copy()
        else:
            # 简单的延迟补偿：基于历史动作的影响进行调整
            recent_actions = list(self.action_history)
            action_trend = np.mean(recent_actions, axis=0) if len(recent_actions) > 1 else np.zeros(3)
            
            # 补偿系数
            compensation_factor = 0.1
            compensated_action = raw_action + compensation_factor * action_trend
            
            # 确保在有效范围内
            compensated_action = np.clip(compensated_action, -1.0, 1.0)
        
        self.action_history.append(raw_action.copy())
        return compensated_action


class PolicyAgent(BaseAgent):
    """主要的策略代理类"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络参数
        self.obs_dim = 5
        self.action_dim = 3
        self.seq_len = 10
        
        # 初始化策略网络
        self.policy = PolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            seq_len=self.seq_len
        ).to(self.device)
        
        # 加载模型权重
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.policy.load_state_dict(state_dict)
                print(f"模型权重加载成功: {model_path}")
            except Exception as e:
                print(f"模型权重加载失败: {e}")
                print("使用随机初始化的权重")
        else:
            print(f"模型文件不存在: {model_path}")
            print("使用随机初始化的权重")
        
        self.policy.eval()
        
        # 初始化辅助组件
        self.noise_filter = NoiseFilter(obs_dim=self.obs_dim)
        self.delay_compensator = DelayCompensator(max_delay=5)
        
        # 观测历史
        self.obs_history = deque(maxlen=self.seq_len)
        
        # 初始化历史缓冲区
        self._initialize_history()
    
    def _initialize_history(self):
        """初始化历史缓冲区"""
        # 用零向量填充初始历史
        zero_obs = np.zeros(self.obs_dim)
        for _ in range(self.seq_len):
            self.obs_history.append(zero_obs.copy())
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        获取动作
        Args:
            obs: 当前观测，形状为 [obs_dim,]
        Returns:
            action: 动作，形状为 [action_dim,]
        """
        try:
            # 确保输入格式正确
            obs = np.array(obs, dtype=np.float32)
            if obs.shape != (self.obs_dim,):
                raise ValueError(f"观测维度错误: 期望 {self.obs_dim}, 得到 {obs.shape}")
            
            # 噪声过滤
            filtered_obs = self.noise_filter.filter(obs)
            
            # 更新观测历史
            self.obs_history.append(filtered_obs)
            
            # 构建时序输入
            obs_sequence = np.array(list(self.obs_history))  # [seq_len, obs_dim]
            obs_tensor = torch.FloatTensor(obs_sequence).to(self.device)
            
            # 策略推理
            with torch.no_grad():
                raw_action = self.policy(obs_tensor)
                raw_action = raw_action.cpu().numpy()
            
            # 时间延迟补偿
            compensated_action = self.delay_compensator.compensate(filtered_obs, raw_action)
            
            # 确保动作在有效范围内
            compensated_action = np.clip(compensated_action, -1.0, 1.0)
            
            return compensated_action.astype(np.float32)
        
        except Exception as e:
            print(f"动作生成错误: {e}")
            # 返回安全的零动作
            return np.zeros(self.action_dim, dtype=np.float32)
