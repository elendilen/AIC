import torch
import torch.nn as nn
from network_type.policy_network import PolicyNetwork
from network_type.mlp import Mlp
import numpy as np
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional
import os


class BaseAgent:
    """基础代理类"""
    def __init__(self, seed: int = None):
        """
        参数:
          seed (int, optional): 随机种子；如果不为 None，将同时设置 Python、NumPy、PyTorch 的随机性。
        """
        if seed is not None:
            self.seed(seed)

        # 自动选择运算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 用于记录整个 episode 内的观测和动作（可用于调试或训练数据收集）
        self.obs_history = []
        self.act_history = []

    def seed(self, seed: int = 123) -> None:
        """设置随机种子，保证结果可复现。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        """
        在每个 episode 开始时由外部调用，
        清空历史缓存。
        """
        self.obs_history.clear()
        self.act_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        统一的动作生成流程：
          1. 将观测拉平成一维向量
          2. 调用子类的 get_action() 得到原始动作
          3. 将动作限制到 [-1, 1]
          4. 缓存本次观测和动作
          5. 返回动作
        """
        obs = obs.reshape(-1).astype(np.float32)
        action = self.get_action(obs)
        action = np.clip(action, -1.0, 1.0).reshape(-1).astype(np.float32)

        self.obs_history.append(obs)
        self.act_history.append(action)

        return action

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        子类必须实现：根据当前观测 obs 计算动作。
        输入:
          obs (np.ndarray): 一维观测向量
        输出:
          action (np.ndarray): 一维动作向量
        """
        ...

    def close(self) -> None:
        """可选：释放资源（如环境连接、文件句柄等）。"""
        pass

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

if __name__ == "__main__":
    agent = PolicyAgent()
    dummy_obs = np.random.rand(150).astype(np.float32)

    a1 = agent.act(dummy_obs)
    a2 = agent.act(dummy_obs * 2)

    print("Action at step 1:", a1)
    print("Action at step 2:", a2)

    agent.reset()
    agent.close()