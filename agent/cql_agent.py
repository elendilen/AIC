import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import BaseAgent


class CQLActor(nn.Module):
    """CQL策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(CQLActor, self).__init__()
        self.max_action = max_action
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class CQLPolicyAgent(BaseAgent):
    """
    基于CQL（Conservative Q-Learning）的策略Agent
    使用离线强化学习训练的策略网络
    """
    
    HIST_LEN = 30  # 保持与原始实现一致的历史长度
    
    def __init__(self, model_path: str = "cql_model.pth"):
        """
        初始化CQL策略Agent
        
        Args:
            model_path: CQL模型文件路径
        """
        super().__init__()
        
        # 根据赛题规则设置网络参数
        self.state_dim = 5 * self.HIST_LEN  # 5维观测 × 30帧历史 = 150维输入
        self.action_dim = 3  # 3维动作
        self.max_action = 1.0  # 动作范围[-1, 1]
        
        # 自动定位模型文件
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        # 初始化CQL策略网络
        self.policy = CQLActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            max_action=self.max_action
        )
        
        # 加载预训练的CQL模型
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'actor_state_dict' in checkpoint:
                # 从完整的CQL检查点加载Actor
                self.policy.load_state_dict(checkpoint['actor_state_dict'])
            else:
                # 直接加载策略网络状态字典
                self.policy.load_state_dict(checkpoint)
            print(f"Successfully loaded CQL model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: CQL model file {model_path} not found. Using randomly initialized policy.")
        except Exception as e:
            print(f"Warning: Failed to load CQL model: {e}. Using randomly initialized policy.")
        
        # 设置为评估模式
        self.policy.to(self.device)
        self.policy.eval()
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        使用CQL策略网络生成动作
        
        Args:
            obs: 当前观测（已拉平为一维）
            
        Returns:
            action: 生成的动作
        """
        # 构建历史观测序列（与原始实现保持一致）
        seq = self.obs_history + [obs]
        if len(seq) < self.HIST_LEN:
            # 不足时用第一帧填充到指定长度
            seq = [seq[0]] * (self.HIST_LEN - len(seq)) + seq
        else:
            seq = seq[-self.HIST_LEN:]
        
        # 倒序并拉平成一维（与原始实现保持一致）
        state = np.array(seq[::-1], dtype=np.float32).reshape(1, -1)
        
        # 转为Tensor并前向推理
        x = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            action = self.policy(x)
        
        # 返回numpy数组
        return action.squeeze(0).cpu().numpy()


# 为了方便切换，提供一个工厂函数
def create_agent(agent_type: str = "cql", **kwargs):
    """
    创建智能体的工厂函数，便于切换不同策略
    
    Args:
        agent_type: 智能体类型 ("cql", "mlp", 等)
        **kwargs: 额外参数
        
    Returns:
        agent: 对应的智能体实例
    """
    if agent_type.lower() == "cql":
        return CQLPolicyAgent(**kwargs)
    elif agent_type.lower() == "mlp":
        # 导入原始的MLP智能体
        from agent import PolicyAgent as MLPPolicyAgent
        return MLPPolicyAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# 兼容性：设置默认的PolicyAgent为CQL版本
PolicyAgent = CQLPolicyAgent


if __name__ == "__main__":
    # 测试CQL策略Agent
    agent = CQLPolicyAgent()
    dummy_obs = np.random.rand(5).astype(np.float32)  # 5维观测
    
    # 测试动作生成
    a1 = agent.act(dummy_obs)
    a2 = agent.act(dummy_obs * 2)
    
    print("CQL Agent Test:")
    print("Action at step 1:", a1)
    print("Action at step 2:", a2)
    print("Action shape:", a1.shape)
    print("Action range:", f"[{a1.min():.3f}, {a1.max():.3f}]")
    
    agent.reset()
    agent.close()
