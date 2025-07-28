import torch
import torch.nn as nn
from agent.agent import PolicyNetwork
import os

# 创建模型
model = PolicyNetwork(obs_dim=5, action_dim=3, hidden_dim=256)

# 保存初始权重
os.makedirs('agent', exist_ok=True)
torch.save(model.state_dict(), 'agent/model.pth')
print('初始模型权重已保存到: agent/model.pth')
print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
