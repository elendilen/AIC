import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    """
    简单三层全连接网络 (MLP)：
      - 输入层 → 隐藏层1 (ReLU)
      - 隐藏层1 → 隐藏层2 (ReLU)
      - 隐藏层2 → 输出层（线性）
    """

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        """
        参数:
          input_dim  (int):  输入特征维度
          hidden_dim1(int): 第一隐藏层单元数
          hidden_dim2(int): 第二隐藏层单元数
          output_dim (int):  输出维度（动作维度）
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算：
          x → ReLU(fc1) → ReLU(fc2) → fc3 → 返回
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)