import torch
import torch.nn as nn

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
