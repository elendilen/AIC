# 离线强化学习工业应用项目

这是一个基于Conservative Q-Learning (CQL)的离线强化学习项目，专门为工业控制场景设计。项目解决了时间延迟、复杂噪声和部分可观测性等关键挑战。

## 项目结构

```
offline_rl_project/
├── agent/
│   ├── agent.py              # 主要代理实现（PolicyAgent类）
│   └── model.pth            # 训练好的模型权重
├── models/
│   └── cql_agent.py         # CQL算法实现
├── training/
│   └── train.py             # 训练脚本
├── utils/
│   └── data_utils.py        # 数据处理工具
├── evaluator.py             # 评估脚本
├── test_agent.py            # 代理测试脚本
├── requirements.txt         # 依赖包列表
├── config.json              # 配置文件
└── README.md               # 项目说明
```

## 核心特性

### 1. 算法设计
- **Conservative Q-Learning (CQL)**: 防止离线数据外推导致的Q值高估
- **LSTM编码器**: 处理时序信息和部分可观测性
- **噪声过滤**: 指数移动平均和移动窗口滤波
- **延迟补偿**: 基于历史动作的延迟补偿机制

### 2. 网络架构
- **策略网络**: LSTM + 全连接层，输出动作范围[-1,1]
- **双Q网络**: 减少Q值高估，提高训练稳定性
- **状态重构**: 处理部分可观测性问题

### 3. 鲁棒性处理
- **时间延迟**: 动作缓冲区和预测补偿
- **噪声过滤**: 多层过滤机制
- **状态估计**: LSTM记忆网络重构完整状态

## 快速开始

### 1. 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda install pytorch pandas numpy matplotlib scikit-learn
```

### 2. 数据准备
将离线数据集CSV文件放在项目根目录，文件应包含以下列：
- `obs`: 5维观测数据
- `action`: 3维动作数据  
- `next_obs`: 下一步观测数据
- `reward`: 奖励值
- `index`: 轨迹索引

### 3. 训练模型
```bash
# 基础训练
python training/train.py

# 自定义参数训练
python training/train.py --data_path data/your_data.csv --batch_size 512 --num_epochs 2000

# 使用配置文件
python training/train.py --config_path config.json
```

### 4. 测试代理
```bash
# 运行完整测试套件
python test_agent.py

# 性能评估
python evaluator.py
```

## 详细说明

### 数据格式要求
- **观测空间**: 5维，取值范围[-1,1]
- **动作空间**: 3维，取值范围[-1,1]  
- **轨迹结构**: 100条轨迹，每条1000步
- **数据字段**: obs, action, next_obs, reward, index

### 模型参数
```python
# 网络参数
state_dim = 5      # 观测维度
action_dim = 3     # 动作维度  
hidden_dim = 256   # 隐层维度
seq_len = 10       # 序列长度

# 训练参数
learning_rate = 3e-4
batch_size = 256
cql_alpha = 1.0    # CQL正则化系数
gamma = 0.99       # 折扣因子
```

### 关键算法组件

#### 1. 策略网络 (PolicyNetwork)
```python
# LSTM编码器处理时序信息
self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim//2, num_layers=2)

# 噪声过滤层
self.noise_filter = nn.Sequential(nn.Linear(...), nn.LayerNorm(...), nn.ReLU())

# 策略输出层
self.policy_head = nn.Sequential(..., nn.Tanh())  # 输出[-1,1]
```

#### 2. CQL损失函数
```python
# 标准Q-learning损失
q_loss = F.mse_loss(current_q, target_q)

# CQL正则化项
cql_loss = torch.logsumexp(q_random, dim=1).mean() - q_data.mean()

# 总损失
total_loss = q_loss + cql_alpha * cql_loss
```

#### 3. 噪声过滤器
```python
# 指数移动平均
filtered_obs = alpha * obs + (1 - alpha) * previous_filtered_obs

# 移动窗口平滑
if len(history) >= window_size:
    smoothed_obs = moving_average(history)
```

#### 4. 延迟补偿
```python
# 动作历史缓冲
action_buffer.append(current_action)
effective_action = action_buffer.pop(0)  # 延迟后的动作

# 补偿计算
compensation = compensation_factor * action_trend
compensated_action = raw_action + compensation
```

## 提交格式

为比赛提交准备的文件结构：
```
submission.zip
├── agent/
│   ├── agent.py           # PolicyAgent类实现
│   └── model.pth         # 模型权重
├── evaluator.py          # 测试环境
└── test_agent.py         # 测试脚本
```

### PolicyAgent接口
```python
class PolicyAgent(BaseAgent):
    def __init__(self):
        # 初始化代理
        pass
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        输入: obs (5,) 观测数组
        输出: action (3,) 动作数组，范围[-1,1]
        """
        return action
```

## 性能优化建议

### 1. 超参数调优
- 学习率: 尝试1e-4到1e-3
- CQL Alpha: 0.1到10.0之间调整
- 批次大小: 128, 256, 512
- 网络深度: 2-4层

### 2. 数据增强
- 添加高斯噪声: std=0.01-0.05
- 时序扰动: 时间偏移±3步
- 状态插值: 线性插值填充

### 3. 模型集成
- 训练多个模型
- 加权平均预测
- 不确定性估计

### 4. 部署优化
- 模型量化: INT8推理
- ONNX导出: 跨平台部署
- 批量推理: 提高吞吐量

## 故障排除

### 常见问题

1. **模型不收敛**
   - 降低学习率
   - 增加CQL Alpha
   - 检查数据质量

2. **动作输出异常**
   - 检查数据范围[-1,1]
   - 验证网络权重初始化
   - 确认梯度裁剪

3. **内存不足**
   - 减小批次大小
   - 降低序列长度
   - 使用梯度累积

4. **训练速度慢**
   - 使用GPU加速
   - 减少网络复杂度
   - 并行数据加载

### 调试技巧
```python
# 检查数据分布
print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

# 监控梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.6f}")

# 验证模型输出
with torch.no_grad():
    test_output = model(test_input)
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
```

## 参考资源

- [Conservative Q-Learning论文](https://arxiv.org/abs/2006.04779)
- [离线强化学习综述](https://arxiv.org/abs/2005.01643)
- [工业控制应用案例](https://arxiv.org/abs/2103.12144)

## 联系方式

如有问题或建议，请通过以下方式联系：
- 赛题交流QQ群：437743461
- 邮箱：songyi.gao@polixir.ai
- 报名官网：www.aicomp.cn

## 许可证

本项目遵循MIT许可证。详情请参阅LICENSE文件。
