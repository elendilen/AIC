# 离线强化学习工业应用项目使用指南

## 项目概述

本项目是为"离线强化学习工业应用"竞赛设计的完整解决方案，基于Conservative Q-Learning (CQL)算法，专门处理工业控制中的三大关键挑战：

1. **时间延迟性** - 控制指令效果的滞后现象
2. **复杂噪声** - 传感器读数中的多种干扰
3. **部分可观测性** - 系统状态信息不完整的问题

## 快速开始

### 方式1: 使用启动脚本（推荐）

```bash
# 1. 完整流程（推荐新手）
python run.py --all

# 2. 分步执行
python run.py --setup          # 设置环境
python run.py --data           # 准备数据
python run.py --quick-train    # 快速训练
python run.py --test           # 测试代理
python run.py --submit         # 创建提交包
```

### 方式2: 手动执行

```bash
# 1. 安装依赖
pip install torch numpy pandas matplotlib scikit-learn

# 2. 训练模型
python training/train.py --data_path sample_data.csv --num_epochs 100

# 3. 测试代理
python test_agent.py

# 4. 评估性能
python evaluator.py
```

## 项目结构详解

```
offline_rl_project/
├── agent/                    # 代理模块
│   ├── agent.py             # 主要代理实现
│   └── model.pth            # 训练好的模型权重
├── models/                   # 算法模块
│   └── cql_agent.py         # CQL算法实现
├── training/                 # 训练模块
│   └── train.py             # 训练脚本
├── utils/                    # 工具模块
│   └── data_utils.py        # 数据处理工具
├── evaluator.py             # 评估脚本
├── test_agent.py            # 代理测试
├── run.py                   # 启动脚本
└── sample_data.csv          # 示例数据
```

## 核心技术特性

### 1. 算法架构

**PolicyNetwork 设计:**
- LSTM编码器：处理时序信息和部分可观测性
- 噪声过滤层：抑制传感器噪声影响
- 状态重构层：从部分观测重建完整状态
- 策略输出层：生成[-1,1]范围的控制动作

**CQL算法优势:**
- 防止Q值高估：避免离线数据外推错误
- 双Q网络：提高训练稳定性
- 保守性正则化：确保策略不偏离数据分布

### 2. 鲁棒性处理

**时间延迟补偿:**
```python
class DelayCompensator:
    def compensate(self, obs, raw_action):
        # 基于历史动作趋势进行补偿
        action_trend = np.mean(self.action_history, axis=0)
        compensated_action = raw_action + 0.1 * action_trend
        return np.clip(compensated_action, -1.0, 1.0)
```

**噪声过滤机制:**
```python
class NoiseFilter:
    def filter(self, obs):
        # 指数移动平均 + 移动窗口平滑
        filtered_obs = alpha * obs + (1-alpha) * self.filtered_obs
        if len(self.history) >= 3:
            moving_avg = np.mean(list(self.history), axis=0)
            filtered_obs = 0.7 * filtered_obs + 0.3 * moving_avg
        return filtered_obs
```

**状态重构网络:**
```python
# LSTM处理时序信息
lstm_out, _ = self.lstm(obs_sequence)
# 重构完整状态表示
reconstructed_state = self.state_reconstructor(lstm_out[:, -1, :])
```

### 3. 数据处理流程

**数据格式要求:**
- 观测空间：5维，范围[-1,1]
- 动作空间：3维，范围[-1,1]  
- 每个时间步：obs, action, next_obs, reward, index

**预处理步骤:**
1. 数据加载与验证
2. 噪声增强（可选）
3. 按轨迹组织数据
4. 创建时序样本
5. 批量数据加载

## 训练配置说明

### 默认超参数
```json
{
  "batch_size": 256,
  "learning_rate": 3e-4,
  "gamma": 0.99,
  "tau": 0.005,
  "cql_alpha": 1.0,
  "num_epochs": 1000,
  "seq_len": 10
}
```

### 超参数调优建议
- **学习率**: 1e-4到1e-3之间，建议3e-4
- **CQL Alpha**: 控制保守程度，0.1-10.0，建议1.0
- **序列长度**: 影响记忆能力，5-20，建议10
- **批次大小**: 影响训练稳定性，128-512，建议256

### 训练监控指标
- **Policy Loss**: 策略损失，应该逐步下降
- **Q1/Q2 Loss**: Q网络损失，反映价值函数学习
- **CQL Loss**: 保守性损失，防止外推错误
- **Total Loss**: 综合损失，评估整体收敛

## 提交准备

### 文件检查清单
- [ ] `agent/agent.py` - PolicyAgent类实现
- [ ] `agent/model.pth` - 训练好的模型权重
- [ ] `evaluator.py` - 测试环境（可选）
- [ ] `test_agent.py` - 测试脚本（可选）

### 压缩包要求
```bash
submission.zip
├── agent/
│   ├── agent.py
│   └── model.pth
├── evaluator.py    # 可选
└── test_agent.py   # 可选
```

### 代理接口规范
```python
class PolicyAgent(BaseAgent):
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        输入: obs - (5,) numpy数组，观测值
        输出: action - (3,) numpy数组，动作值，范围[-1,1]
        """
        return action
```

## 性能基准

### 评估环境
- 模拟工业控制环境
- 目标跟踪任务
- 噪声和延迟干扰
- 1000步episode长度

### 期望性能
- **平均奖励** > -10：良好性能
- **平均奖励** > -50：可接受性能  
- **平均奖励** < -50：需要改进

### 优化建议
1. **数据质量**: 确保数据清洁和格式正确
2. **网络调优**: 调整隐层维度和学习率
3. **正则化**: 适当的CQL Alpha值
4. **集成方法**: 训练多个模型取平均

## 常见问题解决

### Q: 训练不收敛怎么办？
A: 
1. 降低学习率（1e-4）
2. 增加CQL Alpha（2.0-5.0）
3. 检查数据质量和范围
4. 减小批次大小（128）

### Q: 模型输出异常怎么办？
A:
1. 检查输入数据范围[-1,1]
2. 验证网络初始化
3. 添加梯度裁剪
4. 确认激活函数（Tanh）

### Q: 内存不足怎么办？
A:
1. 减小批次大小
2. 降低序列长度
3. 使用梯度累积
4. 减少网络复杂度

### Q: 测试环境不匹配怎么办？
A:
1. 确保Python 3.7+
2. 安装正确的PyTorch版本
3. 检查CUDA兼容性
4. 使用CPU模式作为备选

## 进阶优化

### 模型压缩
```python
# 模型量化
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# ONNX导出
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 集成学习
```python
class EnsembleAgent:
    def __init__(self, model_paths):
        self.models = [load_model(path) for path in model_paths]
    
    def get_action(self, obs):
        actions = [model.get_action(obs) for model in self.models]
        return np.mean(actions, axis=0)
```

### 在线微调
```python
# 在线适应机制
def online_adaptation(agent, new_data, steps=100):
    for step in range(steps):
        batch = sample_batch(new_data)
        agent.update(batch, lr=1e-5)  # 小学习率微调
```

## 技术支持

如需帮助，请联系：
- 赛题交流QQ群：437743461
- 邮箱：songyi.gao@polixir.ai
- 报名官网：www.aicomp.cn

祝您比赛成功！🎉
