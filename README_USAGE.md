# 工业控制离线强化学习项目

本项目基于赛题要求，实现了离线强化学习工业控制智能体，支持多种策略切换。

## 项目结构

```
IndustrialControl-main/
├── agent/
│   ├── agent.py          # 基础Agent类和MLP策略
│   ├── cql_agent.py      # CQL策略Agent
│   └── mlp_model.pth     # 预训练MLP模型
├── data/
│   └── data.csv          # 离线训练数据集
├── cql_algorithm.py      # CQL算法实现
├── train_cql.py          # CQL训练脚本
├── test_agent.py         # Agent测试脚本
├── evaluator.py          # 环境模拟器
├── run.py                # 主运行脚本
├── environment.yml       # Conda环境配置
└── README_USAGE.md       # 使用说明
```

## 快速开始

### 1. 环境配置

#### 使用Conda（推荐）
```bash
conda env create -f environment.yml
conda activate IndustrialControl
```

#### 使用pip
```bash
pip install torch torchvision numpy pandas gym matplotlib tensorboard
```

### 2. 运行方式

#### 方式一：使用主运行脚本（推荐）

完整流程（训练CQL + 测试两种策略）：
```bash
python run.py full
```

单独训练CQL：
```bash
python run.py train
```

测试MLP策略：
```bash
python run.py test --agent mlp
```

测试CQL策略：
```bash
python run.py test --agent cql
```

#### 方式二：分步执行

1. 训练CQL模型：
```bash
python train_cql.py --data_path data/data.csv --save_path agent/cql_model.pth --epochs 1000
```

2. 测试MLP策略：
```bash
python test_agent.py mlp
```

3. 测试CQL策略：
```bash
python test_agent.py cql
```

#### 方式三：使用环境变量

```bash
# 测试MLP策略
set AGENT_TYPE=mlp
python test_agent.py

# 测试CQL策略  
set AGENT_TYPE=cql
python test_agent.py
```

### 3. 策略切换

项目支持便捷的策略切换，有以下几种方式：

#### 方法1：使用工厂函数
```python
from agent.agent import create_agent

# 创建MLP策略
mlp_agent = create_agent("mlp")

# 创建CQL策略
cql_agent = create_agent("cql")
```

#### 方法2：使用环境变量
```python
import os
from agent.agent import get_default_agent

# 设置环境变量
os.environ["AGENT_TYPE"] = "cql"

# 获取对应策略
agent = get_default_agent()
```

#### 方法3：直接导入
```python
# 使用MLP策略
from agent.agent import PolicyAgent as MLPAgent

# 使用CQL策略
from agent.cql_agent import CQLPolicyAgent
```

## 算法说明

### MLP策略（原始）
- 基于预训练的3层全连接网络
- 维护30帧观测历史
- 输入：150维（5维观测 × 30帧）
- 输出：3维动作

### CQL策略（新增）
- 基于Conservative Q-Learning的离线强化学习
- 支持处理时间延迟、噪声、部分可观测性等挑战
- 网络结构：Actor-Critic架构
- 训练数据：10万个历史交互数据点

## 参数配置

### CQL训练参数
- `--epochs`: 训练轮数（默认1000）
- `--batch_size`: 批次大小（默认256）
- `--lr`: 学习率（默认3e-4）
- `--gamma`: 折扣因子（默认0.99）
- `--alpha`: CQL损失权重（默认1.0）

### 测试参数
- 评估轨迹数：100条
- 随机种子：固定为123
- 最大步数：200步

## 性能指标

项目以累计奖励值作为主要评价指标，通过多条轨迹的平均奖励评估策略性能。

## 文件说明

- `cql_algorithm.py`: 完整的CQL算法实现，包含Actor-Critic网络、经验回放等
- `cql_agent.py`: 基于CQL的策略Agent，继承BaseAgent接口
- `train_cql.py`: CQL训练脚本，支持命令行参数配置
- `run.py`: 统一的运行脚本，支持训练和测试的完整流程

## 提交格式

按照赛题要求，提交文件结构：
```
submission.zip
├── agent/
│   ├── agent.py      # 策略实现
│   └── cql_model.pth # CQL模型权重
├── evaluator.py      # 测试模拟器
└── test_agent.py     # 策略测试脚本
```

## 注意事项

1. 确保数据文件`data/data.csv`存在且格式正确
2. 训练前检查GPU/CPU设备可用性
3. CQL模型文件较大时可考虑ONNX格式压缩
4. 测试时确保模型文件路径正确

## 故障排除

1. **ImportError**: 检查依赖包是否安装完整
2. **CUDA错误**: 检查PyTorch CUDA版本兼容性
3. **模型加载失败**: 检查模型文件路径和格式
4. **数据加载错误**: 检查CSV文件格式和字段名称

## 联系方式

如有问题，请参考赛题规则或联系技术支持。
