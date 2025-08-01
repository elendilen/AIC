import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import argparse
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cql_agent import CQLAgent
from utils.data_utils import (
    load_offline_data, preprocess_data, create_dataloader,
    compute_dataset_statistics, save_processed_data
)



class TrainingConfig:
    """训练配置，全部参数从config.json读取，且支持命令行/自定义config_path覆盖"""
    def __init__(self, config_json_path=None, override_dict=None):
        # 1. 读取根目录config.json
        if config_json_path is None:
            config_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        assert os.path.exists(config_json_path), f"配置文件不存在: {config_json_path}"
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)

        # 2. 解析参数
        # 模型参数
        self.state_dim = config_dict.get('state_dim', 5)
        self.action_dim = config_dict.get('action_dim', 3)
        self.hidden_dim = config_dict.get('hidden_dim', 256)
        self.seq_len = config_dict.get('seq_len', 10)
        # obs_init_mean/target_state 只用于评估

        # 训练参数
        self.batch_size = config_dict.get('batch_size', 256)
        self.learning_rate = config_dict.get('learning_rate', 3e-4)
        self.gamma = config_dict.get('gamma', 0.99)
        self.tau = config_dict.get('tau', 0.005)
        self.cql_alpha = config_dict.get('cql_alpha', 1.0)
        self.num_epochs = config_dict.get('num_epochs', 120)
        self.eval_frequency = config_dict.get('eval_frequency', 50)
        self.save_frequency = config_dict.get('save_frequency', 100)
        self.early_stopping_patience = config_dict.get('early_stopping_patience', 200)

        # 数据参数
        self.noise_std = config_dict.get('noise_std', 0.013)
        self.data_augmentation = config_dict.get('data_augmentation', True)

        # 设备设置
        self.device = config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 路径设置
        # 优先data_config.data_path，否则data_path，否则默认
        self.data_path = config_dict.get('data_config', {}).get('data_path', config_dict.get('data_path', 'assets/data.csv'))
        self.save_dir = config_dict.get('save_dir', 'checkpoints')
        self.log_dir = config_dict.get('log_dir', 'logs')

        # 延迟补偿参数
        self.delay_steps = config_dict.get('delay_steps', 6)

        # 允许外部覆盖
        if override_dict:
            for k, v in override_dict.items():
                setattr(self, k, v)


def train_cql_agent(config: TrainingConfig, data_path: str = None) -> CQLAgent:
    """训练CQL代理"""
    
    print("="*50)
    print("开始训练离线强化学习代理")
    print("="*50)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 加载数据
    data_path = data_path or config.data_path
    print(f"1. 加载数据...")
    data = load_offline_data(data_path)
    
    # 数据预处理
    print(f"2. 数据预处理...")
    processed_data = preprocess_data(data, noise_std=config.noise_std)
    
    # 计算数据统计
    stats = compute_dataset_statistics(processed_data)
    print(f"3. 数据集统计:")
    print(f"   - 轨迹数量: {stats['trajectories']['count']}")
    print(f"   - 总步数: {stats['trajectories']['total_steps']}")
    print(f"   - 平均轨迹长度: {stats['trajectories']['avg_length']:.1f}")
    
    # 保存预处理数据
    processed_data_path = os.path.join(config.save_dir, 'processed_data.pkl')
    save_processed_data(processed_data, processed_data_path)
    
    # 创建数据加载器
    print(f"4. 创建数据加载器...")
    dataloader = create_dataloader(
        processed_data,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        shuffle=True
    )
    
    # 初始化代理
    print(f"5. 初始化CQL代理...")
    agent = CQLAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        lr=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
        cql_alpha=config.cql_alpha,
        device=config.device
    )
    
    print(f"6. 开始训练...")
    print(f"   - 设备: {config.device}")
    print(f"   - 批次大小: {config.batch_size}")
    print(f"   - 学习率: {config.learning_rate}")
    print(f"   - CQL Alpha: {config.cql_alpha}")
    
    # 训练循环
    training_history = {
        'epoch': [],
        'policy_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'cql_loss': [],
        'total_loss': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        epoch_losses = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'cql_loss': []
        }
        
        # 训练一个epoch
        agent.policy.train()
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到设备
            for key in batch:
                batch[key] = batch[key].to(config.device)
            
            # 更新网络
            loss_dict = agent.update(batch)
            
            # 记录损失
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
        
        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        total_loss = sum(avg_losses.values())
        
        # 记录训练历史
        training_history['epoch'].append(epoch)
        for key, value in avg_losses.items():
            training_history[key].append(value)
        training_history['total_loss'].append(total_loss)
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}/{config.num_epochs} | "
                  f"Total Loss: {total_loss:.4f} | "
                  f"Policy: {avg_losses['policy_loss']:.4f} | "
                  f"Q1: {avg_losses['q1_loss']:.4f} | "
                  f"Q2: {avg_losses['q2_loss']:.4f} | "
                  f"CQL: {avg_losses['cql_loss']:.4f}")
        
        # 早停检查
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(config.save_dir, 'best_model.pth')
            agent.save_model(best_model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.early_stopping_patience:
            print(f"早停触发，在epoch {epoch}")
            break
        
        # 定期保存
        if epoch % config.save_frequency == 0 and epoch > 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth')
            agent.save_model(checkpoint_path)
    
    print("训练完成!")
    
    # 保存最终模型
    final_model_path = os.path.join(config.save_dir, 'final_model.pth')
    agent.save_model(final_model_path)
    
    # 为提交准备策略模型
    submission_model_path = os.path.join('agent', 'model.pth')
    os.makedirs('agent', exist_ok=True)
    torch.save(agent.policy.state_dict(), submission_model_path)
    print(f"提交用模型保存到: {submission_model_path}")
    
    # 保存训练历史
    history_path = os.path.join(config.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(training_history, config.log_dir)
    
    return agent


def plot_training_curves(history: Dict[str, list], save_dir: str):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))
    
    # 总损失
    plt.subplot(2, 3, 1)
    plt.plot(history['epoch'], history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 策略损失
    plt.subplot(2, 3, 2)
    plt.plot(history['epoch'], history['policy_loss'])
    plt.title('Policy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Q1损失
    plt.subplot(2, 3, 3)
    plt.plot(history['epoch'], history['q1_loss'])
    plt.title('Q1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Q2损失
    plt.subplot(2, 3, 4)
    plt.plot(history['epoch'], history['q2_loss'])
    plt.title('Q2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # CQL损失
    plt.subplot(2, 3, 5)
    plt.plot(history['epoch'], history['cql_loss'])
    plt.title('CQL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 所有损失在一个图中
    plt.subplot(2, 3, 6)
    plt.plot(history['epoch'], history['policy_loss'], label='Policy')
    plt.plot(history['epoch'], history['q1_loss'], label='Q1')
    plt.plot(history['epoch'], history['q2_loss'], label='Q2')
    plt.plot(history['epoch'], history['cql_loss'], label='CQL')
    plt.title('All Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线保存到: {plot_path}")
    
    # 也保存PDF版本
    plot_path_pdf = os.path.join(save_dir, 'training_curves.pdf')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    
    plt.close()


def main():
    """主函数"""

    parser = argparse.ArgumentParser(description='训练离线强化学习代理')
    parser.add_argument('--data_path', type=str, default=None, help='数据文件路径')
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径（json）')
    parser.add_argument('--override', type=str, default=None, help='json字符串，额外参数覆盖')
    args = parser.parse_args()

    # 统一读取根目录config.json，允许--config_path覆盖，允许--override覆盖
    config_json_path = args.config_path if args.config_path else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    override_dict = json.loads(args.override) if args.override else None
    config = TrainingConfig(config_json_path=config_json_path, override_dict=override_dict)

    # 保存当前配置到训练目录
    config_save_path = os.path.join(config.save_dir, 'training_config.json')
    os.makedirs(config.save_dir, exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"训练配置保存到: {config_save_path}")

    # 开始训练
    agent = train_cql_agent(config, args.data_path)
    print("训练完成！模型已保存。")


if __name__ == '__main__':
    main()
