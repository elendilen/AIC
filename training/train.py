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
    """训练配置"""
    
    def __init__(self):
        # 模型参数
        self.state_dim = 5
        self.action_dim = 3
        self.hidden_dim = 256
        self.seq_len = 10
        
        # 训练参数
        self.batch_size = 256
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0
        
        # 训练设置
        self.num_epochs = 1000
        self.eval_frequency = 50
        self.save_frequency = 100
        self.early_stopping_patience = 200
        
        # 数据参数
        self.noise_std = 0.01
        self.data_augmentation = True
        
        # 设备设置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 路径设置
        self.data_path = 'assets/data.csv'
        self.save_dir = 'checkpoints'
        self.log_dir = 'logs'


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
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--cql_alpha', type=float, default=1.0, help='CQL正则化系数')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig()
    
    # 从命令行参数更新配置
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.cql_alpha:
        config.cql_alpha = args.cql_alpha
    
    # 如果提供了配置文件，加载配置
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 保存当前配置
    config_save_path = os.path.join(config.save_dir, 'training_config.json')
    os.makedirs(config.save_dir, exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    print(f"训练配置保存到: {config_save_path}")

    # 生成一份config.json（供评估和环境使用）
    config_json = {
        "obs_init_mean": [-0.45, -0.40, -0.45, -0.48, -0.40],
        "target_state": [-0.477452, -0.406494, -0.436498, -0.456012, -0.378295],
        "noise_std": config.noise_std,
        "delay_steps": 3,
        "data_config": {
            "data_path": config.data_path
        }
    }
    config_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"评估用config.json已生成: {config_json_path}")

    # 开始训练
    agent = train_cql_agent(config, args.data_path)
    print("训练完成！模型已保存。")


if __name__ == '__main__':
    main()
