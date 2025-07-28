#!/usr/bin/env python3
"""
CQL训练脚本
用于训练基于Conservative Q-Learning的离线强化学习智能体
"""

import os
import sys
import argparse
from cql_algorithm import train_cql_agent


def main():
    parser = argparse.ArgumentParser(description='Train CQL Agent for Industrial Control')
    parser.add_argument('--data_path', type=str, default='data/data.csv',
                        help='Path to the offline dataset CSV file')
    parser.add_argument('--save_path', type=str, default='agent/cql_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Evaluation frequency (epochs)')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} does not exist!")
        print("Please make sure the offline dataset is available.")
        return 1
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    try:
        # 训练CQL智能体
        cql_agent = train_cql_agent(
            data_path=args.data_path,
            save_path=args.save_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            eval_freq=args.eval_freq
        )
        
        print(f"Training completed successfully!")
        print(f"Model saved to: {args.save_path}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
