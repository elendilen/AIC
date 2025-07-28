#!/usr/bin/env python3
"""
工业控制离线强化学习项目运行脚本
支持训练CQL模型和测试不同策略
"""

import os
import sys
import argparse
import subprocess


def run_training():
    """运行CQL训练"""
    print("Starting CQL training...")
    try:
        result = subprocess.run([
            sys.executable, "train_cql.py",
            "--data_path", "data/data.csv",
            "--save_path", "agent/cql_model.pth",
            "--epochs", "1000",
            "--batch_size", "256"
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("CQL training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True


def run_test(agent_type="mlp"):
    """运行测试"""
    print(f"Testing {agent_type.upper()} agent...")
    
    # 设置环境变量
    env = os.environ.copy()
    env["AGENT_TYPE"] = agent_type
    
    try:
        result = subprocess.run([
            sys.executable, "test_agent.py", agent_type
        ], env=env, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Testing failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Industrial Control Offline RL Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train CQL agent')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='Test agent')
    test_parser.add_argument('--agent', type=str, default='mlp', 
                            choices=['mlp', 'cql'],
                            help='Agent type to test')
    
    # 完整流程命令
    full_parser = subparsers.add_parser('full', help='Full pipeline: train CQL then test both agents')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        success = run_training()
        return 0 if success else 1
        
    elif args.command == 'test':
        success = run_test(args.agent)
        return 0 if success else 1
        
    elif args.command == 'full':
        print("=== Running Full Pipeline ===")
        
        # 1. 训练CQL
        print("\n1. Training CQL Agent...")
        if not run_training():
            return 1
        
        # 2. 测试MLP
        print("\n2. Testing MLP Agent...")
        if not run_test("mlp"):
            return 1
        
        # 3. 测试CQL
        print("\n3. Testing CQL Agent...")
        if not run_test("cql"):
            return 1
        
        print("\n=== Full Pipeline Completed ===")
        return 0
        
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
