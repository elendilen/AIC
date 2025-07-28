#!/usr/bin/env python3
"""
快速启动脚本
用于快速训练和测试离线强化学习代理
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败: {e}")
        return False

def setup_environment():
    """设置环境"""
    print("设置Python环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("错误: 需要Python 3.7或更高版本")
        return False
    
    print(f"✓ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 安装依赖
    if not run_command("pip install -r requirements.txt", "安装依赖包"):
        print("尝试使用conda安装...")
        packages = ["pytorch", "numpy", "pandas", "matplotlib", "scikit-learn", "tqdm"]
        cmd = f"conda install -y {' '.join(packages)}"
        if not run_command(cmd, "使用conda安装依赖"):
            return False
    
    return True

def download_sample_data():
    """下载或生成示例数据"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_data_path = data_dir / "sample_data.csv"
    
    if not sample_data_path.exists():
        print("生成示例数据...")
        
        # 生成示例数据
        script = """
import numpy as np
import pandas as pd
from pathlib import Path

# 生成示例离线数据
num_trajectories = 100
steps_per_trajectory = 1000
total_steps = num_trajectories * steps_per_trajectory

data = {
    'obs_0': np.random.uniform(-1, 1, total_steps),
    'obs_1': np.random.uniform(-1, 1, total_steps),
    'obs_2': np.random.uniform(-1, 1, total_steps),
    'obs_3': np.random.uniform(-1, 1, total_steps),
    'obs_4': np.random.uniform(-1, 1, total_steps),
    'action_0': np.random.uniform(-1, 1, total_steps),
    'action_1': np.random.uniform(-1, 1, total_steps),
    'action_2': np.random.uniform(-1, 1, total_steps),
    'reward': np.random.randn(total_steps),
    'index': np.repeat(np.arange(num_trajectories), steps_per_trajectory)
}

# 生成next_obs (简单的状态转移)
for i in range(5):
    next_obs = np.roll(data[f'obs_{i}'], -1)
    next_obs[-1] = np.random.uniform(-1, 1)  # 最后一个值
    data[f'next_obs_{i}'] = next_obs

df = pd.DataFrame(data)
df.to_csv('data/sample_data.csv', index=False)
print(f"示例数据已生成: data/sample_data.csv ({len(df)} 行)")
"""
        
        with open("generate_data.py", "w") as f:
            f.write(script)
        
        if run_command("python generate_data.py", "生成示例数据"):
            os.remove("generate_data.py")
            print(f"✓ 示例数据生成完成: {sample_data_path}")
            return str(sample_data_path)
        else:
            return None
    else:
        print(f"✓ 使用现有数据: {sample_data_path}")
        return str(sample_data_path)

def train_model(data_path=None, quick=False):
    """训练模型"""
    if quick:
        # 快速训练模式
        cmd = f"python training/train.py --num_epochs 100 --batch_size 128"
    else:
        # 完整训练模式
        cmd = f"python training/train.py --num_epochs 1000 --batch_size 256"
    
    if data_path:
        cmd += f" --data_path {data_path}"
    
    return run_command(cmd, "训练模型")

def test_agent():
    """测试代理"""
    return run_command("python test_agent.py", "测试代理")

def evaluate_agent():
    """评估代理"""
    return run_command("python evaluator.py", "评估代理性能")

def create_submission():
    """创建提交包"""
    print("创建提交包...")
    
    # 检查必要文件
    required_files = [
        "agent/agent.py",
        "agent/model.pth"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"错误: 缺少必要文件: {missing_files}")
        return False
    
    # 创建提交目录
    submission_dir = Path("submission")
    submission_dir.mkdir(exist_ok=True)
    
    # 复制文件
    import shutil
    
    # 复制agent目录
    agent_dest = submission_dir / "agent"
    if agent_dest.exists():
        shutil.rmtree(agent_dest)
    shutil.copytree("agent", agent_dest)
    
    # 复制其他文件
    other_files = ["evaluator.py", "test_agent.py"]
    for file_name in other_files:
        if Path(file_name).exists():
            shutil.copy(file_name, submission_dir / file_name)
    
    # 创建压缩包
    if run_command(f"cd submission && zip -r ../submission.zip .", "创建ZIP压缩包"):
        print("✓ 提交包已创建: submission.zip")
        return True
    else:
        # 尝试使用Python创建压缩包
        try:
            import zipfile
            with zipfile.ZipFile("submission.zip", "w") as zipf:
                for root, dirs, files in os.walk(submission_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(submission_dir)
                        zipf.write(file_path, arcname)
            print("✓ 提交包已创建: submission.zip")
            return True
        except Exception as e:
            print(f"创建压缩包失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="离线强化学习项目快速启动脚本")
    parser.add_argument("--setup", action="store_true", help="设置环境")
    parser.add_argument("--data", action="store_true", help="准备示例数据")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--quick-train", action="store_true", help="快速训练模式")
    parser.add_argument("--test", action="store_true", help="测试代理")
    parser.add_argument("--eval", action="store_true", help="评估代理")
    parser.add_argument("--submit", action="store_true", help="创建提交包")
    parser.add_argument("--all", action="store_true", help="执行完整流程")
    parser.add_argument("--data-path", type=str, help="指定数据文件路径")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # 如果没有指定参数，显示帮助
        parser.print_help()
        return
    
    print("离线强化学习项目启动脚本")
    print(f"工作目录: {os.getcwd()}")
    
    success = True
    data_path = args.data_path
    
    if args.setup or args.all:
        success &= setup_environment()
    
    if (args.data or args.all) and not data_path:
        data_path = download_sample_data()
        success &= (data_path is not None)
    
    if args.train or args.all:
        success &= train_model(data_path, quick=False)
    elif args.quick_train:
        success &= train_model(data_path, quick=True)
    
    if args.test or args.all:
        success &= test_agent()
    
    if args.eval or args.all:
        success &= evaluate_agent()
    
    if args.submit or args.all:
        success &= create_submission()
    
    if success:
        print("\n🎉 所有操作成功完成！")
        if args.all or args.submit:
            print("\n📦 提交文件已准备就绪:")
            print("   - submission.zip (上传到比赛平台)")
            print("   - agent/model.pth (模型权重)")
    else:
        print("\n❌ 部分操作失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()
