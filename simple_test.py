import pandas as pd
import numpy as np

print("测试加载 assets/data.csv...")

try:
    # 读取数据
    df = pd.read_csv('assets/data.csv')
    print(f"✓ 数据加载成功: {df.shape}")
    
    # 检查列名
    print(f"列名: {df.columns.tolist()}")
    
    # 提取观测和动作数据
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    action_cols = [col for col in df.columns if col.startswith('action_')]
    
    print(f"观测列: {obs_cols}")
    print(f"动作列: {action_cols}")
    
    if len(obs_cols) >= 5 and len(action_cols) >= 3:
        obs_data = df[sorted(obs_cols)[:5]].values
        action_data = df[sorted(action_cols)[:3]].values
        
        print(f"观测数据形状: {obs_data.shape}")
        print(f"动作数据形状: {action_data.shape}")
        print(f"观测范围: [{obs_data.min():.3f}, {obs_data.max():.3f}]")
        print(f"动作范围: [{action_data.min():.3f}, {action_data.max():.3f}]")
        
        # 检查轨迹数量
        unique_indices = df['index'].nunique()
        print(f"轨迹数量: {unique_indices}")
        
        print("\n✓ 数据格式检查通过!")
        print("现在可以开始训练了")
        
    else:
        print("❌ 观测或动作列数不足")
        
except Exception as e:
    print(f"❌ 错误: {e}")
