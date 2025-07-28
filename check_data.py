import pandas as pd
import numpy as np

# 读取数据文件
df = pd.read_csv('assets/data.csv')

print("数据基本信息:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"轨迹数量 (unique index): {df['index'].nunique()}")
print(f"数据范围:")

# 检查观测数据范围
obs_cols = [col for col in df.columns if col.startswith('obs_')]
action_cols = [col for col in df.columns if col.startswith('action_')]

print(f"观测列: {obs_cols}")
print(f"动作列: {action_cols}")

if obs_cols:
    obs_data = df[obs_cols].values
    print(f"观测数据范围: [{obs_data.min():.3f}, {obs_data.max():.3f}]")

if action_cols:
    action_data = df[action_cols].values  
    print(f"动作数据范围: [{action_data.min():.3f}, {action_data.max():.3f}]")

print(f"奖励范围: [{df['reward'].min():.3f}, {df['reward'].max():.3f}]")

print("\n前5行数据:")
print(df.head())
