#!/usr/bin/env python3
"""
测试新数据格式的加载和处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import load_offline_data, compute_dataset_statistics

def test_data_loading():
    """测试数据加载"""
    print("="*50)
    print("测试 assets/data.csv 数据加载")
    print("="*50)
    
    # 加载数据
    data_path = 'assets/data.csv'
    print(f"加载数据文件: {data_path}")
    
    try:
        data = load_offline_data(data_path)
        
        print("\n数据加载成功!")
        print(f"数据键: {list(data.keys())}")
        
        # 验证数据格式
        for key, values in data.items():
            if key != 'index':
                print(f"{key}: shape={values.shape}, dtype={values.dtype}")
                print(f"  范围: [{values.min():.3f}, {values.max():.3f}]")
            else:
                print(f"{key}: shape={values.shape}, dtype={values.dtype}")
                print(f"  唯一值数量: {len(np.unique(values))}")
        
        # 计算统计信息
        print("\n计算数据集统计信息...")
        stats = compute_dataset_statistics(data)
        
        print(f"\n轨迹统计:")
        print(f"  轨迹数量: {stats['trajectories']['count']}")
        print(f"  总步数: {stats['trajectories']['total_steps']}")
        print(f"  平均轨迹长度: {stats['trajectories']['avg_length']:.1f}")
        
        # 检查数据质量
        print(f"\n数据质量检查:")
        
        # 检查观测范围
        obs_data = data['obs']
        if obs_data.min() < -1.0 or obs_data.max() > 1.0:
            print(f"  ⚠️  观测数据超出[-1,1]范围: [{obs_data.min():.3f}, {obs_data.max():.3f}]")
            print("      建议进行归一化处理")
        else:
            print(f"  ✓ 观测数据在有效范围内: [{obs_data.min():.3f}, {obs_data.max():.3f}]")
        
        # 检查动作范围
        action_data = data['action']
        if action_data.min() < -1.0 or action_data.max() > 1.0:
            print(f"  ⚠️  动作数据超出[-1,1]范围: [{action_data.min():.3f}, {action_data.max():.3f}]")
        else:
            print(f"  ✓ 动作数据在有效范围内: [{action_data.min():.3f}, {action_data.max():.3f}]")
        
        # 检查是否有NaN或无穷大值
        for key in ['obs', 'action', 'next_obs', 'reward']:
            values = data[key]
            nan_count = np.isnan(values).sum()
            inf_count = np.isinf(values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️  {key}包含无效值: NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"  ✓ {key}数据完整")
        
        print(f"\n数据加载测试完成!")
        return True
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import numpy as np
    success = test_data_loading()
    
    if success:
        print("\n🎉 数据格式兼容，可以开始训练!")
        print("\n下一步操作:")
        print("1. 运行训练: python training/train.py")
        print("2. 或使用启动脚本: python run.py --train")
    else:
        print("\n❌ 数据加载失败，请检查数据格式")
