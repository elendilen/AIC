import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
from collections import deque
import pickle


class OfflineDataset(Dataset):
    """离线数据集类"""
    
    def __init__(self, data: Dict[str, np.ndarray], seq_len: int = 10):
        self.seq_len = seq_len
        self.data = data
        
        # 按轨迹索引组织数据
        self.trajectories = self._organize_by_trajectory()
        self.samples = self._create_samples()
        
        print(f"数据集初始化完成:")
        print(f"  - 轨迹数量: {len(self.trajectories)}")
        print(f"  - 样本数量: {len(self.samples)}")
        print(f"  - 序列长度: {self.seq_len}")
    
    def _organize_by_trajectory(self) -> Dict[int, Dict[str, np.ndarray]]:
        """按轨迹索引组织数据"""
        trajectories = {}
        unique_indices = np.unique(self.data['index'])
        
        for idx in unique_indices:
            mask = self.data['index'] == idx
            traj_data = {}
            for key, values in self.data.items():
                if key != 'index':
                    traj_data[key] = values[mask]
            trajectories[idx] = traj_data
        
        return trajectories
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        """创建训练样本"""
        samples = []
        
        for traj_idx, traj_data in self.trajectories.items():
            traj_len = len(traj_data['obs'])
            
            # 为每个轨迹创建序列样本
            for i in range(self.seq_len, traj_len):
                # 构建观测序列
                obs_seq = traj_data['obs'][i-self.seq_len:i]
                next_obs_seq = traj_data['next_obs'][i-self.seq_len:i]
                
                sample = {
                    'obs_sequence': obs_seq,
                    'next_obs_sequence': next_obs_seq,
                    'action': traj_data['action'][i-1],  # 当前动作
                    'reward': traj_data['reward'][i-1],  # 当前奖励
                    'done': 1.0 if i == traj_len - 1 else 0.0,  # 是否结束
                    'trajectory_id': traj_idx,
                    'step_id': i
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        return {
            'states': torch.FloatTensor(sample['obs_sequence']),
            'next_states': torch.FloatTensor(sample['next_obs_sequence']),
            'actions': torch.FloatTensor(sample['action']),
            'rewards': torch.FloatTensor([sample['reward']]),
            'dones': torch.FloatTensor([sample['done']])
        }


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, **kwargs):
        """添加经验"""
        self.buffer.append(kwargs)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样批次"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        batch = [self.buffer[i] for i in indices]
        
        # 组织批次数据
        keys = batch[0].keys()
        batched = {}
        
        for key in keys:
            if key in ['states', 'next_states']:
                # 序列数据
                batched[key] = torch.stack([torch.FloatTensor(item[key][-1]) for item in batch])
            else:
                # 标量数据
                batched[key] = torch.stack([torch.FloatTensor(item[key]) for item in batch])
        
        return batched
    
    def __len__(self):
        return len(self.buffer)


def load_offline_data(csv_path: str) -> Dict[str, np.ndarray]:
    """
    加载离线数据集
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        包含obs, action, next_obs, reward, index的字典
    """
    print(f"正在加载数据: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"数据加载成功，形状: {df.shape}")
        
        # 解析数据
        data = {}
        
        # 观测数据 (查找obs_1到obs_5列)
        obs_cols = [col for col in df.columns if col.startswith('obs_')]
        if len(obs_cols) >= 5:
            # 使用obs_1到obs_5列
            obs_cols_sorted = sorted(obs_cols)[:5]  # 取前5列
            data['obs'] = df[obs_cols_sorted].values
            print(f"使用观测列: {obs_cols_sorted}")
        elif 'obs' in df.columns:
            # 如果obs是字符串形式的数组，需要解析
            if isinstance(df['obs'].iloc[0], str):
                obs_data = []
                for obs_str in df['obs']:
                    obs_array = np.fromstring(obs_str.strip('[]'), sep=' ')
                    obs_data.append(obs_array)
                data['obs'] = np.array(obs_data)
            else:
                # 假设前5列是观测数据
                data['obs'] = df.iloc[:, :5].values
        
        # 动作数据 (查找action_1到action_3列)
        action_cols = [col for col in df.columns if col.startswith('action_')]
        if len(action_cols) >= 3:
            # 使用action_1到action_3列
            action_cols_sorted = sorted(action_cols)[:3]  # 取前3列
            data['action'] = df[action_cols_sorted].values
            print(f"使用动作列: {action_cols_sorted}")
        elif 'action' in df.columns:
            if isinstance(df['action'].iloc[0], str):
                action_data = []
                for action_str in df['action']:
                    action_array = np.fromstring(action_str.strip('[]'), sep=' ')
                    action_data.append(action_array)
                data['action'] = np.array(action_data)
            else:
                # 假设接下来3列是动作数据  
                start_col = len(obs_cols_sorted) if 'obs_cols_sorted' in locals() else 5
                data['action'] = df.iloc[:, start_col:start_col+3].values
        
        # 轨迹索引 - 需要在处理next_obs之前先处理index
        if 'index' in df.columns:
            data['index'] = df['index'].values
        elif 'trajectory_id' in df.columns:
            data['index'] = df['trajectory_id'].values
        elif 'traj_id' in df.columns:
            data['index'] = df['traj_id'].values
        else:
            # 如果没有轨迹索引，根据数据长度自动分割
            traj_length = 1000  # 假设每条轨迹1000步
            num_trajectories = len(df) // traj_length
            data['index'] = np.repeat(np.arange(num_trajectories), traj_length)[:len(df)]
            print(f"警告: 未找到轨迹索引，自动创建{num_trajectories}条轨迹")

        # 下一个观测 (生成next_obs数据)
        # 这个数据集似乎没有next_obs列，我们需要生成它
        if 'next_obs' in df.columns or any(col.startswith('next_obs') for col in df.columns):
            next_obs_cols = [col for col in df.columns if col.startswith('next_obs')]
            if len(next_obs_cols) >= 5:
                data['next_obs'] = df[sorted(next_obs_cols)[:5]].values
            else:
                # 使用obs数据向前偏移一步作为next_obs
                data['next_obs'] = np.roll(data['obs'], -1, axis=0)
        else:
            # 生成next_obs：按轨迹组织数据，每个轨迹内部向前偏移一步
            next_obs = []
            unique_indices = np.unique(data['index'])
            
            for idx in unique_indices:
                mask = data['index'] == idx
                traj_obs = data['obs'][mask]
                
                # 创建该轨迹的next_obs
                traj_next_obs = np.zeros_like(traj_obs)
                traj_next_obs[:-1] = traj_obs[1:]  # 向前偏移
                traj_next_obs[-1] = traj_obs[-1]   # 最后一步保持不变
                
                next_obs.append(traj_next_obs)
            
            data['next_obs'] = np.vstack(next_obs)
            print("自动生成next_obs数据")
        
        # 奖励数据
        if 'reward' in df.columns:
            data['reward'] = df['reward'].values
        else:
            # 如果没有奖励列，使用随机奖励作为占位符
            data['reward'] = np.random.randn(len(df))
            print("警告: 未找到奖励列，使用随机数据")
        
        # 轨迹索引
        if 'index' in df.columns:
            data['index'] = df['index'].values
        elif 'trajectory_id' in df.columns:
            data['index'] = df['trajectory_id'].values
        elif 'traj_id' in df.columns:
            data['index'] = df['traj_id'].values
        else:
            # 如果没有轨迹索引，根据数据长度自动分割
            traj_length = 1000  # 假设每条轨迹1000步
            num_trajectories = len(df) // traj_length
            data['index'] = np.repeat(np.arange(num_trajectories), traj_length)[:len(df)]
            print(f"警告: 未找到轨迹索引，自动创建{num_trajectories}条轨迹")
        
        # 数据类型转换和范围处理
        for key in ['obs', 'action', 'next_obs']:
            if key in data:
                data[key] = data[key].astype(np.float32)
                
                # 对于观测数据，如果不在[-1,1]范围内，进行归一化
                if key in ['obs', 'next_obs']:
                    data_min = data[key].min()
                    data_max = data[key].max()
                    if data_min < -1.0 or data_max > 1.0:
                        print(f"警告: {key}数据范围[{data_min:.3f}, {data_max:.3f}]超出[-1,1]，进行归一化")
                        # 归一化到[-1, 1]范围
                        data_range = data_max - data_min
                        data[key] = 2 * (data[key] - data_min) / data_range - 1
                        print(f"归一化后{key}范围: [{data[key].min():.3f}, {data[key].max():.3f}]")
                
                # 对于动作数据，确保在[-1, 1]范围内
                elif key == 'action':
                    if np.any(np.abs(data[key]) > 1.0):
                        print(f"警告: {key}数据超出[-1,1]范围，进行裁剪")
                        data[key] = np.clip(data[key], -1.0, 1.0)
        
        data['reward'] = data['reward'].astype(np.float32)
        data['index'] = data['index'].astype(np.int32)
        
        # 打印数据统计信息
        print("数据统计:")
        for key, values in data.items():
            if key != 'index':
                print(f"  {key}: shape={values.shape}, min={values.min():.3f}, max={values.max():.3f}")
            else:
                print(f"  {key}: shape={values.shape}, unique_values={len(np.unique(values))}")
        
        return data
    
    except Exception as e:
        print(f"数据加载失败: {e}")
        # 生成示例数据
        print("生成示例数据用于测试...")
        return generate_dummy_data()


def generate_dummy_data(num_trajectories: int = 100, steps_per_trajectory: int = 1000) -> Dict[str, np.ndarray]:
    """生成示例数据用于测试"""
    total_steps = num_trajectories * steps_per_trajectory
    
    data = {
        'obs': np.random.uniform(-1, 1, (total_steps, 5)).astype(np.float32),
        'action': np.random.uniform(-1, 1, (total_steps, 3)).astype(np.float32),
        'next_obs': np.random.uniform(-1, 1, (total_steps, 5)).astype(np.float32),
        'reward': np.random.randn(total_steps).astype(np.float32),
        'index': np.repeat(np.arange(num_trajectories), steps_per_trajectory).astype(np.int32)
    }
    
    print(f"生成示例数据: {num_trajectories}条轨迹, 每条{steps_per_trajectory}步")
    return data


def preprocess_data(data: Dict[str, np.ndarray], noise_std: float = 0.01) -> Dict[str, np.ndarray]:
    """
    数据预处理
    
    Args:
        data: 原始数据
        noise_std: 噪声标准差
    
    Returns:
        预处理后的数据
    """
    processed_data = data.copy()
    
    # 添加噪声增强
    if noise_std > 0:
        for key in ['obs', 'next_obs']:
            if key in processed_data:
                noise = np.random.normal(0, noise_std, processed_data[key].shape)
                processed_data[key] = processed_data[key] + noise
                processed_data[key] = np.clip(processed_data[key], -1.0, 1.0)
    
    # 数据平滑（简单的移动平均）
    window_size = 3
    for key in ['obs', 'next_obs']:
        if key in processed_data:
            # 按轨迹进行平滑
            unique_indices = np.unique(processed_data['index'])
            for idx in unique_indices:
                mask = processed_data['index'] == idx
                traj_data = processed_data[key][mask]
                
                # 应用移动平均
                for i in range(len(traj_data)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(traj_data), i + window_size // 2 + 1)
                    processed_data[key][mask][i] = np.mean(traj_data[start_idx:end_idx], axis=0)
    
    return processed_data


def save_processed_data(data: Dict[str, np.ndarray], filepath: str):
    """保存预处理后的数据"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据保存到: {filepath}")


def load_processed_data(filepath: str) -> Dict[str, np.ndarray]:
    """加载预处理后的数据"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"数据加载自: {filepath}")
    return data


def create_dataloader(
    data: Dict[str, np.ndarray],
    batch_size: int = 256,
    seq_len: int = 10,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """创建数据加载器"""
    dataset = OfflineDataset(data, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader


def compute_dataset_statistics(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """计算数据集统计信息"""
    stats = {}
    
    for key, values in data.items():
        if key != 'index':
            stats[key] = {
                'mean': np.mean(values, axis=0),
                'std': np.std(values, axis=0),
                'min': np.min(values, axis=0),
                'max': np.max(values, axis=0),
                'shape': values.shape
            }
    
    # 轨迹统计
    unique_indices = np.unique(data['index'])
    trajectory_lengths = []
    for idx in unique_indices:
        mask = data['index'] == idx
        trajectory_lengths.append(np.sum(mask))
    
    stats['trajectories'] = {
        'count': len(unique_indices),
        'lengths': trajectory_lengths,
        'avg_length': np.mean(trajectory_lengths),
        'total_steps': len(data['index'])
    }
    
    return stats
