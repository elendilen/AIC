import numpy as np
from typing import Dict, Any, List
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockEnvironment:
    """模拟工业控制环境"""
    
    def __init__(self, obs_dim=5, action_dim=3, max_steps=1000, test_data_path=None):
        import pandas as pd
        # 优先用 test_data.csv，否则用 data.csv
        if test_data_path is None:
            test_data_path = os.path.join(os.path.dirname(__file__), 'assets', 'test_data.csv')
        if not os.path.exists(test_data_path):
            test_data_path = os.path.join(os.path.dirname(__file__), 'assets', 'data.csv')
        self.df = pd.read_csv(test_data_path)
        self.obs_cols = [c for c in self.df.columns if c.startswith('obs_')]
        self.action_cols = [c for c in self.df.columns if c.startswith('action_')]
        self.reward_col = 'reward'
        self.max_steps = min(max_steps, len(self.df))
        self.obs_dim = len(self.obs_cols)
        self.action_dim = len(self.action_cols)
        # 支持按轨迹分集
        self.has_index = 'index' in self.df.columns
        if self.has_index:
            self.unique_indices = self.df['index'].unique()
        self.traj_pointer = None  # 当前轨迹id
        self.pointer = 0  # 当前在轨迹内的步数

    def reset(self) -> np.ndarray:
        import numpy as np
        if self.has_index:
            # 随机采样一条轨迹
            self.traj_pointer = np.random.choice(self.unique_indices)
            traj_mask = self.df['index'] == self.traj_pointer
            self.traj_indices = self.df[traj_mask].index.to_list()
            self.pointer = 0
            obs = self.df.loc[self.traj_indices[self.pointer], self.obs_cols].values.astype(np.float32)
            self.traj_len = len(self.traj_indices)
        else:
            # 没有轨迹信息，随机采样起点
            self.traj_pointer = None
            if len(self.df) > self.max_steps:
                self.start_idx = np.random.randint(0, len(self.df) - self.max_steps)
            else:
                self.start_idx = 0
            self.pointer = 0
            obs = self.df.loc[self.start_idx + self.pointer, self.obs_cols].values.astype(np.float32)
            self.traj_len = min(self.max_steps, len(self.df) - self.start_idx)
        return obs

    def step(self, action: np.ndarray) -> tuple:
        self.pointer += 1
        if self.has_index:
            done = self.pointer >= self.traj_len
            if not done:
                idx = self.traj_indices[self.pointer]
                obs = self.df.loc[idx, self.obs_cols].values.astype(np.float32)
                reward = self.df.loc[self.traj_indices[self.pointer-1], self.reward_col]
            else:
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                reward = 0.0
        else:
            done = self.pointer >= self.traj_len
            if not done:
                obs = self.df.loc[self.start_idx + self.pointer, self.obs_cols].values.astype(np.float32)
                reward = self.df.loc[self.start_idx + self.pointer - 1, self.reward_col]
            else:
                obs = np.zeros(self.obs_dim, dtype=np.float32)
                reward = 0.0
        return obs, reward, done, {}


def evaluate_agent(agent, num_episodes=10, max_steps=1000, verbose=True) -> Dict[str, Any]:
    """
    评估代理性能
    
    Args:
        agent: 要评估的代理
        num_episodes: 评估轮数
        max_steps: 每轮最大步数
        verbose: 是否打印详细信息
    
    Returns:
        评估结果字典
    """
    env = MockEnvironment(max_steps=max_steps)
    
    episode_rewards = []
    episode_lengths = []
    
    if verbose:
        print(f"开始评估代理，共{num_episodes}轮...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 获取动作
            action = agent.get_action(obs)
            
            # 环境步进
            next_obs, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose and episode % max(1, num_episodes // 10) == 0:
            print(f"Episode {episode:3d}: Reward = {episode_reward:8.2f}, Length = {episode_length:4d}")
    
    # 计算统计信息
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    if verbose:
        print(f"\n评估结果:")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  奖励范围: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  平均长度: {results['mean_length']:.1f}")
    
    return results


class BaseAgent:
    """基础代理类（用于测试）"""
    def get_action(self, obs):
        return np.zeros(3)


def test_agent_interface():
    """测试代理接口"""
    print("测试代理接口...")
    
    try:
        from agent.agent import PolicyAgent
        agent = PolicyAgent()
        print("✓ PolicyAgent加载成功")
        
        # 测试动作生成
        test_obs = np.random.uniform(-1, 1, 5)
        action = agent.get_action(test_obs)
        
        print(f"  测试观测: {test_obs}")
        print(f"  生成动作: {action}")
        print(f"  动作形状: {action.shape}")
        print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 检查动作是否在有效范围内
        if action.shape == (3,) and np.all(np.abs(action) <= 1.0):
            print("✓ 动作格式和范围正确")
        else:
            print("✗ 动作格式或范围错误")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ 无法导入PolicyAgent: {e}")
        return False
    except Exception as e:
        print(f"✗ 代理测试失败: {e}")
        return False


def run_evaluation():
    """运行完整评估"""
    print("="*50)
    print("离线强化学习代理评估")
    print("="*50)
    
    # 测试代理接口
    if not test_agent_interface():
        print("代理接口测试失败，无法继续评估")
        return
    
    try:
        # 导入代理
        from agent.agent import PolicyAgent
        agent = PolicyAgent()
        
        # 运行评估
        print(f"\n开始性能评估...")
        results = evaluate_agent(agent, num_episodes=20, max_steps=1000)
        
        # 保存结果
        import json
        results_copy = results.copy()
        results_copy['episode_rewards'] = [float(r) for r in results_copy['episode_rewards']]
        results_copy['episode_lengths'] = [int(l) for l in results_copy['episode_lengths']]
        
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/evaluation_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        # 绘制结果
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            # 奖励曲线
            plt.subplot(1, 2, 1)
            plt.plot(results['episode_rewards'])
            plt.axhline(y=results['mean_reward'], color='r', linestyle='--', 
                       label=f'Mean: {results["mean_reward"]:.2f}')
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            
            # 奖励分布
            plt.subplot(1, 2, 2)
            plt.hist(results['episode_rewards'], bins=10, alpha=0.7)
            plt.axvline(x=results['mean_reward'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_reward"]:.2f}')
            plt.title('Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('evaluation_results/evaluation_plots.png', dpi=300)
            print("评估图表保存到: evaluation_results/evaluation_plots.png")
            plt.close()
            
        except ImportError:
            print("无法绘制图表（matplotlib未安装）")
        
        print(f"\n总结:")
        if results['mean_reward'] > -10:
            print("✓ 代理表现良好")
        elif results['mean_reward'] > -50:
            print("△ 代理表现一般，可能需要更多训练")
        else:
            print("✗ 代理表现较差，需要检查模型或训练过程")
        
    except Exception as e:
        print(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_evaluation()
