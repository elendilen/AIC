import numpy as np
from typing import Dict, Any, List
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockEnvironment:
    """模拟工业控制环境"""
    
    def __init__(self, obs_dim=5, action_dim=3, max_steps=1000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.current_obs = None

        # 尝试从 config.json 读取观测均值
        self.obs_init_mean = None
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # 支持 config["obs_init_mean"] 或自动用训练集均值
                if "obs_init_mean" in config:
                    self.obs_init_mean = np.array(config["obs_init_mean"], dtype=np.float32)
                else:
                    # 尝试用 data.csv 的均值
                    data_path = config.get("data_config", {}).get("data_path", None)
                    if data_path and os.path.exists(os.path.join(os.path.dirname(__file__), data_path)):
                        import pandas as pd
                        df = pd.read_csv(os.path.join(os.path.dirname(__file__), data_path))
                        obs_cols = [c for c in df.columns if c.startswith("obs_")]
                        self.obs_init_mean = df[obs_cols].mean().values.astype(np.float32)
            except Exception as e:
                print(f"[警告] config.json读取观测均值失败: {e}")
                self.obs_init_mean = None
        if self.obs_init_mean is None:
            # 默认与训练数据分布接近
            self.obs_init_mean = np.array([-0.45, -0.40, -0.45, -0.48, -0.40], dtype=np.float32)

        # 环境参数
        self.target_state = np.array([0.5, -0.3, 0.0, 0.2, -0.1])  # 目标状态
        self.noise_std = 0.05  # 噪声标准差
        self.delay_steps = 3   # 控制延迟步数
        self.action_buffer = []

        # 系统动态参数
        self.A = np.random.uniform(-0.1, 0.1, (obs_dim, obs_dim))  # 状态转移矩阵
        self.B = np.random.uniform(-0.2, 0.2, (obs_dim, action_dim))  # 控制矩阵

        # 稳定性调整
        self.A *= 0.95  # 确保系统稳定
        np.fill_diagonal(self.A, 0.9)  # 主对角线
    
    def reset(self) -> np.ndarray:
        """重置环境，观测初始化分布与训练一致"""
        self.current_step = 0
        # 用均值加微小扰动初始化
        self.current_obs = self.obs_init_mean + np.random.normal(0, 0.01, self.obs_dim)
        self.action_buffer = [np.zeros(self.action_dim) for _ in range(self.delay_steps)]
        return self.current_obs.copy()
    
    def step(self, action: np.ndarray) -> tuple:
        """环境步进"""
        # 动作限制
        action = np.clip(action, -1.0, 1.0)
        
        # 添加到延迟缓冲区
        self.action_buffer.append(action.copy())
        effective_action = self.action_buffer.pop(0)  # 获取延迟后的动作
        
        # 状态更新
        next_obs = (
            np.dot(self.A, self.current_obs) + 
            np.dot(self.B, effective_action) +
            np.random.normal(0, self.noise_std, self.obs_dim)
        )
        
        # 状态限制
        next_obs = np.clip(next_obs, -1.0, 1.0)
        
        # 计算奖励
        reward = self._compute_reward(self.current_obs, effective_action, next_obs)
        
        # 更新状态
        self.current_obs = next_obs
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        return next_obs.copy(), reward, done, {}
    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> float:
        """计算奖励"""
        # 距离目标的距离奖励
        distance_to_target = np.linalg.norm(next_obs - self.target_state)
        distance_reward = -distance_to_target
        
        # 动作平滑性奖励
        action_penalty = -0.1 * np.linalg.norm(action)
        
        # 稳定性奖励
        state_change = np.linalg.norm(next_obs - obs)
        stability_penalty = -0.05 * state_change
        
        # 状态限制惩罚
        boundary_penalty = 0.0
        if np.any(np.abs(next_obs) > 0.9):
            boundary_penalty = -1.0
        
        total_reward = distance_reward + action_penalty + stability_penalty + boundary_penalty
        
        return total_reward


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
