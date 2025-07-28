#!/usr/bin/env python3
"""
测试代理脚本
用于验证PolicyAgent类是否正确实现
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent.agent import PolicyAgent
    from evaluator import MockEnvironment, evaluate_agent
    
    def test_basic_functionality():
        """测试基本功能"""
        print("1. 测试基本功能...")
        
        # 创建代理
        agent = PolicyAgent()
        print("   ✓ PolicyAgent创建成功")
        
        # 测试单步推理
        test_obs = np.random.uniform(-1, 1, 5).astype(np.float32)
        action = agent.get_action(test_obs)
        
        print(f"   输入观测: {test_obs}")
        print(f"   输出动作: {action}")
        print(f"   动作类型: {type(action)}")
        print(f"   动作形状: {action.shape}")
        print(f"   动作范围: [{action.min():.3f}, {action.max():.3f}]")
        
        # 验证输出格式
        assert isinstance(action, np.ndarray), "动作必须是numpy数组"
        assert action.shape == (3,), f"动作形状错误，期望(3,)，得到{action.shape}"
        assert action.dtype == np.float32, f"动作类型错误，期望float32，得到{action.dtype}"
        assert np.all(np.abs(action) <= 1.0), "动作值必须在[-1,1]范围内"
        
        print("   ✓ 基本功能测试通过")
        return True
    
    def test_consecutive_calls():
        """测试连续调用"""
        print("\n2. 测试连续调用...")
        
        agent = PolicyAgent()
        
        # 模拟连续的环境交互
        obs = np.random.uniform(-1, 1, 5).astype(np.float32)
        actions = []
        
        for i in range(20):
            action = agent.get_action(obs)
            actions.append(action.copy())
            
            # 模拟状态变化 (修复维度不匹配问题)
            # 使用动作的前3个分量来影响观测的前3个分量
            state_change = np.zeros(5)
            state_change[:3] = 0.1 * action[:3]
            obs = obs + state_change + np.random.normal(0, 0.02, 5)
            obs = np.clip(obs, -1, 1).astype(np.float32)
        
        # 检查动作的一致性
        actions = np.array(actions)
        print(f"   生成{len(actions)}个动作")
        print(f"   动作标准差: {np.std(actions, axis=0)}")
        print(f"   动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # 检查是否所有动作都有效
        assert np.all(np.abs(actions) <= 1.0), "所有动作必须在有效范围内"
        assert not np.any(np.isnan(actions)), "动作中不能包含NaN"
        assert not np.any(np.isinf(actions)), "动作中不能包含无穷大"
        
        print("   ✓ 连续调用测试通过")
        return True
    
    def test_edge_cases():
        """测试边界情况"""
        print("\n3. 测试边界情况...")
        
        agent = PolicyAgent()
        
        # 测试极值输入
        test_cases = [
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),  # 最小值
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),       # 最大值
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),       # 零值
            np.random.uniform(-1, 1, 5),                # 随机值
        ]
        
        for i, test_obs in enumerate(test_cases):
            test_obs = test_obs.astype(np.float32)
            action = agent.get_action(test_obs)
            
            print(f"   测试案例 {i+1}: obs={test_obs} -> action={action}")
            
            assert isinstance(action, np.ndarray), f"案例{i+1}：动作类型错误"
            assert action.shape == (3,), f"案例{i+1}：动作形状错误"
            assert np.all(np.abs(action) <= 1.0), f"案例{i+1}：动作超出范围"
            assert not np.any(np.isnan(action)), f"案例{i+1}：动作包含NaN"
        
        print("   ✓ 边界情况测试通过")
        return True
    
    def test_environment_interaction():
        """测试环境交互"""
        print("\n4. 测试环境交互...")
        
        agent = PolicyAgent()
        env = MockEnvironment(max_steps=100)
        
        # 运行一个完整的episode
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(100):
            action = agent.get_action(obs)
            next_obs, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            
            obs = next_obs
            
            if done:
                break
        
        print(f"   完成一个episode: {steps}步，总奖励: {total_reward:.3f}")
        
        # 验证交互是否正常
        assert steps > 0, "至少应该执行一步"
        assert not np.isnan(total_reward), "总奖励不能是NaN"
        assert not np.isinf(total_reward), "总奖励不能是无穷大"
        
        print("   ✓ 环境交互测试通过")
        return True
    
    def test_performance():
        """测试性能"""
        print("\n5. 测试性能...")
        
        agent = PolicyAgent()
        
        # 运行简短的性能评估
        results = evaluate_agent(agent, num_episodes=5, max_steps=200, verbose=False)
        
        print(f"   平均奖励: {results['mean_reward']:.3f}")
        print(f"   奖励标准差: {results['std_reward']:.3f}")
        print(f"   平均episode长度: {results['mean_length']:.1f}")
        
        # 基本性能检查
        assert results['mean_reward'] > -1000, "平均奖励过低，可能存在问题"
        assert results['std_reward'] >= 0, "标准差必须非负"
        assert results['mean_length'] > 0, "平均长度必须为正"
        
        print("   ✓ 性能测试通过")
        return True
    
    def run_all_tests():
        """运行所有测试"""
        print("="*60)
        print("PolicyAgent 测试套件")
        print("="*60)
        
        tests = [
            test_basic_functionality,
            test_consecutive_calls,
            test_edge_cases,
            test_environment_interaction,
            test_performance
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"   ✗ 测试失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{"="*60}")
        print(f"测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过！PolicyAgent实现正确。")
            return True
        else:
            print("❌ 部分测试失败，请检查实现。")
            return False
    
    if __name__ == '__main__':
        success = run_all_tests()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保agent/agent.py文件存在且包含PolicyAgent类")
    sys.exit(1)
except Exception as e:
    print(f"测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
