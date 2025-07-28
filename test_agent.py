import importlib.machinery
import importlib.util
import gym
import numpy as np
import os
from evaluator import get_env, config


def load_agent(path, agent_type=None):
    """动态加载 agent.py 并实例化 Agent"""
    loader = importlib.machinery.SourceFileLoader('agent', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    
    # 如果指定了agent类型，使用工厂函数创建
    if agent_type is not None and hasattr(mod, 'create_agent'):
        return mod.create_agent(agent_type)
    
    # 否则使用默认的PolicyAgent
    return mod.PolicyAgent()


def evaluate_agent(agent, env, seeds):
    results = []
    for seed in seeds:
        obs = env.reset(seed=seed)     
        agent.reset()
        total_reward = 0.0

        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        results.append(total_reward)
    return results


def main():
    # 从环境变量或命令行参数获取agent类型
    agent_type = os.environ.get("AGENT_TYPE", "mlp")
    
    if len(os.sys.argv) > 1:
        agent_type = os.sys.argv[1]
    
    print(f"Using agent type: {agent_type}")
    
    # 1. load agent
    try:
        agent = load_agent("./agent/agent.py", agent_type)
        print(f"Loaded agent: {type(agent).__name__}")
    except Exception as e:
        print(f"Failed to load {agent_type} agent: {e}")
        print("Falling back to default MLP agent...")
        agent = load_agent("./agent/agent.py")
    
    agent.seed(config["seed"])

    # 2. create one env，循环复用
    env = get_env()

    # 3. evaluate
    results = evaluate_agent(agent, env, config["seeds"])

    # 4. clean up
    env.close()
    agent.close()

    # 5. 可选：打印或返回
    mean_score = np.mean(results)
    std_score = np.std(results)
    print(f"Score: {mean_score:.4f} ± {std_score:.4f}")
    return results

if __name__ == "__main__":
    main()