# Legacy model implementations

此目录根部的 DQN、DDQN、PPO、PPO-ReF 和 SAC-Discrete 是重构前的对照实现。
它们不再由 `python -m wukong_rl` 加载，旧权重也不兼容新 Observation、动作 token
和检查点 schema。请只把这些文件用于历史对比，不要与新训练链路混用。
