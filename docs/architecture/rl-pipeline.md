# RL Pipeline Architecture

## 数据流

```text
WGC client frame -> perception(value/confidence/age) -> Observation
       |                                              |
       |                                              v
       +---------------- fixed 8Hz Actor <- recurrent Q policy
                                                      |
                                                      v
                                              bounded transition queue
                                                      |
                                                      v
demonstrations -> fixed replay partition -> GPU Learner -> atomic checkpoint
                                      \------> actor weight snapshots
```

Actor 只负责实时环境与 CPU 推理；Learner 独占 GPU 训练。两者通过有界队列隔离，
指标写入不同 JSONL 流。推理权重每两秒发布一次，训练或磁盘写入不会进入控制时限。

## 稳定性约束

- WGC 必须返回精确的 1280×720 客户区。尺寸不符是致命错误，禁止静默 resize。
- dxcam 仅允许显式启用，会按 Win32 客户区裁剪，并要求配置确认 OSD 已全部关闭。
- HUD 字段同时携带 value、confidence 和 age；低置信度变化不产生奖励。
- Boss 血量在单回合内单调不增，超过阈值的跳变必须持续三帧。
- 终止状态机区分 fighting、won、lost、loading、invalid 和 truncated。
- Transition 分别保存 terminated 和 truncated；序列永不跨越回合边界。
- 回放保存 uint8 原始帧，视觉编码器在每次 Learner 更新中重新计算并获得梯度。
- 所有动作受掩码约束，目标网络选动作时使用 next observation mask。

## 学习器

默认网络约 104 万参数：IMPALA 风格 CNN、标量/置信度/上一动作融合、256 维 LSTM、
Dueling Q 头。训练使用 Double DQN、5-step return、8-step burn-in、32-step unroll、
优先序列回放和 DQfD large-margin 示范损失。默认 batch 中 25% 来自不可淘汰的示范
分区。

通用 LLM/VLM 不参与实时推理。LLM 式思路体现在动作 token、因果序列上下文、
teacher forcing 预训练和离线数据复用；未来 Decision Transformer 可通过相同
Observation/Trajectory schema 增加，不改变环境层。

## 兼容性

根目录旧模型和旧训练器属于 legacy baseline。新系统只读取 `config/rl_pipeline.yaml`
和 schema v2 的检查点/数据集，不会自动加载 `model_weight/` 中的旧权重。检查点同时
校验配置哈希与示范数据版本，避免在数据集悄然变化后继续复用不匹配的优化器状态。
