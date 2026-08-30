# RTX 4060 离线性能基线

记录时间：2026-08-31。环境为 NVIDIA GeForce RTX 4060、PyTorch 2.11.0+cu126，执行：

```powershell
python -m wukong_rl benchmark --iterations 100
```

| 指标 | 结果 | 验收线 |
|---|---:|---:|
| GPU 单步推理 p95 | 1.58 ms | < 15 ms |
| CPU Actor 单步推理 p95（4 线程） | 2.63 ms | < 15 ms |
| HUD + resize 观测处理 p95 | 1.67 ms | < 20 ms |
| batch 16 / 45-step Learner 更新 | 421.46 ms | 8Hz 下可持续 0.25 update/step |
| Learner 峰值显存 | 2260.73 MB | < 8192 MB |
| 网络参数量 | 1,038,896 | — |

观测处理使用 1280×720 合成帧，捕获指标需要游戏窗口存在时增加 `--live-capture` 后
复测。因此本表不能替代最终 8Hz deadline miss、真实 WGC 捕获和 20 局冻结策略验收。
