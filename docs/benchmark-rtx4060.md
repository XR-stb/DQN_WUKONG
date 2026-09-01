# RTX 4060 离线性能基线

注意：下列离线数据不代表 WGC 与识别并存时的实时表现；游戏实际帧率未测量。
卡顿调查参见 [实时性能监测](performance-monitoring.md)。

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

`--live-capture` 的截图计时与后面的合成观测计时不是同一条实时链路。
