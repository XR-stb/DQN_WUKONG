# 寅虎训练指南

## 1. 校准

游戏设置为窗口模式 1280×720并进入战斗，运行 `calibrate`。检查生成的 PNG/JSON：

- 截图必须正好 1280×720；
- 玩家/Boss血量必须接近游戏显示值；
- 稳定画面连续检测时 confidence 应不低于 0.55；
- 不允许出现约 98% 与 1% 交替跳变。

任何一项失败都应先修改 `perception.regions/ranges`，不要开始录制或训练。

推荐按 `docs/telemetry.md` 安装只读遥测并先运行 `telemetry-probe`。正式重新采集时可将
`telemetry.mode` 改成 `required`，这样 Mod 断线会直接停止，而不会无声录入视觉误差。

## 2. 示范录制

运行 `record` 后默认立即录制；F8 暂停/恢复，F9 保存并退出；监听器只观察输入，不会
注入按键。`--seconds` 只累计真正生成 fighting transition 的时间，加载/等待不占额度。
需要先切回游戏再手动开始时，可增加
`--start-paused`，看到 ARMED 后按 F8。建议包含：

- 成功和失败回合；
- 近身、拉开、闪避、喝药和技能使用；
- 至少若干击败或接近击败 Boss 的轨迹。

每局自动形成独立目录；Ctrl+C 会将当前轨迹保存为 truncated。录完后确认每个目录
同时存在 manifest、frames 和 trajectory 文件。`trajectory.npz` 还包含与 8Hz 帧严格
对齐的原始按键/鼠标状态 JSON，便于重新映射动作 token 和数据审计。

## 3. 预训练与在线训练

先运行 `pretrain`。关注 validation accuracy、各动作 recall 和 confusion matrix，不能只
看总准确率；IDLE/轻击占比过大时应补录少数动作。

遥测配置会改变 config hash。启用本版本后不要继续使用旧的 `bc-balanced.pt`；保留旧文件
作为基线，使用新录制数据生成新的 BC 检查点，再开始在线 RL。

预训练会保存三个检查点：命令指定的文件按核心动作平衡分数选择，用于后续在线训练；
同目录的 `*-best-loss.pt` 保存最低验证损失，`*-last.pt` 保存最后一轮。核心动作平衡
分数结合总体准确率与 IDLE、四向移动、轻击、重击、闪避的调和平均召回，避免总体
准确率较高但完全不会闪避的模型被选为默认策略。
每个 batch 还会固定抽取一段真实回合开头，对 burn-in 前缀施加较小的监督损失，避免
模型只学会“已有 LSTM 历史时怎么打”，实际开局零状态却连续输出 IDLE。

在线训练分别查看：

- `actor.jsonl`：检测置信度、deadline miss、动作熵、丢弃 transition；
- `train.jsonl`：TD loss、demo loss、Q/target、梯度、回放序列数；
- 检查点中的 config hash 必须与本次配置一致。

首选在另一个终端运行：

```powershell
python -m wukong_rl monitor --refresh 5 --window 10
```

该只读监控会显示当前 Boss/自身血量、最近与前一窗口的伤害/奖励/存活趋势、动作分布、
TD/Q/梯度、Q 与 skill4 掩码和遥测新鲜度。还可运行 `python -m wukong_rl.dashboard`
查看图形曲线；两者都不会进入 Actor 控制进程，也不会影响 8Hz deadline。

当前奖励按每局 Boss/自身的历史最低血线增量结算。喝药不加分，回血后重新掉到旧低点
不重复扣分；这避免同一管生命被反复累计成超过 100% 的受伤惩罚。修改这套语义后，
新在线轨迹写入 `artifacts/replay/online-v4-<奖励语义哈希>`，旧 `online-v3` 仅保留用于
历史诊断；修改喝药阈值或奖励系数也会自动切换回放目录，避免新旧语义混合。

如果策略退化，优先排查检测错误、无效掩码和 Q 值发散，不要先增加训练时长。

开始在线训练前先保存两组基线：对未训练检查点用 `eval --exploration 1` 记录随机策略，
再对 `bc-pretrained.pt` 用 `eval --exploration 0` 记录纯 BC。二者均需保留 summary、配置
哈希和逐局 1280×720 录像，后续在线策略只能与这两组冻结结果比较。

## 4. 冻结评估

评估使用 exploration=0，轨迹不进入训练回放。每次评估保存 20 个 MP4 和 summary。
`passed=true` 仅表示 20 局胜率达到 50%；仍需检查视频中是否存在利用 UI 识别错误的
异常策略。
