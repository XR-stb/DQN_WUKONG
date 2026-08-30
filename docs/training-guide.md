# 寅虎训练指南

## 1. 校准

游戏设置为窗口模式 1280×720并进入战斗，运行 `calibrate`。检查生成的 PNG/JSON：

- 截图必须正好 1280×720；
- 玩家/Boss血量必须接近游戏显示值；
- 稳定画面连续检测时 confidence 应不低于 0.55；
- 不允许出现约 98% 与 1% 交替跳变。

任何一项失败都应先修改 `perception.regions/ranges`，不要开始录制或训练。

## 2. 示范录制

运行 `record` 后正常使用键鼠战斗。监听器只观察输入，不会注入按键。建议包含：

- 成功和失败回合；
- 近身、拉开、闪避、喝药和技能使用；
- 至少若干击败或接近击败 Boss 的轨迹。

每局自动形成独立目录；Ctrl+C 会将当前轨迹保存为 truncated。录完后确认每个目录
同时存在 manifest、frames 和 trajectory 文件。`trajectory.npz` 还包含与 8Hz 帧严格
对齐的原始按键/鼠标状态 JSON，便于重新映射动作 token 和数据审计。

## 3. 预训练与在线训练

先运行 `pretrain`。关注 validation accuracy、各动作 recall 和 confusion matrix，不能只
看总准确率；IDLE/轻击占比过大时应补录少数动作。

在线训练分别查看：

- `actor.jsonl`：检测置信度、deadline miss、动作熵、丢弃 transition；
- `train.jsonl`：TD loss、demo loss、Q/target、梯度、回放序列数；
- 检查点中的 config hash 必须与本次配置一致。

可在另一个终端运行 `python -m wukong_rl.dashboard` 查看只读实时图表；它不会进入
Actor 控制进程，也不会影响 8Hz deadline。

如果策略退化，优先排查检测错误、无效掩码和 Q 值发散，不要先增加训练时长。

开始在线训练前先保存两组基线：对未训练检查点用 `eval --exploration 1` 记录随机策略，
再对 `bc-pretrained.pt` 用 `eval --exploration 0` 记录纯 BC。二者均需保留 summary、配置
哈希和逐局 1280×720 录像，后续在线策略只能与这两组冻结结果比较。

## 4. 冻结评估

评估使用 exploration=0，轨迹不进入训练回放。每次评估保存 20 个 MP4 和 summary。
`passed=true` 仅表示 20 局胜率达到 50%；仍需检查视频中是否存在利用 UI 识别错误的
异常策略。
