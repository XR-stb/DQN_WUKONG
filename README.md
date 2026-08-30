# DQN_WUKONG — Recurrent RL Pipeline

使用屏幕画面和键鼠输入训练 AI 挑战《黑神话：悟空》Boss。当前主线已经从旧的
DQN/PPO/SAC 单循环实现迁移为：

- 1280×720 WGC 客户区捕获，严格校验尺寸；
- 160×90 RGB 视觉观测和带置信度的 HUD 状态；
- 固定 8Hz 基础动作，不再使用 0.1–10 秒不等的阻塞宏动作；
- 人类示范行为克隆预训练；
- R2D3/DQfD 风格的 CNN + LSTM、Double/Dueling DQN、n-step 和优先序列回放；
- 独立 Actor/Learner、原子检查点、训练/评估分离。

旧 DQN/PPO/SAC 文件仅作为历史基线保留，默认入口不会加载旧权重。
dxcam 只能通过配置显式启用，并要求先设置 `dxcam_osd_disabled: true`；该后端会按
Win32 客户区裁剪，但仍必须在游戏与显卡工具中关闭所有 OSD。

## 安装

要求 Windows 10/11、Python 3.10+、NVIDIA GPU，以及窗口模式 1280×720 的游戏。

```powershell
uv venv --python 3.10
uv pip install --python .\.venv\Scripts\python.exe -e ".[dev]"

# 如需实时图形仪表板
uv pip install --python .\.venv\Scripts\python.exe -e ".[dashboard]"
```

也可继续安装 `requirements.txt`，但推荐使用 `pyproject.toml`，它是新的依赖来源。

## 推荐工作流

所有命令默认读取 `config/rl_pipeline.yaml`。

```powershell
# 1. 进入寅虎战斗后校准，必须先确认所有框和置信度正确
python -m wukong_rl calibrate

# 2. 正常手动打 Boss，录制 30–60 分钟示范；Ctrl+C 安全保存当前回合
python -m wukong_rl record --boss yinhu

# 3. 行为克隆预训练
python -m wukong_rl pretrain --dataset artifacts/datasets

# 4. 示范优先回放 + 在线强化学习
python -m wukong_rl train --boss yinhu `
  --dataset artifacts/datasets `
  --checkpoint artifacts/checkpoints/bc-pretrained.pt

# 5. 关闭探索，冻结策略连续评估 20 局
python -m wukong_rl eval `
  --checkpoint artifacts/checkpoints/latest.pt `
  --episodes 20 --exploration 0

# 离线模型性能；增加 --live-capture 可同时测真实截图
python -m wukong_rl benchmark

# 可选：在另一个终端打开新 JSONL 训练仪表板
python -m wukong_rl.dashboard
```

兼容入口 `python main.py` 等价于默认 `train`。如未安装 editable package，可使用：

```powershell
$env:PYTHONPATH = "src"
python -m wukong_rl --help
```

## 动作空间

策略每 125ms 选择一个动作 token：待机、四向奔跑、轻击、重击保持、闪避、
技能 1–4、法宝、变身、喝药。连续选择重击会继续按住右键，切换动作立即释放。
五连击、连闪、隐身蓄力等行为由序列策略自行组合。

技能、法宝、变身、葫芦和低精力动作由检测置信度生成动作掩码。输入执行器在暂停、
异常和退出时统一释放全部按键与鼠标按钮。

## 奖励

默认奖励只使用可靠结果：Boss 每掉 1% 血 `+0.1`，自身每掉 1% 血 `-0.12`，
每个控制 tick `-0.001`，胜利 `+10`，失败 `-10`。非终局奖励裁剪到 `[-2, 2]`。
选择攻击、动作多样性和喝药本身都不会获得奖励。

## 数据与产物

- 示范：`artifacts/datasets/<boss>/<episode>/frames.npy + trajectory.npz + manifest.json`
- 磁盘映射回放：`artifacts/replay/`
- 检查点：`artifacts/checkpoints/`
- Actor/Learner/预训练指标：`artifacts/metrics/*.jsonl`（Dashboard 只读这些新日志）
- 冻结策略录像和报告：`artifacts/evaluations/<run>/`

这些目录默认不进入 Git。当前旧训练 CSV 已被判定含有血条跳变和假奖励，只用于失败
基线，不能导入新训练器。

## 验证

```powershell
python -m pytest -q
python -m wukong_rl benchmark --iterations 100
```

首版发布门槛是冻结策略连续 20 局寅虎胜率不低于 50%。工程性能门槛为推理 p95
低于 15ms、观测处理 p95 低于 20ms、8Hz deadline miss 低于 5%、显存低于 8GB。

更多设计细节见 `docs/architecture/rl-pipeline.md`，录制与训练排障见
`docs/training-guide.md`，本机离线性能基线见 `docs/benchmark-rtx4060.md`。
