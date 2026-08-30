<div align="center">

# 🐵 DQN_WUKONG

**用深度强化学习打黑神话悟空 Boss**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[📺 演示视频](https://www.bilibili.com/video/BV1DrpheREXh) · [🚀 快速开始](#-快速开始) · [🎯 自定义奖励](#-自定义你的奖励函数) · [📊 训练监控](#-实时训练仪表板)

</div>

---

## ✨ 特性

- 🎮 **多算法支持** — DQN / DDQN / PPO / SAC / PPO-ReF，一键切换
- 🧠 **即插即用** — 修改配置文件即可适配不同 Boss、不同分辨率
- 📊 **实时监控** — 训练过程中实时查看奖励曲线、胜率、综合评分
- ⚡ **高效采集** — 基于 DXCam 的高性能游戏画面捕获
- 🔧 **全配置化** — 奖励函数、游戏参数、模型超参全部 YAML 配置，零代码调参

## 📁 项目结构

```
DQN_WUKONG/
├── config/                     # ⚙️ 所有配置文件 (新用户从这里开始!)
│   ├── game_conf.yaml          #    游戏窗口 & Boss血条坐标
│   ├── reward_conf.yaml        #    奖励函数参数 (核心调参文件)
│   ├── models_conf.yaml        #    模型算法 & 超参数
│   ├── actions_conf.yaml       #    动作空间定义
│   └── dashboard_conf.yaml     #    训练监控参数
├── models/                     # 🧠 强化学习算法实现
│   ├── dqn.py                  #    DQN
│   ├── ddqn.py                 #    Double DQN
│   ├── ppo.py                  #    PPO
│   ├── sac_discrete.py         #    SAC (离散动作版)
│   └── ppo_ref.py              #    PPO + 经验回放增强
├── train_data/                 # 📊 训练数据 & 可视化
│   └── live_dashboard.py       #    实时训练仪表板
├── utils/                      # 🔧 工具脚本
│   ├── display_game_info.py    #    调试: 查看血量识别效果
│   ├── change_window.py        #    自动校正游戏窗口位置
│   └── ...
├── main.py                     # 🚀 程序入口
├── judge.py                    # 🎯 奖励函数 (读取 reward_conf.yaml)
├── window.py                   # 🖥️ 画面识别 (读取 game_conf.yaml)
├── tracker.py                  # 📈 训练数据记录器
├── process_handler.py          # 🔄 训练循环控制
├── context.py                  # 📡 多进程共享内存通信
├── actions.py                  # 🎮 动作执行器
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建 Conda 环境 (推荐)
conda create --name wukong python=3.10
conda activate wukong

# 安装 PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

<details>
<summary>💡 使用 uv 安装 (更快)</summary>

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv
uv pip install -r requirements.txt
uv run main.py
```
</details>

验证安装:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}', 'GPU ✅' if torch.cuda.is_available() else 'GPU ❌')"
```

### 2. 游戏设置

| 设置项 | 要求 |
|--------|------|
| 显示模式 | **窗口模式** |
| 分辨率 | **1280×720** (或自定义，见下方) |
| 视角锁定 | **开启自动锁定Boss** |

### 3. 适配你的游戏

打开 `config/game_conf.yaml`，只需修改两处:

```yaml
# 1️⃣ 修改为你的游戏分辨率
game_window:
  width: 1280
  height: 720

# 2️⃣ 选择你要打的Boss
active_boss: '青背龙'     # 改成你的目标Boss
```

然后运行调试工具，确认血量识别是否正确:
```bash
python -m utils.display_game_info
```

### 4. 开始训练

```bash
python main.py
```

> 💡 进入 Boss 对战后，等几秒看到 Boss 血条，按 **`G`** 键开始/暂停训练

### 5. 实时监控训练进度

在另一个终端运行:
```bash
python train_data/live_dashboard.py
```

## 🎯 自定义你的奖励函数

奖励函数是 AI 学习的核心驱动力。所有参数集中在 `config/reward_conf.yaml`:

```yaml
# 想让AI更激进? 提高Boss伤害奖励
events:
  boss_blood_change_multiplier: 6.0  # ← 调大这个值

# 想让AI更防御? 提高受伤惩罚
injury:
  base_penalty_multiplier: 50        # ← 调大这个值

# 想让AI多用技能? 提高技能奖励
skills:
  skill_ready_reward: 100            # ← 调大这个值
```

> 📖 完整参数说明见 [`config/reward_conf.yaml`](config/reward_conf.yaml) 文件注释

## ⚙️ 切换算法

在 `config/models_conf.yaml` 中修改一行即可:

```yaml
model:
  type: 'SAC_Discrete'   # 可选: DQN / DDQN / PPO / SAC_Discrete / PPO_ReF
```

| 算法 | 适合场景 | 显存占用 |
|------|----------|---------|
| `DQN` | 入门学习 | ~2 GB |
| `DDQN` | 基础训练 | ~2 GB |
| `PPO` | 通用场景 | ~3 GB |
| `SAC_Discrete` | **推荐首选**，采样效率高 | ~4 GB |
| `PPO_ReF` | PPO增强版，效率更高 | ~4 GB |

## 📊 实时训练仪表板

训练时在另一个终端运行 `python train_data/live_dashboard.py`，实时查看 6 项核心指标:

```
┌─────────────────┬──────────────────┬─────────────────┐
│ 📈 回合总奖励    │ 🩸 Boss剩余血量   │ 🏆 胜率趋势     │
├─────────────────┼──────────────────┼─────────────────┤
│ 💥 受伤次数      │ ⏱ 存活时间       │ 🎯 AI综合评分    │
└─────────────────┴──────────────────┴─────────────────┘
```

## 🤝 致谢

- [DQN_play_sekiro](https://github.com/analoganddigital/DQN_play_sekiro) — 灵感来源
- [pygta5](https://github.com/Sentdex/pygta5) — 屏幕捕获思路
- [GameAISDK](https://github.com/Tencent/GameAISDK) — 更通用的游戏AI框架