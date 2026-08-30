# 🎮 DQN_WUKONG — AI Agent 运行指南

> 基于深度强化学习的《黑神话：悟空》Boss 战自动对战系统

---

## 📋 目录

- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [项目架构](#项目架构)
- [配置文件说明](#配置文件说明)
- [运行步骤（详细）](#运行步骤详细)
- [训练监控](#训练监控)
- [奖励函数设计](#奖励函数设计)
- [模型说明](#模型说明)
- [热键操作](#热键操作)
- [常见问题](#常见问题)
- [调参指南](#调参指南)

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 打开黑神话悟空游戏，进入Boss战前的场景（如寅虎前）
#    游戏窗口分辨率设置为 1280×720

# 3. 运行AI
python main.py

# 4. 等待检测到游戏窗口后，按 G 键开始训练
```

---

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| Python | 3.10+ |
| GPU | NVIDIA GPU（CUDA 支持），建议 RTX 3060+ |
| 内存 | 16GB+ |
| 游戏 | 黑神话：悟空 |
| 分辨率 | 游戏窗口模式 1280×720 |

### 关键依赖

```
torch (PyTorch with CUDA)
numpy
opencv-python
pynput          # 键鼠控制
PyYAML          # 配置文件
```

---

## 项目架构

```
DQN_WUKONG/
├── main.py                 # 🚀 入口：初始化相机、创建共享内存、启动训练进程
├── process_handler.py      # 🧠 训练循环：动作选择 → 执行 → 观察 → 奖励 → 训练
├── reward_calculator.py    # 🎯 奖励函数 v3：收敛优先设计（核心文件）
├── context.py              # 📡 共享内存IPC：主进程↔训练进程的帧/状态通信
├── grabscreen.py           # 📸 WGC截屏：Windows Graphics Capture实时截图
├── window.py               # 👁️ UI检测：血条、技能、精力等像素级识别
├── actions.py              # ⌨️ 动作执行器：键鼠模拟
├── keys.py                 # 🎹 虚拟按键映射
├── tracker.py              # 📊 数据追踪：CSV日志 + 实时JSON状态
├── judge.py                # 🔧 旧版奖励入口（已弃用，保留兼容）
├── log.py                  # 📝 日志配置
├── timing_decorator.py     # ⏱️ 性能计时装饰器
│
├── models/                 # 🤖 RL算法实现
│   ├── base_agent.py       # 基类（抽象接口）
│   ├── sac_discrete.py     # ★ SAC-Discrete v2（当前使用，带CNN特征提取）
│   ├── ppo.py              # PPO（可选）
│   ├── ppo_ref.py          # PPO + 经验回放增强（可选）
│   ├── dqn.py              # DQN（可选）
│   ├── ddqn.py             # DDQN（可选）
│   └── ddqn_resnet.py      # DDQN-ResNet（可选）
│
├── config/                 # ⚙️ 配置文件
│   ├── models_conf.yaml    # 模型选择与超参数
│   ├── game_conf.yaml      # 游戏窗口参数（1280×720）
│   ├── actions_conf.yaml   # 动作定义（17个热键动作）
│   ├── reward_conf.yaml    # 奖励参数参考（实际值在reward_calculator.py中）
│   └── dashboard_conf.yaml # 训练仪表盘配置
│
├── utils/                  # 🔧 工具函数
│   ├── overlay_hud.py      # OSD叠加层（实时显示训练信息）
│   ├── show_game_window.py # 游戏窗口显示工具
│   ├── display_game_info.py# 游戏状态信息显示
│   └── find_blood_location.py # 血条定位工具
│
├── train_data/             # 📈 训练数据
│   ├── data/               # CSV日志 + JSON实时状态
│   └── live_dashboard.py   # 实时训练仪表盘
│
├── model_weight/           # 💾 模型权重存储目录
├── bench/                  # 🔬 性能基准测试
└── images/                 # 🖼️ 参考图片
```

---

## 配置文件说明

### `config/models_conf.yaml`（核心配置）

```yaml
model:
  type: 'SAC_Discrete'     # 选择算法：SAC_Discrete | PPO | DDQN 等
  model_file: 'model_weight' # 模型保存目录

  SAC_Discrete:
    replay_size: 100000     # 经验回放缓冲区大小
    gamma: 0.98             # 折扣因子
    lr: 0.0003              # Actor+CNN 学习率
    critic_lr: 0.001        # Critic 学习率
    batch_size: 64          # 训练批大小
    tau: 0.005              # 目标网络软更新系数
    net_width: 256          # MLP隐层宽度
    warmup_steps: 1000      # 前N步纯随机探索

training:
  episodes: 5000            # 总训练回合数
  update_step: 50           # 每N步更新目标网络
  save_step: 4              # 每N回合保存模型
  restart_action: 'FUZHAN_STAND_RESTART'  # 死亡后重启动作

environment:
  width: 224                # 图像resize宽度
  height: 224               # 图像resize高度
```

### `config/game_conf.yaml`

```yaml
game:
  game_width: 1280          # 游戏窗口宽度
  game_height: 720          # 游戏窗口高度
  active_boss: '寅虎'      # 当前训练的Boss
  roi: [300, 400]           # 战斗区域ROI
```

### `config/actions_conf.yaml`

定义了17个可用动作（hot_list）:

| 索引 | 动作名 | 说明 | 耗时(秒) |
|------|--------|------|----------|
| 0 | IDLE | 待机 | 0.1 |
| 1 | LIGHT_ATTACK | 轻攻击(平A) | 0.39 |
| 2 | HEAVY_ATTACK | 重攻击 | 2.68 |
| 3 | DODGE | 闪避 | 0.50 |
| 4 | DODGE_TWO | 双段闪避 | 1.00 |
| 5 | DODGE_THREE | 三段闪避 | 1.50 |
| 6 | ATTACK_DODGE | 攻击+闪避 | 0.89 |
| 7 | QIESHOU | 切手 | 1.50 |
| 8 | FIVE_HIT_COMBO | 五连击 | 4.15 |
| 9 | SKILL_1 | 技能1(定身) | 0.56 |
| 10 | STEALTH_CHARGE | 隐身蓄力 | 10.0 |
| 11 | SKILL_3 | 技能3 | 2.12 |
| 12 | FABAO | 法宝 | 1.35 |
| 13 | TISHEN | 替身 | 4.31 |
| 14 | GO_BACK | 后退 | 1.2 |
| 15 | GO_FORWARD | 前进 | 1.2 |
| 16 | DRINK_POTION | 喝药 | 1.49 |

---

## 运行步骤（详细）

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 2. 游戏设置

1. 打开《黑神话：悟空》
2. 设置为**窗口模式 1280×720**
3. 进入要训练的Boss战场景前（如寅虎前的土地庙）
4. 确保游戏窗口没有被其他窗口遮挡

### 3. 启动训练

```bash
python main.py
```

启动后会依次：
- 初始化 WGC 截屏
- 等待检测游戏窗口（自动识别）
- 创建共享内存
- 启动训练子进程
- 启动 OSD 叠加层
- 启动实时仪表盘

### 4. 开始训练

- 按 **G** 键开始/暂停训练
- AI 会自动执行动作、观察结果、计算奖励、更新模型
- 每回合结束后自动重启Boss战

### 5. 停止训练

- 按 **Ctrl+C** 优雅退出（等待当前动作完成）
- 再按一次 **Ctrl+C** 强制退出

---

## 训练监控

### OSD 叠加层（游戏内）

- 按 **F1** 切换显示/隐藏
- 显示实时信息：回合数、步数、动作名、奖励、血量等

### 实时仪表盘

- 自动随主程序启动
- 在浏览器中查看训练曲线、动作分布、血量变化等

### 训练日志

- CSV日志：`train_data/data/training_log.csv`
- 实时状态JSON：`train_data/data/realtime_state.json`

---

## 奖励函数设计

### v3 设计哲学（当前版本）

**核心原则：正向激励为主，惩罚保持克制**

| 行为 | 奖励值 | 说明 |
|------|--------|------|
| 任何攻击动作 | +8 | 基础正奖励，鼓励进攻 |
| 安全攻击(平A) | +5 (额外) | 平A快出快收，额外奖励 |
| 造成Boss伤害 | +30/1% | 每打掉Boss 1%血 |
| 安全输出(未受伤) | +15 | 造成伤害且没受伤 |
| 闪避(威胁窗口内) | +10 | 被攻击时正确闪避 |
| 受伤 | -20 (固定) | **不递增**，避免后期惩罚爆炸 |
| IDLE | -3 | 轻微惩罚，推动探索 |
| 连续IDLE(>3次) | -8 | 稍重惩罚 |
| 胜利 | +500 | 适度，不造成梯度不稳 |
| 失败 | -100 | 适度 |

### vs v2 的关键改变

1. **移除CD惩罚**: 不再因动作冷却扣分(-60移除)
2. **移除CD硬拦截**: 不再强制替换为IDLE
3. **固定受伤惩罚**: -20固定（v2的递增 -15×cnt 移除）
4. **降低回合奖惩**: WIN 3000→500，LOSE -1500→-100
5. **攻击miss不惩罚**: v2的-3移除，鼓励尝试攻击

---

## 模型说明

### SAC-Discrete v2（当前使用）

**架构**:
```
图像输入 (3×224×224)
    ↓
CNN特征提取器 (Conv2d×3 + AdaptiveAvgPool + FC)
    ↓
视觉特征 (128维)  +  上下文特征 (14维)
    ↓
拼接 → MLP (256×256) → 动作概率分布 (17维)
```

**核心特性**:
- **CNN视觉处理**: 用卷积网络压缩图像，替代v1的150K维MLP
- **Off-policy**: 经验回放100K条，采样效率高
- **自动温度调节**: 自适应探索强度
- **双Q网络**: 减少过估计
- **Warmup**: 前1000步纯随机探索
- **奖励归一化**: Running normalization 稳定训练

### 状态空间

- **视觉输入**: 游戏画面ROI (300×400)，resize到224×224，通过CNN压缩为128维
- **上下文特征** (14维):
  - self_blood (0~1): 玩家血量
  - boss_blood (0~1): Boss血量
  - self_energy (0~1): 精力
  - self_magic (0~1): 法力
  - hulu (0~1): 药瓶
  - skill_1~4 (0/1): 技能可用状态
  - skill_ts (0/1): 替身可用
  - skill_fb (0/1): 法宝可用
  - gunshi1~3 (0/1): 棍势层数

### 动作空间

17个离散动作（见上方动作表）

---

## 热键操作

| 按键 | 功能 |
|------|------|
| G | 开始/暂停训练 |
| F1 | 切换OSD叠加层显隐 |
| Ctrl+C | 优雅退出 |
| Ctrl+C ×2 | 强制退出 |

---

## 常见问题

### Q: 游戏窗口检测不到？
检查游戏是否为窗口模式 1280×720，确保没有全屏或无边框全屏。

### Q: 训练没有反应？
确保按了 G 键开始训练。训练开始后日志会输出回合信息。

### Q: GPU内存不足？
减小 `batch_size`（64→32）或 `replay_size`（100000→50000）。

### Q: 训练收敛很慢？
正常。前1000步是warmup（纯随机），之后才开始学习。预期20-50回合开始看到Boss血量下降趋势。

### Q: 如何切换Boss？
修改 `config/game_conf.yaml` 中的 `active_boss`，并修改 `restart_action` 为对应Boss的重启动作。

### Q: 如何从头重新训练？
删除 `model_weight/` 下的所有 `.pth` 文件和 `train_data/data/` 下的日志文件。

---

## 调参指南

### 想让AI更激进攻击
```python
# reward_calculator.py
ATTACK_BASE_REWARD = 12.0  # 从8提高到12
SAFE_ATTACK_BONUS = 8.0    # 从5提高到8
```

### 想让AI更注重防御
```python
DODGE_SUCCESS_REWARD = 20.0  # 从10提高到20
INJURY_PENALTY = -30.0       # 从-20加重到-30
```

### 想加快探索
```yaml
# config/models_conf.yaml
warmup_steps: 500      # 减少warmup
target_entropy_ratio: 0.8  # 提高目标熵
```

### 想更稳定训练
```yaml
tau: 0.001              # 降低软更新速率
lr: 0.0001              # 降低学习率
batch_size: 128         # 增大batch
```

---

## 多进程架构

```
main.py (主进程)
├── context.update_status()  # 持续更新游戏状态到共享内存
├── WGC截屏线程              # 后台截图
├── OSD叠加层进程            # 常驻，F1切换显隐
├── 实时仪表盘进程           # 独立子进程
└── 训练进程 (process_handler.py)
    ├── choose_action()      # 从策略分布采样动作
    ├── take_action()        # 键鼠模拟执行
    ├── 事件监控              # 紧急/普通事件队列
    ├── judge()              # 计算奖励
    ├── store_data()         # 存入经验回放
    └── train_network()      # 梯度更新
```

---

*最后更新: 2026-03-29 | 奖励函数 v3 + SAC-Discrete v2 (CNN)*
