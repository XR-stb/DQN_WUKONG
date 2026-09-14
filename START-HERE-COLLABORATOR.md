# 协作者开箱指南

这是仍在研发中的寅虎训练快照，不是已收敛模型。当前模型可用于复现实验、继续采集、
改进 BC/RL 和对照评测，不应把现有权重的胜率当作发布结果。

## 1. 初始化

要求 Windows 10/11、Python 3.10、支持当前 PyTorch 的 NVIDIA 驱动，以及你自己安装的
《黑神话：悟空》。在解压目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\scripts\bootstrap_collaborator.ps1
```

脚本会创建 `.venv`、安装核心/开发/图表依赖并运行测试。使用 Git 的协作者也可直接克隆：

```powershell
git clone --branch codex/rl-pipeline-rebuild --single-branch https://github.com/XR-stb/DQN_WUKONG.git
cd DQN_WUKONG
.\scripts\bootstrap_collaborator.ps1
```

默认不安装可选的进程/GPU 资源采样包，以免它影响首次安装。需要该组指标时重新运行：

```powershell
.\scripts\bootstrap_collaborator.ps1 -SkipTests -WithPerformanceMetrics
```

Git 仓库不包含私有训练资产。若要复现实验，请同时取得项目维护者生成的 starter 压缩包；
其中 `COLLABORATION-MANIFEST.json` 包含每个本地资产的 SHA-256。

## 2. 游戏与遥测

保持窗口模式 1280×720、开启 Boss 自动锁定。先进入寅虎战斗并校准：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl calibrate
```

精确血量遥测为可选的本地只读 Mod。必须完全退出游戏后安装；starter 包带有我们自己的
预编译 DLL，第三方 B1CSharpLoader 仍由脚本从固定发布地址下载并校验 SHA-256：

```powershell
.\scripts\install_telemetry_mod.ps1 -UsePrebuilt -EnableJit `
  -GameDirectory 'D:\SteamLibrary\steamapps\common\BlackMythWukong'
```

安装后进入战斗验证：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl telemetry-probe --seconds 30
```

不要把 Mod 用于联网、竞技或绕过游戏规则。本项目的 Mod 只读本机战斗状态，但游戏更新、
其他 Loader 或错误的 JIT 配置仍可能造成不兼容。

## 3. 继续当前实验

starter 包内的演示数据与检查点版本匹配。不要先修改 `config/rl_pipeline.yaml`，否则配置
哈希保护会拒绝加载旧权重。先做一次只读检查：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl monitor --once --window 10
```

进入寅虎战斗后继续在线训练：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl train --boss yinhu `
  --dataset artifacts/datasets-telemetry-clean `
  --checkpoint artifacts/checkpoints/latest.pt
```

另开终端查看图形仪表盘：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl dashboard --refresh 2 --window 50
```

若希望从 BC 基线重新开始，把检查点换成
`artifacts/checkpoints/bc-branched-v3.pt`。当前不建议不做冻结评测就长时间堆训练。

## 4. 协作约定

- 从 `codex/rl-pipeline-rebuild` 拉取并建立自己的功能分支；
- 不提交 `artifacts/`、`.venv/`、游戏文件、Loader 二进制或个人日志；
- 算法修改必须通过 `python -m pytest -q`；
- 奖励、动作空间、遥测字段或数据语义发生变化时，不得继续混用旧回放；
- 合并前附上冻结评测、配置哈希以及 Dashboard 截图。
