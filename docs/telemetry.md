# 只读游戏遥测与视觉融合

## 为什么采用混合感知

纯屏幕方案仍是安全兜底，但血条在特效、UI 淡出、死亡动画和加载期间会失真。新的默认
模式 `telemetry.mode: prefer` 从本地 C# Mod 读取精确数值，并继续用 WGC 提供策略画面、
葫芦和未校准技能状态。命名管道中断或数据超过 350ms 时会立即回退到屏幕检测，不会等待
管道，也不会阻塞 8Hz Actor。

遥测 Mod 只调用游戏已有的 getter，当前读取玩家/锁定目标的血量、法力、体力、棍势、
法宝/变身能量、死亡/战斗状态，以及可配置的技能可用状态。它不会修改属性、调用动作、
读写存档或向网络发送数据。数据只发布到本机
`\\.\pipe\wukong_rl_telemetry`。

## 构建与安装

安装前必须退出游戏。安装脚本会校验游戏 EXE、官方加载器压缩包 SHA-256，并在发现不同
的 `version.dll` 时停止，避免覆盖其他 Mod Loader。

```powershell
# 1. 拉取固定版本的开发接口（只写 artifacts/vendor，不进 Git）
.\scripts\setup_telemetry_dev.ps1

# 2. 离线构建和打包，不接触游戏目录
.\scripts\build_telemetry_mod.ps1

# 3. 退出游戏后安装
.\scripts\install_telemetry_mod.ps1
```

若 Steam 游戏不在默认目录，显式传入根目录：

```powershell
.\scripts\install_telemetry_mod.ps1 -GameDirectory 'D:\SteamLibrary\steamapps\common\BlackMythWukong'
```

加载器来自 `czastack/B1CSharpLoader` v0.0.8；编译接口固定为
`game-a11y/B1CSharpLoader` commit `5d607c9d32a14e03608a26e1913dc085bcaaa911`。
二者都是本机依赖，不会提交到仓库。

首次安装使用 `Develop=0`、`EnableJit=0`，因为本 Mod 不需要热加载或 Harmony Hook；
`Console=1` 暂时保留启动日志，完成首次探针后可改为 `0` 隐藏控制台。

## 首次探针与校准

安装后启动游戏、进入寅虎战斗，另开终端执行：

```powershell
.\.venv\Scripts\python.exe -m wukong_rl telemetry-probe --seconds 30
```

确认输出中：

- `connected=true`、`fresh=true`，`age_ms` 通常低于 200；
- 玩家与 Boss 血量随战斗连续变化，Boss 的 `target_res_id` 在锁定寅虎时稳定；
- 使用技能后 `last_skill_id` 更新。

当前游戏版本中寅虎的 `target_res_id` 实测为 `0`，不能作为稳定白名单；首版维持空的
`accepted_boss_res_ids`，仅在锁定目标存在时覆盖 Boss HUD。依次只释放一个技能，记录
`last_skill_id`，将四个 ID 同时写入：

- `config/rl_pipeline.yaml` 的 `telemetry.skill_ids`；
- 游戏目录下 `CSharpLoader/Mods/WukongTelemetry/skill_ids.txt`。

重启游戏后再次探针，四个 `skills[].ready` 应随冷却变化。ID 为 `0` 的槽位继续使用视觉
识别，不会伪造可用状态。

## 模式与故障策略

- `off`：完全关闭 Mod 客户端，使用纯屏幕检测；
- `prefer`：新默认值，使用新鲜遥测并自动视觉兜底；
- `required`：遥测缺失立即报错，适合正式数据录制和训练验收，避免悄悄混入污染数据。

遥测配置属于观测语义的一部分，会进入 config hash。因此接入后旧检查点不会被自动加载；
需要重新录制一小批探针数据并重新预训练。原数据和权重不会被删除。

## 性能与安全检查

Mod 默认 10Hz 在 Unreal 游戏线程只做 getter 采样；JSON 编码和命名管道写入位于后台线程。
管道刻意使用后台线程上的同步 I/O；不要改成 `PipeOptions.Asynchronous`，游戏内置 Mono 在
客户端断开时存在原生完成回调异常，会导致异常风暴和游戏假死。
Python 客户端关闭时也不会跨线程强关正在阻塞读取的句柄；它等待下一帧让后台 reader
自行退出，异常服务端不再有机会卡住 CLI 主线程。
训练指标额外记录 `telemetry_connected`、`telemetry_fresh` 和 `telemetry_age_ms`。正式录制前
仍需运行 `diagnose --mode baseline` 与 `diagnose --mode observe` 对比游戏 PresentMon 帧时间，
确认 Mod 与 WGC 的增量开销。
