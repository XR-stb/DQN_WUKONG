# 录制卡顿诊断

当前重点是测量真实录制链路，不是用离线 CNN 跑分推断游戏是否流畅。
本轮不修改捕获后端、动作选择和检测算法，也不宣称已修复 WGC 竞争或实现游戏帧率测量。

## 直接录制

```powershell
# 可选资源指标依赖；没有时，阶段计时仍工作，资源字段显示 unavailable
uv pip install --python .\.venv\Scripts\python.exe -e ".[performance]"

# 默认开启监测，原录制命令不变
python -m wukong_rl record --boss yinhu

# CLI 默认待命；切回游戏后 F8 开始/暂停，F9 保存并结束。
# 限时只计算实际处于 RECORDING 的时间。
python -m wukong_rl record --boss yinhu --seconds 120

# 自动化/旧行为：启动后立即录制，不等待 F8
python -m wukong_rl record --boss yinhu --seconds 120 --immediate

# 用于监测开关对照，不改变示范的数据格式
python -m wukong_rl record --boss yinhu --no-profile
```

启动会输出 profile 目录。默认每 5 秒打印状态、循环 Hz、采样数、已缓冲的 transition 数和监测丢包数。
即使没有进入 `fighting`，也保存 waiting 阶段的 HUD 与耗时；这样可以区分“没识别到战斗”与“控制循环太慢”。
`buffered/recorded_transitions` 不等于已落盘；只有成功的 `save` 事件与 `saved_episodes` 才代表完整写入。
Ctrl+C 会结束本次监测并生成报告；关闭整个终端或杀进程仍可能留下不完整文件。

每次运行的独立目录 `artifacts/profiles/<时间>-<模式>-<ID>/` 包含：

- `run.json`：配置、哈希、依赖版本、进程和平台信息。
- `events.jsonl`：逐 tick 分阶段墙钟/当前线程 CPU 时间、HUD、帧序号、输入队列、保存和异常事件。
- `resources.jsonl`：默认每秒一次的 CPU、RSS、线程数、读写速率、整卡利用率/显存/功耗/温度。
- `summary.json`：每 5 秒原子更新，结束时更新最终结果。
- `report.md`：结束时生成的离线报告。

不复制游戏画面到监测队列；画面仍只由示范数据集负责。监测日志默认被 Git 忽略。
资源依赖缺失、游戏进程没找到或 NVML 不支持时显示 N/A/原因，不能当成零占用。
支持通过 YAML `monitoring.game_process_name` 修改进程名，默认 `b1-Win64-Shipping.exe`。

## 指标口径

| 指标 | 含义与边界 |
|---|---|
| Loop Hz | 同阶段 tick 起点间隔算出的平均循环频率，不是游戏 FPS |
| Deadline miss | tick 本体耗时或同阶段起点间隔超过 `125ms + 5ms` 容差的比例；包含保存/调度间隙 |
| `capture_read` | 等锁、读取、复制、适配客户区与连续 BGR 内存的总耗时 |
| `capture_lock_wait_ms` / `capture_copy_ms` | WGC 缓存锁等待 / 复制墙钟时间；含被调度出去的时间 |
| `callback_hz` | 从捕获启动到最近缓存发布的平均回调频率，包含启动等待；不是 GPU 或显示刷新率 |
| `frame_age_ms` / `observation_age_ms` | 从 Python 回调发布缓存，到读取/完成观测的时间；不包含此前的驱动捕获延迟 |
| `repeated_frame` | 本次与上次读取的是同一缓存序号；不等于画面像素重复，也不是游戏掉帧数 |
| detect / terminal / resize 等 | 识别、终止判断、特征/掩码、颜色转换、缩放、打包的耗时 |
| input age | 当前 125ms 区间末尾所消费的最新离散输入有多旧；等待期输入丢弃，同一区间多次脉冲合并为最新一个并计数 |
| HUD invalid rate | 自身或 Boss HP 不可靠的比例；waiting 与 fighting 分开，没有观测时为 N/A |
| CPU one_core / machine | `100%` 分别代表占满一个逻辑核 / 整机所有逻辑核；不要混用 |
| GPU | NVML 整张卡占用，含游戏、捕获和其它程序，不是 torch 或游戏专属显存 |

耗时 p50/p95/p99 取每项最近 4096 个样本；count、mean_all、max_all 为全程累计。
诊断循环按预算 sleep，录制循环仍采用现有调度；二者端到端频率不能直接等同，阶段耗时可以辅助对比。
墙钟大、当前线程 CPU 小，只提示等待/调度/竞争/原生工作线程等可能，不能直接证明 GIL 问题。
Windows 的线程 CPU 时间精度也可能使短任务读数为零。

资源采样和 JSON 写入放在独立进程；录制进程用有界队列 `put_nowait` 发送纯标量数据。
队列满时丢监测事件，不阻塞控制，并报告 `dropped_events`；不是丢示范 transition。
报告同时监测采样进程自身 CPU/RSS、队列延迟和前次发送开销，不能假定监测成本为零。
录制入口现在按需加载，不会加载 PyTorch、模型或 CUDA。

## 四组可复现对照

保持同一画质、分辨率、窗口焦点、场景和其它后台程序，分别启动独立命令；不要同时运行。
建议每组 30–60 秒、多轮交替；短跑仅检查功能，不能作为游戏性能验收。
所有 diagnose 模式都不监听/注入按键、不训练、不保存示范。

```powershell
# 仅资源采样与睡眠循环，不启动截屏；用户正常玩游戏
python -m wukong_rl diagnose --mode baseline --seconds 30

# 增加 WGC，只读画面
python -m wukong_rl diagnose --mode capture --seconds 30

# 增加 HUD、状态和观测处理
python -m wukong_rl diagnose --mode observe --seconds 30

# 无 WGC，用原始校准帧反复处理；省略 --frame 会明确标记 synthetic
python -m wukong_rl diagnose --mode offline --frame artifacts/calibration/yinhu.png --seconds 30

# 当前目录与参考目录来自上述命令的输出
python -m wukong_rl profile-report --run artifacts/profiles/<observe目录> --compare artifacts/profiles/<offline目录>

# 可选仪表板：只读取固定大小摘要，不扫描全部历史 JSONL，不叠加到游戏画面
python -m wukong_rl.dashboard --profile artifacts/profiles/<目录>
```

`profile-report` 生成 `analysis.md/json`，不会覆盖示范数据或原始 summary。
对照首先看 callback/frame age、capture_read、detect/resize p95 与进程 CPU、整卡占用：

- capture 组就出现异常：继续调查 WGC 回调、复制、锁竞争和游戏 CPU/GPU 资源竞争。
- observe 组恶化、offline 组正常：调查捕获与处理并存时的调度/线程竞争，不仅看检测算法。
- 输入 age 越来越大：当前动作标签落后于画面，先解决吞吐与事件对齐，再收集训练数据。
- save 阶段明显尖峰：再考虑分块/异步写示范数据。监测写盘与示范写盘是不同路径。

这些是定位方向，不是仅凭单次 A/B 就能证明的原因。

## 游戏真实帧时间：导入 PresentMon

录制器自身不能测出游戏展示帧率。可另外使用官方 PresentMon 控制台导出 CSV；本项目不自动安装、提权、
启用 OSD 或改变游戏设置。已安装时，在另一个终端按同一时间窗口采集，例如：

```powershell
PresentMon.exe --process_name b1-Win64-Shipping.exe --v2_metrics --timed 30 --terminate_after_timed --output_file game-frames.csv
python -m wukong_rl profile-report --run artifacts/profiles/<目录> --presentmon-csv game-frames.csv
```

参数随安装版本而异，以 [PresentMon 官方控制台说明](https://github.com/GameTechDev/PresentMon/blob/main/README-ConsoleApplication.md) 为准。
导入支持 v1/v2/当前版常见列，先按程序筛选，再选择有效应用帧间隔最多的单一 PID/swapchain，报告明确列出选择结果。
应用帧间隔与显示持续时间分开统计；显示指标可能包含生成帧。NA、零和负值不参与间隔均值。
报告含 p95/p99 和应用帧间隔超过 33.33/50ms 的计数；不会将应用提交率冒充显示 FPS。
CSV 与录制日志尚未自动对齐时间，必须使用匹配场景/窗口；包含菜单或加载的长 CSV 不应直接与战斗比较。

CPU 百分比定义及首次非阻塞读数预热遵循 [psutil 官方文档](https://psutil.readthedocs.io/stable/index.html)。
没有外部 CSV 时，报告始终明确标记游戏 FPS 未测量。

## 已知限制

本轮监测接入录制和诊断命令；训练 Actor/Learner 保留已有训练指标，未统一接入这套逐阶段采样。
WGC 旧线程退出超时仍需单独修复，四组诊断请使用独立进程。
WGC 请求的最小更新间隔按捕获 `target_fps` 计算（当前 30Hz 对应 34ms）；这是限频请求，
不是恒定帧率保证，必须检查实测 `callback_hz`。
强杀进程无法保证最终摘要完整；先查看原始日志与 `end_reason`、`dropped_events`。
监测没有修正既有识别失效、输入积压或录制存盘性能，因此不能据此开始长时间训练。
