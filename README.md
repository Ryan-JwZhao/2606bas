# BAS 使用说明

## 简介

BAS 是一个台球辅助系统，主链路包含：

- 采集：相机或视频输入
- 感知：球与球杆检测
- 跟踪：跨帧目标跟踪
- 状态：回合、进球、胜负等状态判定
- 规划：推荐击球路线、目标球与自由球路径
- 投影：把规划结果投到球台上

当前仓库建议优先使用 `modern` 状态机，并基于本地视频或相机配置做联调。

## 模块结构

- `bas/capture`：相机与视频输入
- `bas/perception`：检测模型与检测区域
- `bas/tracking`：时序跟踪
- `bas/state`：状态机、进球判定、事件流
- `bas/planning`：规则路线、target shot、free shot、cue sector
- `bas/projection`：投影叠加与动画
- `bas/ui`：桌面控制台
- `scripts`：本地调试和辅助脚本
- `tests`：单测与回归测试

## 快速开始

1. 安装依赖

```powershell
Setup_Environment.cmd
```

2. 启动桌面控制台

```powershell
Start_BAS.cmd
```

3. 直接用 Python 启动 UI

```powershell
.\.venv\Scripts\python.exe -m bas ui
```

4. 无界面跑主流程

```powershell
.\.venv\Scripts\python.exe -m bas run --headless --max-frames 90
```

5. 检查相机

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
```

## Shared Aim 优化说明

这次针对 `planner` 的性能瓶颈做了根因修复，没有优先改状态机本体。

优化点：

- `GeometryPhysicsPlanner.plan()` 每帧创建一个共享的 `PlannerAimFrameContext`
- `CueStickAimDetector.detect()` 在同一帧内最多调用一次
- `target_shot` 与 `cue_sector` 共享同一个 aim 结果
- `_pointed_ball()` 改为纯几何选球，不再对每个候选球重复跑 Hough/Canny
- `CueStickAimPx` 增加最小必要的 track 元数据，供共享结果做后置质量过滤

实现位置：

- `bas/planning/aim_context.py`
- `bas/planning/planner.py`
- `bas/planning/target_shot.py`
- `bas/planning/cue_sector.py`
- `bas/planning/cue_aim.py`

## 本地性能复测

共享 aim 优化后的本地 benchmark 脚本：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_planner_shared_aim.py --frames 40 90 120
```

脚本会自动读取当前有效配置：

- `UserSettings.load().apply_to_config(AppConfig.load()).resolve_paths()`

输出指标：

- `fps`
- `planner_plan_avg_ms`
- `target_shot_avg_ms`
- `detect_calls_per_frame`
- `max_detect_calls_per_frame`

目标不是锁死某个绝对 fps，而是确认：

- 同一帧 `CueStickAimDetector.detect` 调用次数不超过 `1`
- `planner.plan` 与 `target_shot` 平均耗时明显低于旧基线

## 当前基线与复测口径

修复前基线（同一套本地 `video + ultralytics + modern` 配置）：

- `40f = 2.57 fps`
- `90f = 3.70 fps`
- `120f = 4.68 fps`
- `planner.plan ≈ 322.9 ms/frame`
- `target_shot ≈ 291.7 ms/frame`

修复后请直接运行上面的 benchmark 脚本，以当前本机配置复测。

## 回归测试

这次优化相关的回归命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_state_modern.py tests\test_planner.py tests\test_cue_aim.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

## 仓库约定

- 大文件、日志、回放产物不要提交，相关目录需要维护在 `.gitignore`
- 调试逻辑优先做成独立脚本或模块，避免把临时测量代码塞进核心链路
- 排查问题时优先复用 `tests`、`scripts` 和 replay/benchmark 工具，保证可复现
