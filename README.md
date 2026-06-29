# BAS 本地说明

## 简介

`2606BAS` 是一个台球辅助系统，包含实时采集、目标检测、跟踪、状态机、路线规划、前端预览和投影显示。

当前主要有两种工作模式：

- `规则模式`：给出建议击球路线和候选目标球。
- `自由模式`：按球杆方向模拟白球碰撞路径。

## 如何使用

1. 准备好相机、标定文件、检测模型和投影设备。
2. 在项目根目录启动程序：

```powershell
python -m bas
```

3. 先点击“开始采集”，确认前端预览正常。
4. 如需落地投影，再点击“开始投影”。
5. 根据现场需要切换 `规则模式` / `自由模式`。

## 2026-06-29 本次更新

### 1. 白球锚定的球杆方向解析

球杆方向不再依赖“最近端点”或单纯时间平滑，而是先把 `cue_stick` 检测结果视为一条无方向轴，再用白球位置决定这条轴的真实击打朝向。

核心原则：

- 球杆本体应该主要位于白球后方。
- 合理的击打方向应当从白球朝球杆本体的反方向发出。
- 这样在蓄力前后抽动时，只要检测到的还是同一条杆轴，就不会因为端点前后交换而翻线。

实现位置：

- `bas/planning/cue_direction_resolver.py`
- `bas/planning/cue_aim.py`

### 2. 球杆走廊锁定能力

为了解决“小幅抖动导致矩形框命中球一闪一闪”的问题，球杆矩形走廊现在改成了三层机制：

- `严格入框`
  - 新目标第一次进入走廊时，仍按当前矩形宽度做严格判定。
- `保持走廊`
  - 一旦某个目标已经被锁定，后续几帧会使用更宽松的横向容差和前向容差继续判定它是否仍合理。
- `连续释放`
  - 只有连续多帧都不再满足保持条件，才真正释放锁定。

这意味着：

- 单帧抖动不再立刻清空目标。
- 锁定球只要还在“保持走廊”里，就会继续输出，不会一直闪。
- 真正离开走廊时，系统仍然会按连续帧规则释放，避免锁死。

相关实现：

- `bas/planning/cue_sector.py`

### 3. 球杆矩形候选框只在前端显示

主界面新增 `显示球杆矩形候选框` 开关。

用途：

- 在前端预览里显示球杆走廊和候选目标球，方便排查识别和纠正范围。
- 仅在前端显示，不会进入投影 overlay，不会打到球台上。

相关位置：

- `bas/ui/cue_sector_preview.py`
- `bas/ui/main_window.py`

## 调参建议

如果现场仍然需要微调，优先看下面这些参数：

- `planner.cue_sector_corridor_width_px`
  - 严格入框的矩形总宽度。
- `planner.cue_sector_lock_margin_px`
  - 已锁定目标的额外横向保持余量。
- `planner.cue_sector_lock_forward_tolerance_px`
  - 已锁定目标允许的前向抖动容差。
- `planner.cue_sector_lock_release_frames`
  - 连续多少帧不满足保持条件才释放锁定。
- `planner.cue_sector_switch_confirm_frames`
  - 目标切换需要连续确认多少帧。
- `planner.cue_sector_switch_min_score_delta`
  - 新旧目标分数至少差多少才允许切换。

建议顺序：

1. 先确认球杆方向已经稳定。
2. 再调 `cue_sector_corridor_width_px`。
3. 如果目标还会闪，优先调 `cue_sector_lock_margin_px` 和 `cue_sector_lock_release_frames`。
4. 最后再调目标切换确认帧数和最小分差。

## 本地验证

本次修改已通过以下检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile bas\planning\cue_sector.py bas\config.py bas\user_settings.py tests\test_planner.py
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py tests\test_cue_aim.py tests\test_cue_sector_preview.py tests\test_ui_runtime.py -q --basetemp=pytest_tmp_codex\cue_lock
```

## .gitignore 约定

本项目默认不提交本地运行产物，常见目录和文件例如：

- `local_settings/`
- `logs/`
- `outputs/`
- `replays/`
- 模型权重、视频、图片和临时文件

新增大文件、录像、日志目录时，记得同步维护 `.gitignore`。
