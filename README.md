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

这次不再把球杆方向问题当成简单抖动补丁处理，而是把它改成了一个独立的方向解析步骤：

- 先把 `cue_stick` 检测结果视为一条“无方向的球杆轴”。
- 再以白球中心为锚点，对这条轴的两个相反方向分别打分。
- 只保留“球杆本体主要位于白球后方”的那个方向。
- 这样在蓄力时，哪怕球杆前后抽动，只要识别到的还是同一条杆轴，就不会因为端点前后交换而切到反向路线。

实现位置：

- `bas/planning/cue_direction_resolver.py`
- `bas/planning/cue_aim.py`

### 2. 球杆矩形候选框只在前端显示

主界面新增了 `显示球杆矩形候选框` 开关。

用途：

- 在前端预览里显示球杆走廊和候选目标球，方便排查识别和纠正范围。
- 仅在前端显示，不会进入投影 overlay，不会打到球台上。

相关位置：

- `bas/ui/cue_sector_preview.py`
- `bas/ui/main_window.py`

## 调参建议

如果现场仍然需要微调，优先看这几个参数：

- `planner.cue_sector_corridor_width_px`
  - 控制球杆矩形走廊总宽度。
- `planner.cue_sector_switch_confirm_frames`
  - 控制目标切换需要连续确认多少帧。
- `planner.cue_sector_switch_min_score_delta`
  - 控制新旧目标分数至少差多少才允许切换。
- `planner.cue_sector_min_stick_quality`
  - 控制球杆检测最低质量门槛。

建议顺序：

1. 先确认球杆方向是否已经稳定。
2. 再调矩形走廊宽度。
3. 最后才调切换确认帧数和最小分差。

## 本地验证

本次修改已通过以下检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile bas\planning\cue_direction_resolver.py bas\planning\cue_aim.py bas\ui\main_window.py tests\test_cue_aim.py tests\test_planner.py tests\test_ui_runtime.py
.\.venv\Scripts\python.exe -m pytest tests\test_cue_aim.py tests\test_planner.py tests\test_cue_sector_preview.py tests\test_ui_runtime.py -q --basetemp=pytest_tmp_codex\cue_direction
```

## .gitignore 约定

本项目默认不提交本地运行产物，常见目录和文件例如：

- `local_settings/`
- `logs/`
- `outputs/`
- `replays/`
- 模型权重、视频、图片和临时文件

新增大文件、录像、日志目录时，记得同步维护 `.gitignore`。
