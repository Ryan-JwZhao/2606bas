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

## 2026-06-30 本次更新

### 用户意图级目标球锁定

规则模式现在新增 `target_lock` 用户意图锁定层。它位于球杆方向识别和路线规划之间，优先级高于普通候选评分和球杆走廊纠正。

核心行为：

- 球杆连续指向同一颗目标球后，系统会把这颗球作为用户意图目标锁定。
- 锁定后，伸缩球杆、短暂无球杆、瞄准抖动、出杆运动阶段都不会切换到其他球。
- 只有球杆连续、明确指向另一颗距离足够远的球，才会提交换锁。
- 锁定目标会写入 `ShotPlan.locked_target_id` 和 `ShotPlan.target_lock_status`，便于前端和调试回放确认状态。

相关实现：

- `bas/planning/target_lock.py`
- `bas/planning/planner.py`
- `bas/schemas.py`
- `bas/state_debug.py`

主要参数：

- `planner.target_lock_enabled`
  - 是否启用用户意图级锁定，默认启用；可在“设置 / 路线策略 / 用户意图目标锁定”调整。
- `planner.target_lock_confirm_frames`
  - 首次锁定同一目标需要连续确认多少帧，默认 3；可在设置窗口调整。
- `planner.target_lock_switch_confirm_frames`
  - 从已锁定目标切换到另一颗球需要连续确认多少帧，默认 8；可在设置窗口调整。
- `planner.target_lock_missing_release_frames`
  - 锁定球在非运动阶段连续丢失多少帧后释放，默认 45。
- `planner.target_lock_corridor_width_px`
  - 用于选球入锁的球杆走廊宽度，默认 140。
- `planner.target_lock_switch_min_distance_px`
  - 允许换锁的新目标与旧目标锚点的最小距离，默认 70。

### 目标击球模式

规则模式现在新增自动触发的目标击球模式。球杆指向白球时仍是正常击打状态；球杆连续指向非白球达到阈值后，系统进入目标击球模式，强制锁定该目标球。

目标击球模式的路线规划不使用球杆方向。它只基于白球、目标球、库边和袋口几何，枚举允许次数内的反弹路线，计算理论上可由白球撞击目标球后进袋的最佳路线。如果没有理论可行路线，则保持锁定状态但不显示路线。

核心行为：

- 默认连续指向非白球 15 帧后触发；可在“设置 / 路线策略 / 目标击球模式”调整。
- 目标击球模式激活后，锁定不会因为短暂无球杆、指回白球或普通抖动而解除。
- 只有白球进入击出阶段，或连续指向另一颗非白球达到阈值，才会释放或切换目标击球锁定。
- 投影中白球到虚拟撞击点为实线，目标球到袋口的多库路线为虚线。

相关实现：

- `bas/planning/target_shot.py`
- `bas/planning/planner.py`
- `bas/projection/overlay.py`

主要参数：

- `planner.target_shot_enabled`
  - 是否启用目标击球模式，默认启用。
- `planner.target_shot_trigger_frames`
  - 连续指向非白球多少帧后进入目标击球模式，默认 15；可在设置窗口调整。
- `planner.target_shot_max_rebounds`
  - 目标球路线最多允许几次库边反弹，默认 2。

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
.\.venv\Scripts\python.exe -m py_compile bas\planning\target_lock.py bas\planning\planner.py bas\schemas.py bas\state_debug.py bas\ui\main_window.py
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py tests\test_route_freeze.py tests\test_ui_runtime.py tests\test_state_debug.py -q --basetemp=pytest_tmp_codex\target_lock
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
