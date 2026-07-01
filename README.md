# BAS 使用说明

## 简介

BAS 是一个台球辅助系统，运行链路包含采集、检测、跟踪、状态机、规划和投影。

这次重构重点修了 `modern` 状态机里的进洞判定与结算链路：

- 进洞不再靠单次消失直接确认，而是改成“证据累计 -> tentative -> confirmed / review / rejected”两阶段判定。
- `ledger`、花色推进、黑八结算只消费 `POCKET_CONFIRMED`，不会再被临时误判直接污染。
- 胜利动画只在 `GAME_STATUS_CHANGED` 时触发一次，不再因为后续 `TURN_RESOLVE` 重复播放。
- 调试输出增加了 pocket evidence，能直接看到为什么判进、为什么没判进、为什么要求复核。

`legacy` 状态机仍然保留，当前增强只落在 `modern` 链路。

## 模块结构

- `bas/capture`：相机与视频输入
- `bas/perception`：目标检测
- `bas/tracking`：时序跟踪
- `bas/state`：状态机、进洞判定、裁判意图、结算
- `bas/planning`：走位与目标球规划
- `bas/projection`：投影叠加与动效
- `bas/ui`：桌面控制台
- `tests`：单元测试、集成测试、回放 fixture

## 快速开始

1. 安装依赖：

```powershell
Setup_Environment.cmd
```

2. 打开桌面控制台：

```powershell
Start_BAS.cmd
```

3. 直接用 Python 启动：

```powershell
.\.venv\Scripts\python.exe -m bas ui
```

4. 无界面跑主流程：

```powershell
.\.venv\Scripts\python.exe -m bas run --headless --max-frames 90
```

5. 检查相机：

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
```

6. 检查标定：

```powershell
.\.venv\Scripts\python.exe -m bas inspect-calib
```

## 进洞判定调试

### 事件语义

- `POCKET_TENTATIVE`：近袋消失，先挂起，不落账
- `POCKET_CONFIRMED`：证据成立，正式提交进球
- `POCKET_REVIEW_REQUIRED`：证据不足或洞口晃动，需要人工复核
- `POCKET_REJECTED`：重现、回桌或其它否决条件成立
- `GAME_STATUS_CHANGED`：比赛状态真正发生变化，只发一次

### 调试输出重点

`modern` 状态调试里现在会输出每颗候选球的 pocket evidence，重点看这些字段：

- `zone`
- `inward_speed_mm_s`
- `candidate_reason`
- `missing_ms`
- `reappear_veto`
- `decision`
- `reason_codes`

如果出现“洞口晃一下就播进球动画”之类的问题，先看该球最后是 `review_required` 还是 `confirmed`，再看 `reason_codes`。

### 常见判定规则

- `mouth` 区静止后直接消失，默认不会提交，只会进入复核
- 只有跨过 `throat` / 进入 `interior` 且 missing 持续成立，才会 `POCKET_CONFIRMED`
- 同袋口附近短时间重现，即使换了 `track_id`，也会否决旧 tentative
- 仅 `occluded` 几帧但未真正持续 lost，不会立刻提交最终进球

## 回放与复现

### 小型 fixture 回放

仓库里内置了 3 个 pocket trace fixture，可直接跑：

```powershell
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_mouth_review.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_true_confirm.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_black_victory_once.json
```

回放输出会包含：

- 逐帧事件序列
- 最终 `rule_state`
- 最后一次 `SHOT_CONTEXT_FINALIZED`
- 最后一次 `REFEREE_INTENT`
- pocket debug 快照

### 现有 replay 摘要

如果只是想先看已有录制的 `events.jsonl` 里有什么类型的事件：

```powershell
.\.venv\Scripts\python.exe -m bas replay-summary replays\your_session\events.jsonl
```

## 关键参数

状态机相关参数在 `state` 配置段：

- `pocket_confirm_missing_ms`：tentative 持续多久后允许最终决策
- `pocket_reappear_window_ms`：近袋消失后，多久内的重现会否决 tentative
- `turn_resolve_grace_ms`：`TURN_RESOLVE` 遇到 pending pocket 时最多等待多久
- `observation_reconcile_stable_frames`：YOLO 可见计数要稳定多少帧才参与复核

不建议再靠单纯调大/调小这些阈值修 bug。阈值只负责微调，正确性主要依赖事件链路和证据逻辑。

## 测试

完整回归：

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q --basetemp=.\pytest_tmp_codex\full_state_refactor -o cache_dir=.\pytest_cache_local\full_state_refactor
```

本次重构新增和更新了：

- pocket FSM 单元测试
- `modern` 结算与黑八状态测试
- 动画触发测试
- 学习采集测试
- pocket trace 回放回归测试

## 仓库说明

- 调试产物、日志、大文件不要直接入库，相关目录应加入 `.gitignore`
- 推荐优先改 `modern` 链路，`legacy` 仅作为回退
- 需要新增调试脚本时，尽量做成独立模块或 CLI 子命令，方便复现和排查
