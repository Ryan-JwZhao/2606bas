# BAS 使用说明

## 简介

BAS 是一个台球辅助系统，主链路包含采集、检测、跟踪、状态机、规划和投影。

当前仓库里建议优先使用 `modern` 状态机。最近这轮重构重点收口了“进洞确认 / ledger 提交 / 胜负动画”这条链路，核心语义如下：

- 只有真正允许修改 `ledger`、切换花色、触发胜负状态和动画的事件，才会发 `POCKET_CONFIRMED`
- 洞口 `mouth` 级别证据只会进入 `candidate / tentative / review_required`，不会再直接确认
- `POCKET_COMMIT_READY` 是内部提交态，表示 pocket 证据已经够强，但还要等 `TURN_RESOLVE` 统一 reconcile 后才能真正提交
- 只要该杆存在 `review_required`、可见球和 ledger 冲突、近袋口重现冲突，整杆都会冻结，不会污染全局状态
- 黑八也走同一套提交栅栏；只有 clean confirm 才允许发 `GAME_STATUS_CHANGED` 和 `GAME_OVER_CANDIDATE`

## 模块结构

- `bas/capture`：相机与视频输入
- `bas/perception`：目标检测
- `bas/tracking`：跨帧跟踪
- `bas/state`：状态机、进洞判定、裁判意图、ledger 与复核接口
- `bas/planning`：走位与击球规划
- `bas/projection`：投影叠加与动画
- `bas/ui`：桌面控制台
- `tests`：单元测试、回放测试、交互测试

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

## 进洞 / 胜负状态语义

### 关键事件

- `POCKET_CANDIDATE`：检测到近袋口候选
- `POCKET_TENTATIVE`：候选暂挂，继续等缺失窗口或重现冲突
- `POCKET_COMMIT_READY`：内部提交态，外层还不能播动画
- `POCKET_CONFIRMED`：该杆在 `TURN_RESOLVE` 后 clean commit，允许真正记分和播动画
- `POCKET_REVIEW_REQUIRED`：证据不足或发生冲突，需要人工复核
- `POCKET_REJECTED`：候选被否决，例如同袋口近距离重现
- `GAME_STATUS_CHANGED`：比赛状态真正发生变化，只在提交态发出

### 当前判定规则

- `mouth + inward + disappear`：最多到 `candidate -> tentative -> review_required`
- 只有 `crossed_throat` 或 `entered_interior` 才有资格进入自动确认链路
- 同袋口、近距离、窗口期内任意重现都会否决原 pocket
- 如果重现时 `track_id` 或 `group` 变化，会作为复核证据保留下来
- 投影层只消费提交态事件，因此 `tentative` / `review_required` 不会再触发自动进球或胜利动画

## 人工复核

桌面控制台里现在提供三类显式决议：

- `确认复核`：确认冻结中的 pocket，提交本杆
- `驳回复核`：驳回冻结中的 pocket，保持冻结前 ledger 与胜负状态
- `确认开台花色`：开台同时进了不同花色时，显式指定本杆花色归属

不再建议使用“只清标记不修状态”的思路。复核动作必须同时完成状态决议和全局状态修正。

## Pocket Replay

仓库内置了 5 个 pocket trace fixture，可直接回放：

```powershell
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_mouth_review.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_true_confirm.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_black_victory_once.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_track_id_churn_reject.json
.\.venv\Scripts\python.exe -m bas pocket-replay tests/fixtures/pocket_trace_group_flip_review.json
```

这几类 fixture 分别覆盖：

- 洞口晃动后消失，只能进复核
- 真正跨 throat / interior 的正常进球
- 黑八 clean confirm 只触发一次胜负状态变化
- 同袋口换 `track_id` 重现，原 pocket 应被 reject
- 同袋口近距离跨组重现，整杆应冻结并要求复核

## 测试

推荐先跑和这次重构直接相关的测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_state_modern.py tests\test_state_replay.py tests\test_projection_interaction.py tests\test_ui_runtime.py -q --basetemp=.\pytest_tmp_codex\pocket_state_refactor -p no:cacheprovider
```

完整回归：

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests -q --basetemp=.\pytest_tmp_codex\full_state_refactor -o cache_dir=.\pytest_cache_local\full_state_refactor
```

## 仓库说明

- 大文件、日志、回放产物不要直接入库，相关目录应加入 `.gitignore`
- 调试脚本尽量做成独立模块或 CLI 子命令，方便复现与排查
- 排查 pocket / win 问题时，优先看 replay、`SHOT_CONTEXT_FINALIZED`、`REFEREE_INTENT` 和 `pending_review`
