# BAS 使用说明

## 简介

BAS 是一个台球辅助系统，主要能力包括：

- 采集相机或回放视频画面
- 检测球、球杆和台面几何信息
- 跟踪球体轨迹并维护回合状态
- 规划推荐击球路线、目标球和进袋线
- 将规划结果投影或叠加到预览界面

当前仓库默认围绕本地调试、视频回放和桌面控制台使用场景组织，建议优先使用 `modern` 状态机和现有回放工具做联调与回归。

## 目录结构

- `bas/capture`：相机、视频输入与采集服务
- `bas/perception`：检测器、检测区域与感知逻辑
- `bas/tracking`：跨帧跟踪
- `bas/state`：回合状态、进球判定与裁决逻辑
- `bas/planning`：路线规划、target shot、free shot、cue sector 等模块
- `bas/projection`：投影叠加与渲染
- `bas/ui`：桌面控制台
- `scripts`：本地调试、基准和辅助脚本
- `tests`：单元测试与回归测试

## 环境准备

优先使用仓库内置脚本安装依赖：

```powershell
Setup_Environment.cmd
```

如果已经有虚拟环境，也可以直接按需安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动方式

启动桌面控制台：

```powershell
Start_BAS.cmd
```

或直接使用 Python 命令启动 UI：

```powershell
.\.venv\Scripts\python.exe -m bas ui
```

无界面运行主流程：

```powershell
.\.venv\Scripts\python.exe -m bas run --headless --max-frames 90
```

检查当前可用相机：

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
```

## 主控台界面

当前主控台围绕“三栏 + 中间 16:9 预览”组织：

- 左侧为运行控制、采集设置、显示与模式、现场抓取、交互调试。
- 中间为实时预览，以及最佳路线与候选路线列表。
- 右侧为模块状态、状态机人工介入、复核、事件与日志。

常用操作入口如下：

- `开始采集`：启动或停止实时采集链路。
- `开始投影`：打开或关闭投影窗口。
- `抓拍原图`：保存当前无画线校正照片。
- `导出纯净回放`：导出前 60 秒纯净视频回放。
- `开始原始录制` / `开始路线录制`：分别录制无画线视频和进洞路线画线视频。

界面改动后，左侧和右侧都支持滚动，交互调试与复核区域支持折叠。若窗口缩小，预览区域会继续保持 16:9 比例，避免黑边占满整块布局。

## 测试与回归

执行规划相关回归：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py tests\test_route_geometry.py tests\test_cue_sector_preview.py tests\test_user_settings.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

如果只想快速检查核心规划链路，也可以运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

## 最近一次规划修复说明

这次修复聚焦于“矩形框轻微歪斜后进洞线消失”问题，核心做法是统一 `target_lock`、`target_shot` 与 rule 路线在目标选择和直线进袋判定上的行为：

- 抽出共用的走廊目标选择逻辑，严格以“球心在走廊内”为命中标准
- 让 `target_lock` 和 `target_shot` 复用同一套目标排序规则，避免擦边球抢锁定
- 将零库直线进袋判定切到 center playable polygon 语义，并复用现有袋口放宽规则
- 保持 bank/rebound、评分、时序参数和 UI 配置项不变，避免修复范围外扩

## 仓库约定

- 本地大文件、日志、回放、模型和缓存目录不要提交，相关路径应维护在 `.gitignore`
- 调试优先做成独立模块、脚本或测试，避免把临时代码堆进主链路
- 复现和排查问题时，优先依赖 `tests`、`scripts` 和回放素材，保证结论可重复验证
