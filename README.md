
# BAS 本地说明
## 2026-06-28 球杆扇形二次纠正
- 新增规则模式下的“球杆扇形二次纠正”：当系统没有检测到有效球杆，或球杆没有指向白球时，进洞路线继续使用原有算路逻辑，不做额外干预。
- 当有效球杆进入 outline 并指向白球后，系统会以“球杆方向穿过白球中心”的方向作为中心线，按设置中的总夹角自动左右平分形成扇形。默认总夹角 `15°`，即左右各 `7.5°`。
- 在扇形生效时，推荐目标必须位于扇形内；贴近扇形两侧边界的球会按“边缘剔除”参数直接排除，减少路线在边界附近频繁闪烁。
- 如果当前回合是纯色/花色：
  - 扇形内有己方球时，只在己方球里按进洞评分推荐。
  - 扇形内没有己方球、但有黑球时，推荐黑球。
  - 扇形内只有对方球时，临时推荐对方球，并进入二次确认：出杆后的第一碰如果确实是这颗球，才把状态机回合目标更新为该球组；如果第一碰不是它，则不修改状态机，下一杆仍按原回合重新判断。

### 使用方式
1. 启动程序：

```powershell
python -m bas
```

2. 使用规则模式画线。
3. 打开“设置 -> 基础 -> 二次纠正”，可调整：
   - `启用球杆扇形二次纠正`
   - `扇形总夹角(度)`：填写总夹角，系统自动左右平分。
   - `边缘剔除(度)`：用于剔除靠近两侧边界的球，降低闪烁。
   - `切换确认连续帧` / `切换最小分差`：用于抑制两个候选目标评分接近时的来回跳变。
4. 参数会保存到 `local_settings/user_settings.json`，下次启动自动恢复；本地设置、日志、回放和大文件已经由 `.gitignore` 排除。

## 2026-06-28 新旧台球状态机切换
- 系统现在同时保留两套台球状态机：
  - `状态机（旧）`：继续使用原有时序状态机，作为默认选项，保证现场台球辅助系统不受重构影响。
  - `状态机（新）`：基于 `example/0628_状态机规划.md` 重写，拆分为时序相位、单球进洞确认、整杆证据聚合、库存账本和裁判接口层。
- 新状态机的核心变化：
  - 进洞不再由单帧消失直接提交，而是经过袋口几何、多帧缺失、重现撤销和袋口停顿窗口确认。
  - 纯色、花色、黑球剩余数来自确认进洞后的单调库存账本，不再由当前可见球数直接决定。
  - `turn_target_group` 只作为下一杆规划提示；组别归属、自由球范围、犯规标记和复核需求会进入结构化裁判接口。
  - 先进智能裁判字段已经预留，包括 `foul_flags`、`ball_in_hand_scope`、`review_required`、`group_choice_required` 和 `GAME_OVER_CANDIDATE`，但暂不作为强制裁判文案输出。

### 使用方式
1. 启动程序：

```powershell
python -m bas
```

2. 在主界面左侧“状态机”下拉框选择 `状态机（旧）` 或 `状态机（新）`。
3. 也可以打开“设置 -> 基础 -> 台球状态机”选择同样选项。
4. 正在采集时切换状态机会自动重启采集管线；设置会写入 `local_settings/user_settings.json`，下次启动自动恢复。
5. 现场调试建议先用 `状态机（旧）` 保持稳定使用，再切到 `状态机（新）` 开启“深入调试”对比 `POCKET_CONFIRMED`、`REFEREE_INTENT`、库存账本和球局结束候选事件。

## 2026-06-28 新状态机进球后花色判断优化
- 新状态机现在会在 `TURN_RESOLVE` 到来时检查袋口是否仍有未决进洞候选；若存在，会先输出 `TURN_RESOLVE_DEFERRED`，等待候选完成确认/撤销或达到宽限时间，再输出 `TURN_RESOLVE_COMMITTED` 与 `REFEREE_INTENT`。
- 这样可以避免“球刚消失但 `POCKET_CONFIRMED` 还没来，回合已经提前结算”的问题，尤其针对进球后下一杆花色判断。
- 新增 YOLO 长期观察一致性融合：当前可见纯色/花色/黑球数量不会单帧改写账本，但如果连续多帧与事件账本不一致，会生成 `LEDGER_OBSERVATION_MISMATCH`，并把 `effective_remaining` 提供给本次裁判判断。
- 当 YOLO 长期看到的球数 **多于** 账本剩余数时，会优先防止错误升黑球；当 YOLO 长期看到的球数 **少于** 账本剩余数时，必须同时存在进洞候选、未确认消失等事件支持，才会参与本次判断。

### 调参入口
- `state.turn_resolve_grace_ms`：进洞候选未决时，回合结算最多等待多久，默认 `900ms`。
- `state.observation_reconcile_stable_frames`：YOLO 数量需要连续稳定多少帧才参与综合判断，默认 `12`。
- `state.observation_reconcile_min_quality` / `state.observation_reconcile_min_confidence`：参与数量融合的 track 质量与检测置信门槛。
- `state.observation_reconcile_infer_missing_with_event`：是否允许“YOLO 长期少于账本 + 有事件支持”时推断账本偏高，默认开启。

## 2026-06-27 深入调试模式
- 右侧“状态机人工介入 -> 目标状态（STABLE_IDLE 所在框）”里新增“开始深入调试”按钮。
- 开启后会自动联动三件事：
  - 开始无画线视频；
  - 开始进洞路线视频；
  - 开始逐帧抓取状态机排查信息。
- 逐帧抓取内容包含：
  - 当前 phase、状态机 confidence；
  - 每帧事件列表与关键 payload；
  - 候选路线、best candidate、显示中的路线信息；
  - 状态机内部计数与判定信号（`stable/moving/armed/settle/anomaly`、`cue_stick_seen`、`cue_motion`、`shot_start_voted` 等）。
- 停止深入调试后，会在 `local_settings/deep_debug/state_debug_时间戳/` 下导出：
  - `state_machine_frames.jsonl`：完整逐帧原始数据；
  - `state_machine_timeline.srt`：可直接和视频画面对时的字幕时间轴；
  - `session_summary.json`：本次调试摘要、事件计数、产物路径。

### 使用方式
1. 正常开始采集。
2. 在右侧 `STABLE_IDLE` 所在的“目标状态”区域点击“开始深入调试”。
3. 打一杆或复现场景。
4. 再次点击“停止深入调试”。
5. 到 `local_settings/deep_debug/` 查看本次会话目录，并将 `state_machine_timeline.srt` 与同批次视频一起对照。

### 说明
- 深入调试模式开启时，会锁住单独的“无画线视频 / 进洞路线视频”按钮，避免录制状态被手工打断。
- 新产物全部落在 `local_settings/` 下，默认不会进入 Git。

## 2026-06-27 前 60 秒纯净视频回放
- 新增“1 秒分段的 H.264 环形缓存 + 一键导出前 60 秒纯净视频”能力，缓存默认持续保留最近 `120` 秒原始相机画面，超过窗口后自动淘汰旧分段。
- 缓存录制与导出都基于无画线相机画面，不会把进洞路线、调试线或其它画线内容写进视频，导出结果为纯净视频。
- 录制链路会优先使用采集阶段已经完成 OpenCV 畸变校正的画面；如果运行时输入帧尚未预校正，但本地配置了 OpenCV 标定文件，也会在写入缓存或视频前补做一次校正，确保导出与录像统一使用校正后画面。
- 当采集 backend=`video` 时，系统会默认把输入视频视为“已经完成去畸变”的画面，不会再额外叠加一层 OpenCV 畸变校正；这个模式适合回放系统里已经校正过的录屏或录像。
- 如果当前既没有预校正输入帧，也没有可用的 OpenCV 标定文件，录制与回放缓存不会偷偷回退到原始画面，而是直接拒绝写入并在日志里提示。
- 主界面“现场抓取”区新增“导出前60秒纯净视频”按钮；运行中点击后会把最近 `60` 秒缓存片段导出为独立 MP4 文件，不参与后续覆盖。
- 为避免误触和重复导出，新增 `30` 秒冷却限制：30 秒内连续多次触发只响应第一次，后续请求会被拒绝，直到冷却结束。
- 新增 Stream Deck / 外部按键控制命令：
  - `python -m bas.cli remote-control save-retro-clip`
  - `scripts\BAS_SD_Save_Retro_Clip.cmd`
- 默认参数位于 `configs/default.yaml` 的 `instant_replay` 段：
  - `segment_seconds: 1`
  - `buffer_seconds: 120`
  - `export_seconds: 60`
  - `cooldown_seconds: 30`
  - `bitrate_kbps: 6000`

## 2026-06-27 运动时冻结路线
- 新增“运动时冻结路线”功能：当状态机判断球局进入运动阶段后，前端预览和投影会优先保持上一份稳定路线，避免进洞线在击球后持续闪烁。
- 该功能可在“设置 -> 基础 -> 路线防闪烁”中手动开启或关闭，修改后会保存到 `local_settings/user_settings.json`。
- 目前提供以下可量化参数，便于现场调参：
  - `进入冻结连续帧`：连续多少帧判定为运动后开始冻结路线。
  - `解冻连续稳定帧`：连续多少帧回到稳定状态后恢复实时路线。
  - `同路线刷新位移(mm)`：同一目标球/袋口方案下，球位抖动小于该阈值时继续沿用旧路线。
  - `同路线刷新分差`：同一方案分数波动小于该阈值时不刷新路线。
  - `切换确认连续帧`：新方案需要连续出现多少帧才允许切换。
  - `切换最小位移(mm)` / `切换最小分差`：只有当新旧路线差异足够明显时才真正切换。
- 建议起步参数：`进入冻结 2`、`解冻 8`、`同路线 12mm`、`切换确认 3`。如果现场仍闪，可优先增大“解冻连续稳定帧”和“切换确认连续帧”。

## 2026-06-27 线宽微调
- 规则击球模式和自由击球模式的实线改为“颗星公式”线宽的 `2 倍`，虚线改为 `1.5 倍`。
- 这次调整同时覆盖前端预览和投影 overlay，避免两个显示界面的线宽观感不一致。
- 线路样式现在走共享线宽计算；球圈描边仍保持颗星公式同级基准，后续如果继续微调颗星公式基准，规则/自由模式也会一起同步。

## 简介
这是一个台球辅助系统，包含实时采集、检测、跟踪、状态机、路线规划、前端预览和投影显示。

本次更新重点补了两类能力：

- 修正 `inline` 与 `pocket` 的组合几何显示与边界生成逻辑，避免每个袋口被单独闭合成“月牙”。
- 保留原有实体校正板流程，同时新增“一键联动校正”，用多图样混合校正投影仪，其中以 ChArUco 为主、编码网格为辅。
- 新增运行时“投影调试模式”，用于现场肉眼检查投影映射是否与台面几何和识别球位置一致。

## Stream Deck 控制接口

### 设计说明

- 本次没有加全局热键，也没有注入其它程序。
- 接口采用“本地命令队列”方案：Stream Deck 只负责执行一个本地命令，BAS 主界面会轮询 `local_settings/stream_deck/queue/` 并执行收到的控制指令。
- 这样做的好处是改动小、冲突少，不会抢占系统快捷键，也不会影响其它软件。

### 使用方式

1. 先正常启动 BAS 主界面：

```powershell
python -m bas
```

2. 在 Stream Deck 的按键动作里，调用下面任意一种本地命令：

```powershell
python -m bas.cli remote-control start-capture
python -m bas.cli remote-control start-projection
python -m bas.cli remote-control toggle-target-group
python -m bas.cli remote-control toggle-shot-mode
python -m bas.cli remote-control free-shot-once
python -m bas.cli remote-control black-shot-once
python -m bas.cli remote-control toggle-star-formula
python -m bas.cli remote-control save-retro-clip
```

3. 如果 Stream Deck 适合直接绑定固定 `.cmd` 文件，可以使用下面这些独立脚本：

```powershell
scripts\BAS_SD_Start_Capture.cmd
scripts\BAS_SD_Start_Projection.cmd
scripts\BAS_SD_Toggle_Target_Group.cmd
scripts\BAS_SD_Toggle_Shot_Mode.cmd
scripts\BAS_SD_Free_Shot_Once.cmd
scripts\BAS_SD_Black_Shot_Once.cmd
scripts\BAS_SD_Toggle_Star_Formula.cmd
scripts\BAS_SD_Save_Retro_Clip.cmd
```

4. 如果你之后还想扩展额外动作，仍然可以继续使用通用脚本：

```powershell
scripts\BAS_StreamDeck_Command.cmd start-capture
scripts\BAS_StreamDeck_Command.cmd start-projection
scripts\BAS_StreamDeck_Command.cmd toggle-target-group
scripts\BAS_StreamDeck_Command.cmd toggle-shot-mode
scripts\BAS_StreamDeck_Command.cmd free-shot-once
scripts\BAS_StreamDeck_Command.cmd black-shot-once
scripts\BAS_StreamDeck_Command.cmd toggle-star-formula
scripts\BAS_StreamDeck_Command.cmd save-retro-clip
```

### 支持的指令

- `start-capture`：开始采集；如果已经在采集，则忽略。
- `start-projection`：开始投影；如果已经在投影，则忽略。
- `toggle-target-group`：切换当前杆目标花色；`纯色 -> 花色`，`花色 -> 纯色`，未指定时默认切到 `纯色`。
- `toggle-shot-mode`：规则击球模式 / 自由击球模式 全局切换。
- `free-shot-once`：只把当前这一杆临时切为自由击球；到底层状态机完成本杆结算后自动恢复。
- `black-shot-once`：只把当前这一杆临时指定为黑球目标，仅在规则击球模式下生效。
- `toggle-star-formula`：颗星公式开关切换。
- `save-retro-clip`：导出最近 60 秒无画线纯净视频，并受 30 秒冷却保护。

### 说明

- `free-shot-once` 和 `black-shot-once` 都是“单杆临时覆盖”，不会改掉你平时的全局模式。
- `black-shot-once` 如果当前全局已经切到自由模式，会被忽略。
- 命令是写入本地队列文件，不依赖焦点窗口；BAS 主界面开着即可响应。

## 几何与袋口说明

- `outline.json` 表示台球桌外轮廓矩形范围。
- `inline.json` 表示台面内沿线段。
- `pocket.json` 表示六个袋口的开口曲线。
- 程序现在会把 `inline + pocket` 自动拼成一条真实的组合闭合边界，用于后续标定与重点区域联动。
- 前端参考线中，`pocket` 会按开口曲线显示，不再单独闭合成封闭月牙。

## 投影调试模式

### 作用

- 开启后，投影窗口会额外显示 `inline` 线段和 `pocket` 开口曲线，仍然不会投射 `outline`。
- 开启后，会为当前识别并跟踪到的每一颗球投射一个圆圈和中心点，圆心和半径都通过当前映射关系从相机画面换算到投影坐标；如果当帧还没形成稳定 track，会先回退使用 detection 结果画出临时球圈。
- 该模式只用于现场排查轻微校准偏移，便于观察路线是否穿过球心、袋口和内沿是否对齐。

### 使用方法

1. 启动程序并开始采集。
2. 打开投影窗口。
3. 在主界面左侧勾选“投影调试模式”。
4. 观察投影到台面上的绿色 `inline`、橙色 `pocket` 和紫色球圈。
5. 取消勾选后，调试线和球圈会立即从投影中消失，正常进洞路线继续按原逻辑显示。

说明：调试模式不会保存到 `user_settings.json`，每次启动程序都默认关闭，需要现场手动开启。

## 一键联动校正

### 作用

- 自动播放总览网格、全域 ChArUco、中心细化和六个袋口重点图样。
- 让投影范围完整覆盖 `outline` 对应的矩形区域。
- 对 `inline` 与 `pocket` 相接的关键区域做重点采样。
- 自动识别 ChArUco 角点，自动求解 `camera_px -> projector_px` 的投影校正。
- 自动保存结果，并可再次加载与可视化显示。
- 主界面仅保留“校正投影仪”入口；“一键联动校正”保留在二级校正菜单中，避免重复入口。

### 前提

- 建议先准备好工业相机畸变标定文件。
- 联动校正会优先复用 `camera.distortion_correction_file`，尽量先消除工业相机内部畸变影响。
- 如果 `calibration.camera_file` 为空，程序会自动把畸变标定文件同步成联动校正的相机标定输入。

### 使用方法

1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

2. 启动程序

```powershell
python -m bas
```

3. 在“设置”中确认以下路径和参数

- `outline.json`
- `inline.json`
- `pocket.json`
- 工业相机畸变标定文件
- 投影分辨率与投影屏幕编号

4. 选择一种标定方式

- 实体板流程：点击“校正投影仪”，继续使用原有人工/实体板校正逻辑。
- 自动流程：点击“一键联动校正”，程序会自动停掉当前采集、独占相机、播放图样、自动识别并保存结果。

5. 校正完成后

- 可直接在投影窗口查看当前校正结果。
- 可查看残差箭头，观察局部误差方向。
- 可继续用 Holdout 数据做独立验收。

## 本地验证

本次修改已通过以下测试：

```powershell
.\.venv\Scripts\python.exe -m py_compile bas\geometry.py bas\ui\geometry_reference.py bas\calibration\projector.py bas\calibration\linked.py bas\ui\main_window.py tests\test_geometry_reference.py tests\test_linked_calibration.py
.\.venv\Scripts\python.exe -m pytest tests\test_geometry_reference.py tests\test_linked_calibration.py tests\test_calibration.py -q --basetemp=pytest_tmp_codex\linked
```

## 说明

- 当前 `calibration.projection_file` 仍可作为加载入口使用。
- 每次联动校正成功后，程序会自动在 `2606BAS/local_settings/calibrations/` 下生成一个带时间戳的新文件名，例如 `projection_calibration_20260625_120102.json`，然后立即切换并加载这份最新校正结果。
- 如果没有配置投影校正输出路径，程序会回落到 `local_settings/projection_calibration_linked.json`。
- `local_settings/`、日志、回放、临时目录和大文件应保持在 `.gitignore` 中，避免提交运行产物。

## 本地依赖位置

- Nori SDK: `example/Nori_Xvision_Development_Kit_Ver10.00.06_Windows/`
- 检测模型与类别文件: `example/parameters/`
- 相机畸变与标定文件: `calib/`
- 当前导入的投影校正文件: `presets/projection_calibration.json`

## 设置勾选框显示

- 设置弹窗和主界面侧栏的复选框统一使用浅灰色方框，勾选后显示黑色勾号，避免在深色背景中看不清。
- 使用方法不变：启动程序后点击主界面“设置”，按需勾选或取消相关选项即可。

## 联动校正故障补充说明

- 已规避一类 OpenCV 4.13 的 ChArUco 生成问题：当局部 ROI 尺寸落在类似 `384x209` 的边界组合时，OpenCV 可能在 `board.generateImage()` 内部触发 `cv::Mat::Mat` 的 ROI 越界断言；程序现在会自动回退到安全尺寸生成并居中贴回原画布。
- 联动校正失败时，不会覆盖当前 `calibration.projection_file`，也不会保存半成品结果。
- 如果失败前配置里已经指向旧的投影校正文件，后续重新初始化模块或重新启动采集时，程序仍会按这个旧路径继续加载之前的校正结果。
- 针对“联动校正后个别角点被局部拉爆”的问题，求解器现在只会使用 `RANSAC` 内点构建 `residual_field`，不会再让错误匹配点直接污染局部插值；若采集过程中混入异常图样，误差统计也会直接暴露出来，便于现场回退到旧标定结果继续使用。

## 三套边界拆解

- 程序现在把原先混用的一条 `inner_polygon_mm` 拆成了三套边界：
  `projection_visible_polygon_mm` 用于投影显示裁切；
  `inner_polygon_mm` 现在作为物理库边边界；
  `center_playable_polygon_mm` 作为球心可达边界，供规则规划和自由击球碰撞使用。
- 边界由相机几何文件实时换算后再做分层内缩，不再直接把 `inline + pocket` 的拼接结果同时拿去做显示和物理判定。
- 当前可通过以下配置调节：
  `calibration.projection_visible_inset_top_mm`
  `calibration.projection_visible_inset_right_mm`
  `calibration.projection_visible_inset_bottom_mm`
  `calibration.projection_visible_inset_left_mm`
  `calibration.physical_rail_inset_top_mm`
  `calibration.physical_rail_inset_right_mm`
  `calibration.physical_rail_inset_bottom_mm`
  `calibration.physical_rail_inset_left_mm`
  `calibration.center_reachable_extra_margin_mm`
- 当前这台桌子的本地默认值已经写入 `local_settings/user_settings.json`：
  近投影长边的可见边界先内缩 `12mm`；
  物理库边四边统一内缩 `10mm`；
  球心可达边界会在物理库边基础上再内缩 `半个球径 + 2mm`。
## 2026-06-25 边界与投影偏移排查补充

- 物理库边边界和球心可达边界的内缩逻辑已经改成按边归属做平移，不再对整条边做软权重挤压。这样原本应该保持笔直的长边、短边会继续保持直线，避免靠近袋口时把整段边界带弯。
- `table_mm <-> projector_px` 现在优先使用 `table_polygon_proj` 四点透视映射，不再只按包围盒做线性缩放。对“越远离投影中心越偏”的圆圈和边界漂移，这一层系统误差已经被去掉。
- 如果调试模式下球圈在左右两侧、远端区域仍然明显偏移，优先检查三件事：
  1. 相机畸变文件是否和当前分辨率、当前取流链路一致。
  2. 投影标定采样是否覆盖左右两侧、远端、六个袋口，而不是只覆盖中间区域。
  3. `table_polygon_proj` 的四点顺序是否严格对应左上、右上、右下、左下。
- 现场建议的处理顺序：
  1. 先开投影调试模式，只看球圈圆心是否压在真实球心上。
  2. 如果球圈越到边缘越偏，先重做投影标定，再看三条边界；不要先靠手调边界 inset 掩盖标定误差。
  3. 如果圆圈已经准，但物理库边边界仍然太贴近可见边界，再增大 `calibration.physical_rail_inset_*_mm`。
  4. 如果物理库边合理，但球心可达边界仍偏外，再微调 `calibration.center_reachable_extra_margin_mm`。
## 2026-06-25 GUI 投影修正补充

- 设置窗口新增了“投影修正”页，可直接在 GUI 中调四类参数：
  1. 投影可见边界四边 inset。
  2. 物理库边四边 inset。
  3. 上中袋 / 下中袋 `relief`，用于让中袋附近的物理库边更贴近可见边界。
  4. 球心投影补偿，包括参考点 X/Y 和 X/Y 拉回比例。
- 球心补偿的含义是：把检测到的球心，按一定比例向参考点方向拉回，再参与台面映射和投影。这个补偿会同时影响调试球圈、规则规划和自由击球规划，不再只是调试显示。参考点表示“压缩/拉回要朝向的图像参考位置”，不是单独去模拟投影仪的物理坐标。
- 推荐现场调参顺序：
  1. 先开投影调试模式，只调球圈，让中间、左右两侧、远端的球圈先压到球心。
  2. 球圈稳定后，再调物理库边四边 inset。
  3. 如果只有上中袋和下中袋附近过分内缩，再增大对应的 `relief`。
  4. 最后再微调球心可达边界的额外安全边 `center_reachable_extra_margin_mm`。
- 推荐起步值：
  1. 默认勾选“使用画面中心作为参考点”，程序会自动以当前相机画面中心作为参考点。
  2. `X 拉回比例` 和 `Y 拉回比例` 可先从 `1.0%` 到 `3.0%` 开始试。
  3. 如果需要手动输入参考点，取消勾选自动模式后可直接填写 X/Y，支持负值，适合处理投影仪中心落在画面外的情况。
  4. 如果上中袋、下中袋应与可见边界几乎重合，可先把对应 `relief` 调到接近 `physical_rail_inset_top_mm / bottom_mm` 的数值。
- 交互行为补充：
  1. 在“投影修正”页里，参数一旦调整就会立刻刷新当前投影预览。
  2. 如果不点保存直接关闭设置窗口，这一页的改动会回滚到打开设置前的状态。
  3. 点击保存后，参数会写入 `local_settings/user_settings.json`，下次启动会自动加载。
## 2026-06-26 工业相机白平衡控制

- “设置”窗口里在“工业相机曝光”下面新增了“工业相机白平衡”。
- 新增两个组件：`自动白平衡` 勾选框、`白平衡色温` 滑块；滑块右侧会实时显示当前 Kelvin 数值。
- 勾选自动白平衡后，滑块会禁用；取消勾选后，可手动调整色温值。
- 若当前相机走 Nori SDK，设置窗口会优先读取 SDK 返回的白平衡 `min/max/step/current`，尽量按设备真实范围生成滑块。
- 保存设置后，程序会把白平衡模式和手动色温一并写入 `local_settings/user_settings.json`，并在下次启动 Nori 采集时自动下发，减少自动白平衡频繁跳变。

## 2026-06-26 黑球算路规则补充
- 规则模式下，黑球不再默认参与所有进洞路线候选，而是跟随当前回合目标球型。
- 如果当前是纯色回合且台面纯色球已清零，则该回合只对黑球进行算路与推荐。
- 如果当前是花色回合且台面花色球已清零，则该回合只对黑球进行算路与推荐。
- 如果纯色和花色都已清零，则后续所有回合都只对黑球进行算路与推荐，直到球局结束。

## 2026-06-26 投影校准双模式
- 设置页新增了“投影校准模式”，支持 `传统模式` 和 `工程模式`，选择会写入 `local_settings/user_settings.json`，下次启动自动恢复。
- `传统模式` 继续沿用现有 `projection_calibration.json + residual field + 球心拉回补偿` 的链路，方便保留旧流程和旧文件。
- `工程模式` 在平面校准文件之外，新增一份“工程球体补偿文件”，用于把球体问题和台布平面问题拆开处理。
- `工程平面校准文件` 和 `工程球体补偿文件` 必须是两份不同的 JSON，不能共用同一路径；否则球体补偿保存时会覆盖平面校准，下一次工程向导就无法继续加载平面基准。
- 设置页里的“平面校正文件”会跟随当前模式切换：
  - 选中 `传统模式` 时，编辑的是传统模式自己的平面校准文件路径。
  - 选中 `工程模式` 时，编辑的是工程模式自己的平面校准文件路径。
- 当前工程模式已经先落了 MVP：
  - 运行时会按模式加载对应的平面校准文件。
  - 如果配置了工程球体补偿文件，球心会额外叠加基于 `camera_px -> delta_table_mm` 的补偿。
  - 投影调试模式下，工程模式的球圈半径不再跟随检测框 `radius_px` 直接变化，而是按真实球径 `57.15mm` 投影，优先解决远端球圈明显偏大的问题。
- 建议现场使用顺序：
  1. 先在 `传统模式` 下确认台边、袋口和基础平面映射是可用的。
  2. 再切到 `工程模式`，加载工程平面文件和工程球体补偿文件。
  3. 打开“投影调试模式”，优先观察球圈中心是否压中球心、远端球圈是否仍明显偏大。

## 2026-06-26 工程球体补偿自动采样向导
- “校正投影仪”对话框里新增了 `工程球体补偿向导` 入口，只在工程模式下建议使用。
- 向导会自动生成一组覆盖台面内侧的采样点，并把目标圈投到投影窗口；现场只需要保留一颗球，并把它逐点移动到目标圈中央。
- 新版本会优先根据 `inline + pocket` 几何刷新 `center playable` 球心可达区，所有工程采样点都会尽量落在这条内圈的安全区域内，不再沿整张台面的外接矩形贴边布点。
- 每个采样点都会单独读取相机画面、调用当前球检测后端、等待球心位置稳定后自动记录，不需要每个点手动点确认。
- 现在每个采样点在检测到目标球后，会先进入 `3 秒稳定倒计时`，只有倒计时内持续稳定才会正式入样；如果球又移动、离开目标圈或稳定性丢失，倒计时会自动重置。
- 自动采样点默认提升为 `6 x 5 = 30` 个，并优先覆盖 `四个角` 与 `边缘附近`，专门加强靠边区域的校准精度。
- 当当前标定已经提供 `center playable` 边界时，采样点会尽量贴近这条球心可达边界附近，而不会再额外大幅向桌面中心收缩，方便把边缘残余误差也一起校准进去。
- 采样完成后，程序会自动生成新的工程球体补偿 JSON，写入 `local_settings/calibrations/`，并把最新文件路径回写到当前设置。
- 建议使用方法：
  1. 先完成工程模式的平面校准，并确认检测模型可用。
  2. 清空台面，只保留一颗标准球。
  3. 启动向导后，按照投影圈依次移动球，等待程序自动跳到下一个点。
  4. 完成后再打开“投影调试模式”复核全台面球圈中心与远端球圈半径。
## 2026-06-27 单实例启动

- BAS 现在带有运行期单实例保护：`Start_BAS.cmd`、`python -m bas`、`python -m bas ui`、`python -m bas run` 在已有系统运行时都会直接静默返回，不再重复拉起第二套系统。
- 重复触发启动时不弹窗、不显示“已运行”提示，也不会再创建额外的 BAS 主窗口，方便后续做 Stream Deck 或其他自动化控制。
- 远程控制命令 `python -m bas remote-control <action>` 不受单实例限制，仍然可以在系统运行期间继续向正在运行的主界面投递控制指令。
> 2026-06-27 更新：YOLO 台球类别现在只在 `inline + pocket` 拼出的最大投影可见边界内生效，`cue_stick` 只在 `outline` 内生效。正常使用不需要额外开关；更新几何文件后重新启动程序或等待几何热重载即可应用。
# BAS 本地说明（UTF-8）

## 简介

这是一个台球辅助系统，包含实时采集、检测、跟踪、状态机、路线规划、前端预览和投影显示。

## 快速使用

1. 先确认相机、标定文件、投影屏幕索引和模型路径已经在界面“设置”里配置好。
   - 如果使用 `video` backend，请同时填写 `camera.video_path`；程序会把这一路输入当作已去畸变视频，不再重复做 OpenCV 畸变校正。
2. 启动程序：

```powershell
python -m bas
```

3. 在主界面先点“开始采集”，确认预览画面正常。
4. 再点“开始投影”，确认投影窗口已经出现在配置的屏幕上。

## 2026-06-27 运行稳定性修复

- 采集启动流程已改为在主线程创建 `RuntimePipeline`，避免“后台线程创建相机/SDK/推理对象，主线程再去读取和释放”的跨线程使用问题。
- 主循环 `_tick()` 已增加异常兜底。现在如果运行期再出现异常，程序会先记录日志并停掉 pipeline，而不是直接卡住后闪退。
- 如果仍有异常，请优先查看 `logs/bas.log` 里的最后几行记录。
