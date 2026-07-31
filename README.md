# BAS台球辅助系统 使用说明

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
- `bas/training`：0–15 编号球跟踪、训练项目、摆球校验、进球顺序验证与训练投影
- `bas/state`：回合状态、进球判定与裁决逻辑
- `bas/planning`：规则路线、勾球全局选优、target shot、cue sector 等模块；旧 `free_shot.py` 仅作历史封存
- `bas/projection`：投影叠加与渲染
- `bas/ui`：桌面控制台
- `bas/web_control`：局域网 Web 控制服务、HTTP API、MJPEG 与 2604 移动端页面
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

### 摄像头采集格式与低帧率排查

实时摄像头采集只允许使用 `MJPG`。OpenCV/UVC 模式会在打开设备时一次性协商
FOURCC、分辨率和帧率，并在打开后读回 FOURCC；如果驱动返回 `YUY2`、`YUV2`
或其它格式，该后端会被拒绝，不会静默降级继续采集。Decxin/Nori SDK 模式同样
只枚举和打开 MJPG 视频格式。

刷新摄像头或运行以下命令时，OpenCV 设备会以当前配置的分辨率和帧率进行 MJPG
探测，结果中的 `opencv_mjpg` 或 `nori_mjpg` 表示已通过格式校验：

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
```

如果 1920×1080 实时画面只有约 2 FPS，优先检查运行日志中的采集信息。
`metadata` 应包含 `media_type: MJPG` 和 MJPG FOURCC 数值 `1196444237`。
若设备不支持所选分辨率/帧率的 MJPG，程序会明确报错；此时请选择摄像头实际
支持的 MJPG 分辨率或帧率，不要改回 YUY2/YUV2。

### 工业相机畸变校正总开关

设置中的“工业相机畸变矫正”是运行时相机畸变校正的唯一总开关：

- 开启且校正文件有效：在采集边界统一校正一次，实时预览、检测、Web 画面、
  抓拍静帧、无画线视频、有画线路线视频、即时回放和状态回放都使用同一份
  已校正帧，不允许下游再次加载后备校正文件。
- 关闭：运行时不加载相机畸变文件，也不校正图像或相机像素坐标；上述所有
  消费者统一使用未校正帧。
- 视频文件输入同样遵循开关。播放已经校正过的视频时应关闭开关，避免二次校正；
  播放原始视频且需要校正时应开启开关。
- 开关变化会重启采集管线，保证同一轮运行中不会混用两套帧坐标。

投影平面标定、联动校正、工程球体补偿向导以及 `inspect-calib` /
`verify-calib` 也必须遵循这个总开关，不再存在“标定工具强制启用相机内参”
的例外：

- 开关关闭时，所有标定工具都不加载相机内参，不校正整帧，也不校正 ChArUco
  角点、球心或 Holdout 相机像素坐标；平面标定直接使用当前原始画面坐标。
- 开关开启且文件有效时，采集层只校正一次，标定工具读取已经校正的帧，不再
  对角点或球心重复校正。
- `inspect-calib` 会输出 `distortion_correction_enabled` 和
  `camera_coordinate_domain`，便于确认当前使用的是 `raw` 还是
  `undistorted` 坐标域。
- 投影平面映射、工程球心补偿和纯投影训练仍可独立工作，但不得改变相机画面
  的坐标域。

### 工程球心补偿圆圈覆盖

工程球心补偿向导固定生成 `6 × 5 = 30` 个目标圆圈。圆圈位置根据当前
`inline + pocket` 拼接区域实时生成，而不是使用固定像素坐标：

- 先从当前画面重建 physical rail 和 ball-center reachable 区域。
- 六个袋口各保留一个最近的安全边界锚点，避免袋口附近缺少样本。
- 其余边缘点沿完整联合轮廓按弧长和最大间距均匀分布，不再截取轮廓数组开头
  的若干顶点。
- 剩余圆圈使用最远点优先方式填充内部，避免按坐标顺序集中在球桌上半区。
- 所有圆圈中心仍位于球心可达安全区内，圆半径按当前球直径和投影标定动态
  计算。

### 更换 UVC 工业相机与曝光控制

当 Decxin/Nori SDK 无法识别新厂商相机，但 `probe-cameras` 能列出
`opencv` 设备时，可在 `configs/default.yaml` 中使用标准 UVC 回退：

```yaml
camera:
  backend: opencv
  exposure_control: uvc
  device_index: 0
  exposure_auto: false
  exposure_level: -8
```

Windows 下程序优先使用 DirectShow，并在开始采集后回读曝光值进行校验。
常见 UVC 相机的手动曝光档位为 `-10` 到 `0`；数值越小，曝光时间越短，
画面越暗且运动拖影越少。相机应只由一个程序占用，修改配置后需要重新启动
采集。若日志出现 `UVC exposure write was not confirmed`，表示驱动接受了调用
但未实际改变曝光，此时应关闭其他相机程序后重试。

设置窗口的“工业相机曝光”区域可手动选择曝光控制方式：

- `自动选择`：优先尝试 Decxin/Nori SDK，不可用时回退 UVC/DirectShow。
- `Decxin SDK`：只使用原 Decxin 控制链；SDK 相机不可用时明确报错，不静默切换。
- `UVC / DirectShow`：跳过 Decxin SDK，直接使用标准 UVC 控制链。

选择会保存到 `local_settings/user_settings.json`，下次启动自动恢复。运行中切换
控制方式会触发采集管线重启，使新模式立即生效。

## 主控台界面

当前主控台围绕“三栏 + 中间 16:9 预览”组织：

- 左侧为运行控制、采集设置、显示与模式、现场抓取、交互调试。
- 中间为实时预览，以及最佳路线与候选路线列表。
- 右侧为模块状态、状态机人工介入、复核、事件与日志。

常用操作入口如下：

- `开始采集`：启动或停止实时采集链路。
- `开始投影`：打开或关闭投影窗口。
- `开启 Web 控制`：按“设置 → Web 控制”中的监听地址和端口启动局域网控制页。
- `抓拍原图`：保存当前运行时相机帧，不含检测框、路线等画线；它不做二次校正，
  因而与实时预览底图一致，并统一遵循工业相机畸变校正开关。
- `导出纯净回放`：导出前 60 秒纯净视频回放。
- `开始原始录制` / `开始路线录制`：分别录制无画线视频和进洞路线画线视频；
  二者共享同一运行时底图，路线模式只在该底图上增加画线，不再自行加载任何
  相机校正文件。

### 视频采集时间线

当左侧“输入类型”选择 `video` 并开始采集后，实时预览下方会显示“暂停/继续”按钮、视频时间线和“当前时间 / 总时长”。拖动滑块到目标位置并松开，即可快速跳转到对应视频帧继续检测；也可以使用方向键微调。跳转后系统会自动清空检测缓存、跨帧跟踪、进球状态和路线规划的旧时序数据，避免跳帧前的状态干扰新位置的检测。

点击“暂停”后，视频不再读取下一帧；暂停期间仍可拖动时间线，系统会检测并显示新位置的单帧画面，但不会自动继续播放。点击“继续”后从当前位置恢复检测和播放。

时间线只在 `video` 视频文件采集模式中显示；`auto`、`nori`、`opencv` 和 `synthetic` 模式不会显示。视频文件路径可在“设置”中的采集配置里指定。

界面改动后，左侧和右侧都支持滚动，交互调试与复核区域支持折叠。若窗口缩小，预览区域会继续保持 16:9 比例，避免黑边占满整块布局。

当前实时预览会按可用空间计算最大的 16:9 画面，宽或高至少有一边顶满预览视口；小尺寸输入帧也会被放大到预览区域内显示，保持不裁切。预览标题与画面之间只保留紧凑间距，避免标题上下出现大块空白。

实时采集和本地 `video` 文件共用预览缩放控件。点击“放大”或“缩小”可按 25% 的相对步长调整播放画面，当前比例显示在按钮右侧；点击“适应”会立即恢复为当前预览区域内保持 16:9、不裁切且能实现的最大画面。缩放只影响桌面预览显示，不会改变相机采集、检测坐标、录像内容、Web 画面或投影输出。

## 局域网 Web 控制

Web 控制沿用 `2604BilliardsAssistanceSystem` 的移动端页面与接口设计，默认监听 `0.0.0.0:17070`。使用步骤：

1. 打开桌面控制台的 `设置 → Web 控制`，确认监听地址为 `0.0.0.0`、接口端口为 `17070`；如端口冲突可在此修改并保存。
2. 回到主界面，点击 `开启 Web 控制`。按钮变为 `关闭 Web 控制` 即表示服务已启动。
3. 同一局域网内的手机或电脑访问 `http://<运行 BAS 的电脑 IP>:17070`。
4. 开始采集后，网页会显示实时 MJPEG 画面和运行状态；点击画面中的目标球可临时指定规划目标。

移动端页面按手机竖屏组织，组件宽度随屏幕变化：顶部显示连接状态，全宽视频保持 16:9，右下角按钮可进入横屏纯视频全屏。顶部可在“规则比赛”和“编号球训练”之间切换。规则比赛继续提供纯色球/花色球切换、颗星公式、规则模式/勾球模式、下一杆勾球/黑球和精彩瞬间录制控制；编号球训练则显示训练项目、摆球说明、实时判定、进度、用时、开始与重置按钮。

规则比赛检测到进球后，桌面操作端的视频预览和 Web 控制页都会显示约 `4.5 s` 的醒目提示。球类缩写为：白球 `wb进球`、黑球 `bb进球`、纯色球 `sob进球`、花色球 `stb进球`；同一批多球进袋会合并显示。提示模块订阅 modern 的高置信 `POCKET_DETECTED`/最终 `POCKET_CONFIRMED` 与 legacy 的 `POT_PROBABLE`，并按决定编号去重，所以裁判层等待账本复核时也不会漏掉已经通过重现窗口的进球提示。它只属于服务端操作界面和 Web 控制状态，不写入视频帧；投影交互明确忽略 `POCKET_DETECTED` 和 `POT_PROBABLE`，不会把这个文字提示或未最终确认的进球动画投影出去。

兼容接口包括 `/health`、`/state`、`/frame.jpg`、`/mjpeg` 及对应的 `/api/*` 别名；控制请求由 HTTP 线程排队后交给 Qt 主线程执行。若其它设备无法打开，请检查 Windows 防火墙是否允许 Python 监听当前端口，并确认两台设备处于同一局域网。该服务默认不含身份认证，请只在可信局域网内开启。

## PWA 模式（可选）

普通浏览器访问时保持原有行为，不会因为手机横屏自动切换布局。将页面安装为 PWA 并从桌面图标启动后，页面运行在 `standalone` 窗口中：横屏时自动使用现有画面全屏布局，竖屏时自动恢复普通页面布局。

PWA 安装需要 HTTPS 安全上下文（或在本机 `localhost` 调试）；当前默认的 `http://<局域网 IP>:17070` 仍可正常使用普通 Web 模式，但手机浏览器可能不会提供 PWA 安装入口。局域网 HTTPS 反向代理配置位于 `deploy/nginx`，默认使用 `10.1.5.175:443` 转发到 BAS 的 `17070` 端口。自签名证书可用于验证转发，但浏览器信任和 PWA 安装资格仍受证书信任链限制。PWA service worker 只缓存页面壳资源，不会缓存实时 MJPEG 或 `/api/*` 数据。

## Modern 进球识别

Modern 进球识别由四个独立模块协作：

- `bas/perception/regions.py`：球检测区域为“台面内框 + 六个袋口保护带”，保护带由各袋标定后的球直径生成，不使用固定像素值。
- `bas/perception/pocket_observer.py`：只处理六个小 ROI，通过球尺度帧差、颜色前景和最近轨迹识别 `inward_crossing`、`outward_crossing`、`lip_occupied`、`clear`；高速穿袋时先按视觉穿越质心关联袋口三球径内的当前可见球，再回退到 `450 ms` 内刚消失且运动方向对准袋口的轨迹，避免袋边静止球或历史轨迹抢走类别；不持续运行第二次 YOLO。
- `bas/state/pocket_trajectory.py`：判断可见轨迹、预击球历史和跨 ID 轨迹是否会穿过袋口。
- `bas/state/pocket.py`：把轨迹、袋口视觉和跨 ID 证据合并为一个逻辑球决定，负责确认、撤销、去重和诊断原因。

袋口局部坐标以真实 pocket curve 的质心为原点。`tangent_unit` 沿袋口曲线两端点方向，`outward_normal` 明确定义为“从桌内指向袋外”，`inward_normal = -outward_normal`。原始 inline 拼接出的 table edge polygon 只用于确定法向方向；physical rail inset 后的 `inner_polygon_mm` 不参与这一步；ball-center reachable polygon 只用于判断球心是否已回到正常台面。

对球心 `ball_center`，统一计算：

```text
delta = ball_center - pocket_center
depth = dot(delta, outward_normal)
lateral = abs(dot(delta, tangent_unit))
```

判定由深到浅执行，且每层都必须同时满足深度和横向宽度：

```text
INTERIOR: depth >= interior_depth
          and lateral <= interior_width / 2

THROAT:   depth >= throat_depth
          and lateral <= throat_width / 2 + ball_radius

MOUTH:    depth >= -ball_radius * 0.6
          and lateral <= mouth_width / 2 + ball_radius

否则:     NONE
```

Modern 不再使用径向 distance 产生 zone；`pocket_funnel_radius_mm` 和 `pocket_mouth_settle_ms` 只为 legacy engine 保留。真实曲线不存在时，才使用袋口中心相对台面质心的方向作为兼容法向，但仍执行相同的二维 zone 和几何自检。

高速球常在球心进入严格 MOUTH 前就被袋口遮挡，因此另设“不改变 zone 的袋前轨迹走廊”：候选深度约 `125 mm`，跨 ID 接力窗口约 `450 ms`，预击球历史保留 `1.5 s`。只有球组、袋号、位移速度和捕获宽度均相容时才继承轨迹。对于跟踪器在袋口才新建、只有一两帧且速度尚未建立的轨迹，只有帧差至少 `0.75`、前景至少 `0.60`、穿越深度至少 `0.20` 个球径的强视觉证据才能建立候选，随后仍须消失并完成 `1.3 s` 反证窗口。对于一两帧拉长、低置信且随后完全消失的高速球，状态机另使用受限的 `blurred_single_frame_disappearance` 证据；正常圆形静止球不满足这些分支。
高速模糊兜底还要求检测框宽高比不超过 `pocket_blur_max_aspect_ratio`（默认 `2.8`），避免把球杆、手臂或栏边形成的超长伪框当成球。投影进袋候选使用独立的连续回弹计时：明确向外速度立即拒绝；位置回退但速度估计仍短暂向内时，连续 `90 ms` 后拒绝，证据时间刷新不会重新开始这段计时。

自动进球时序为：

1. 球进入严格袋区、满足袋前投影/跨 ID 连续性，或袋口观察器给出已关联球组的向内穿越，产生一个逻辑 `POCKET_CANDIDATE`。
2. 候选进入袋口后等待 `1.3 s`。期间若出现向外穿越、桌面侧重现、轨迹反向或持续袋唇占用，则自动产生 `POCKET_REJECTED`。
3. 只有逻辑球已经持续离开桌面且没有反证的强候选才产生一次 `POCKET_DETECTED`；球仍处于 `visible` 时，即使袋口视觉短暂误判为 `clear` 也不得播报。单纯跟踪器 `occluded` 外推不能确认。视觉证据不足会自动拒绝并记录原因，不进入人工进球审核。
4. 同袋多候选分别维护，真实进球与无效候选互不阻塞；过期视觉候选自动拒绝，避免跨杆残留和重复播报。
5. 账本观测数量不一致只记录裁判异常，不阻止已经确认的 `POCKET_DETECTED` 和服务端提示。

每份强证据绑定 `decision_id + pocket_index`。同 track 或兼容的新 ID 在桌面侧重现时会撤销旧决定；本回合早已存在的另一颗同组球不会被误当作重现，因此两颗同组球可以先后进入同一袋。投影轨迹重新向台内回退、向外穿越或横向离开捕获走廊时会立即撤销。袋唇球如果在候选后长时间仍可见，即使跟踪器随后外推到袋内，也会保持否决直到自动超时拒绝。

高速球、单帧检测、跨 ID、击球相位滞后、撞袋回弹、袋唇静止、同袋多候选以及真进球与假候选并存均有独立回归测试。

几何只在上下文变化时重建，并自动检查六袋坐标轴、桌面中心 `zone == NONE`，以及固定随机种子的 500 个 ball-center reachable 内点；INTERIOR 比例必须小于 `2%`。单袋无效时排除该袋，全局检查失败时 fail-closed，Modern 自动进球停用且不会回退到径向判定。可通过 `ModernMatchStateMachine.debug_snapshot()["pocket_geometry"]` 查看六袋中心、切线、法向、宽深、探针距离、随机抽样比例和失败原因。

默认配置使用：

```yaml
state:
  engine: modern
  pocket_tentative_missing_ms: 300
  pocket_commit_ready_missing_ms: 700
  pocket_reappear_window_ms: 800
  pocket_visual_confirmation_ms: 1300
  pocket_lip_veto_ms: 1100
  pocket_entry_candidate_depth_mm: 125.0
  pocket_entry_history_depth_mm: 450.0
  pocket_entry_history_ms: 1500
  pocket_entry_handoff_ms: 450
  pocket_blur_max_aspect_ratio: 2.8
  pocket_entry_min_speed_mm_s: 100.0
  pocket_entry_max_speed_mm_s: 4000.0
  turn_resolve_grace_ms: 900
```

旧 `pocket_confirm_missing_ms` 仍可作为 commit-ready 门槛别名；只在新字段未配置时生效。修改代码或配置后必须正常重启 BAS，再验证新逻辑。

长视频标签位于 `tests/fixtures/long_video_goal_labels.json`。本地放好被 `.gitignore` 排除的视频和回放后执行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pocket_video.py --replay replays\session_20260720_135725_516792 --video local_settings\captures\no_line_video_20260627_185946_1_1.mp4
```

逐帧复核确认第 4 杆右上袋 `stb` 和第 27 杆中上袋 `stb` 都属于真实进球，因此完整验收基准为最初统计的 13 球。当前结果为 `PASS: matched=13/13 detected=13`、0 误报、0 重复，观察器 p95 为 `3.87 ms`，估算整体帧率下降 `4.84%`。第 2 杆为 `sob`、第 4 杆为 `stb`、第 27 杆为 `stb`、第 34 杆为 `bb`，第 3/18/24/30 杆无播报，所有决定延迟不超过 `1.5 s`。回放验收按 `contact_frame` 对齐真实击球，不依赖可能被空杆误触发影响的内部杆号，并会把同一杆多报的错误袋号单独计为误报。

只复现本次第 3/4 杆问题可执行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pocket_video.py --replay replays\session_20260720_172814_256157 --video local_settings\captures\no_line_video_20260627_185946_1_1.mp4 --labels tests\fixtures\shot3_shot4_regression_labels.json --stop-frame 878
```

当前该片段结果为 2/2 命中、0 误报、0 重复；第 4 杆在 774 帧播报 `stb`、袋号 `2`、决定延迟 `1328 ms`，第 3 杆不播报。观察器 p95 约 `4.04 ms`，估算整体帧率下降约 `4.53%`。

桌面预览和 Web 前端会明显显示 `wb进球`、`bb进球`、`sob进球` 或 `stb进球`。提示只消费服务端状态事件；投影交互层明确忽略 `POCKET_DETECTED`/`POT_PROBABLE`，不会显示进球文字。

进球模块定向回归：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp .tmp\pytest-pocket tests\test_pocket_geometry.py tests\test_pocket_observer.py tests\test_pocket_visual_state.py tests\test_pocket_evaluator.py tests\test_state_modern.py tests\test_state_replay.py tests\test_web_pocket_notice.py tests\test_projection_interaction.py tests\test_web_control.py -q
```

## 测试与回归

执行规划相关回归：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py tests\test_route_geometry.py tests\test_cue_sector_preview.py tests\test_user_settings.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

如果只想快速检查核心规划链路，也可以运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_planner.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

检查主控制台前端布局、预览缩放和按钮关联：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ui_runtime.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

检查 Web 控制接口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_control.py -q --basetemp=pytest_tmp_codex -o cache_dir=pytest_cache_local\pytest_cache
```

## 最近一次规划修复说明

这次修复聚焦于“矩形框轻微歪斜后进洞线消失”问题，核心做法是统一 `target_lock`、`target_shot` 与 rule 路线在目标选择和直线进袋判定上的行为：

- 抽出共用的走廊目标选择逻辑，严格以“球心在走廊内”为命中标准
- 让 `target_lock` 和 `target_shot` 复用同一套目标排序规则，避免擦边球抢锁定
- 将零库直线进袋判定切到 center playable polygon 语义，并复用现有袋口放宽规则
- 保持 bank/rebound、评分、时序参数和 UI 配置项不变，避免修复范围外扩

## 勾球模式

勾球模式替代原自由击球模式。系统遍历状态机当前花色对应的所有可见目标球，并比较每颗球的直球、一库和两库路线，显示几何评分最高的一条；母球只碰目标球一次，其他球作为障碍物，不计算借球传球。当前花色尚未确定时会遍历纯色球和花色球，但不会自行修改状态机花色，也不会自动选择黑球。

使用方式：

1. 在桌面端或 Web 控制器选择“勾球模式”，即可持续显示当前花色的全局最优路线。
2. 将球杆稳定对准某颗目标球并达到保持时间，可临时锁定该球；保持计时未完成前仍显示全局最优路线。规则模式原有的球杆范围选择逻辑保持不变，但勾球模式的自动选优不受该范围限制。
3. 在 Web 实时画面中点击任意非白球，可临时只规划该球；点击白球会被拒绝。状态机确认 `SHOT_STARTED` 后自动解除手动锁定。
4. Stream Deck 可使用 `scripts/BAS_SD_Toggle_Shot_Mode.cmd` 切换规则/勾球模式，使用 `scripts/BAS_SD_Hook_Shot_Once.cmd` 启用“下一杆勾球”。旧 `BAS_SD_Free_Shot_Once.cmd` 和 `free-shot-once` 命令仅作兼容别名，实际只进入新勾球逻辑。

旧 `bas/planning/free_shot.py`、相关历史数据结构和参数仍保留用于追溯，但主规划器不再导入、实例化或调用旧自由击球规划器。旧设置中的 `free` / `free_shot` 会在读取时迁移为 `hook`。

## 编号球训练模式（YOLO11s）

规则比赛与训练使用两套互不混用的模型：

- 规则比赛继续使用 `detector` 中配置的 YOLO11m 四分类模型：`wb / sob / stb / bb`。
- 编号球训练只使用 `training_detector` 中配置的 `example/yolo11s_0719_16seg_best_v1.pt`。模型类别为 `0–15`，其中 `0=白球`、`1–7=实色球`、`8=黑八`、`9–15=花色球`。
- 检测模型按运行模式惰性加载。启动规则比赛时不会预加载 YOLO11s；第一次切到训练模式时才加载编号球模型。切回规则比赛后恢复 YOLO11m 检测、规则状态机和路线规划。

训练项目分为新手、入门、白球控制、进阶和其它训练五组。桌面端和 Web 端按组显示，并保留原有项目 ID 和存档兼容性。

新手训练模式：

1. `新手训练1：单球自由入袋`（`BEGINNER_SINGLE_FREE`）：只摆白球和 1 号球，1 号球进入任意袋即完成。
2. `新手训练2：单球直线入袋`（`BEGINNER_SINGLE_STRAIGHT`）：以宽松共线误差练习中短距离直球。
3. `新手训练3：1–3 顺序短线`（`BEGINNER_1_TO_3_LINE`）：复用一字线校验，按 `1→2→3` 完成。
4. `新手训练4：1–3 三点顺序`（`BEGINNER_1_TO_3_POINTS`）：按归一化球桌坐标定义三个宽松区域，按 `1→2→3` 完成。
5. `新手训练5：三球自由清台`（`BEGINNER_3_BALL_FREE`）：1–3 进球顺序自由。
6. `新手训练6：五球自由清台`（`BEGINNER_5_BALL_FREE`）：1–5 进球顺序自由。

入门训练模式：

1. `入门训练1：1–3 顺序短线`（`ENTRY_1_1_TO_3_LINE`）：1、2、3 排成较短一字线，按 `1→2→3` 进球，练习认识目标球和简单连续击球。
2. `入门训练2：1–3 三点顺序`（`ENTRY_2_1_TO_3_POINTS`）：1、2、3 放在三个容易处理的推荐区域，按 `1→2→3` 进球，练习方向切换。
3. `入门训练3：三球自由清台`（`ENTRY_3_3_BALL_FREE`）：1、2、3 分散摆放且顺序不限，练习观察台面和自主选择目标球。
4. `入门训练4：左右换边三球`（`ENTRY_4_LEFT_RIGHT_3_BALL`）：1 号在左侧、2 号在右侧、3 号在中部推荐区域，按 `1→2→3` 进球，练习大方向走位。
5. `入门训练5：五球自由清台`（`ENTRY_5_5_BALL_FREE`）：1–5 分散摆放、顺序不限，完成第一次小型清台。

其中 1、2、3、5 直接复用对应新手项目的球号、阶段、布局、安全约束和 `guided` 规则，仅使用独立场景 ID 出现在入门目录中；4 只增加三个归一化位置区域，继续走相同的摆球校验、进球判定、会话和投影提示链路。

白球控制训练模式：

1. `白球控制训练1：两球基础衔接`（`CUE_CONTROL_TWO_BALL_LINK`）：必须先打 1 号，再打 2 号；1 号进球后，白球需要停入较大的合格区域，合格后才进入 2 号球阶段。
2. `白球控制训练2：白球停球区`（`CUE_CONTROL_STOP_ZONE`）：1 号球进袋后，白球应停在开球位置附近的大圆形区域，训练中杆停球。
3. `白球控制训练3：白球前进区`（`CUE_CONTROL_FOLLOW_ZONE`）：1 号球进袋后，白球应进入目标球前方的宽泛圆形区域，训练跟杆基础。
4. `白球控制训练4：白球回位区`（`CUE_CONTROL_DRAW_ZONE`）：1 号球进袋后，白球应回到目标球后方的宽泛圆形区域，训练低杆基础。

这四个项目复用同一个白球停止观察模块。目标球确认进袋后，训练会话暂停阶段推进；只有可见白球的毫米速度低于静止阈值、位置在抖动容差内连续稳定至少 `650 ms`，才记录本杆白球停止位置并判断目标区。白球丢失、仍在移动或尚未稳定时不会提前给出结果。两球衔接的第二杆虽然没有额外目标区，也必须等白球稳定后才完成训练。

白球目标区使用球桌毫米坐标生成 64 段圆形轮廓，并通过当前投影校准映射到投影画布：黄色虚线表示待完成，绿色表示达标，红色表示未达标。停球区以本杆初始白球位置为中心；前进区和回位区根据初始白球到目标球的方向自动计算，因此不依赖某一套固定相机或投影分辨率。

进阶训练模式：

1. `进阶训练1：1–7 顺序一字线`：校验 1–7 是否按号码排成直线，并要求按 `1→7` 进球。
2. `进阶训练2：15 球中线蛇彩`：校验 1–15 中线排列，并要求按 `1→15` 清台。
3. `进阶训练3：1–9 轮转清台`：球形可自由分散，始终要求先打当前最低号球。
4. `进阶训练4：单双号双排阶梯`：奇数、偶数分别成排，按 `1/3/5/7/9/2/4/6/8` 训练跨区走位。
5. `进阶训练5：实色清台 + 黑八`：1–7 组内顺序自由，8 号球必须最后进。
6. `进阶训练6：花色清台 + 黑八`：9–15 组内顺序自由，8 号球必须最后进。
7. `进阶训练7：6–7–8 三球收官`：短局反复训练关键球、过渡球和黑八收尾。

其它训练：

1. `其它训练1：出杆检测`（`OTHER_STROKE_CHECK`）：将 `example/Stroke-Check-Tool.pdf` 的有效图形按真实物理尺寸重画到球桌投影，忽略原图左下角 Logo。三条横线的中心间距均为 `3 in`（`76.2 mm`），圆圈中心线直径为标准台球的 `2.25 in`（`57.15 mm`），粗线表示球杆方向。

出杆检测是纯投影项目，不加载或读取工业相机画面，也不执行编号识别、袋口判定和路线规划，并且不叠加颗星公式。选择该项目并点击“开始投影”即可使用：初始圆圈中心位于球桌长轴 `1/4`、短轴中线处，击球方向朝上侧长边，粗线与该长边垂直。把白球放入圆圈，沿粗线方向反复出杆，通过三条横线观察球杆在行程中的横向偏移。所有图形先以球桌毫米坐标生成，再经过当前投影标定映射，因此不会因投影分辨率或梯形校正改变物理比例。

训练开始前会检查袋口标定、所需球是否齐全、是否存在项目外的球、白球是否可见，以及一字线/双排项目的初始队形。袋口标定不可用时训练不允许开始。点击“开始验证”后，编号跟踪器只负责提供球号身份；进球事实统一交给规则模式的 `PerBallPocketFSM` 判定。训练分支与 Modern 规则模式共享完整台面几何、袋前轨迹、跨 ID 接力、袋口视觉观察和重现反证窗口，只有收到高置信 `POCKET_DETECTED` 才把对应号码记为进球，不再用普通消失或“最后可见位置距离袋口约 190 mm”推断进球。袋口观察器在训练模式只接收当前项目要求的号码和白球，偶发识别出的项目外球号不会抢占袋口关联。

带推荐摆球区域的项目会在投影画面上直接显示彩色虚线区域，并在区域中心标注“摆放 N 号球”。引导覆盖新手训练 4、三球自由清台、对应入门训练及白球控制衔接等所有声明了 `zones` 的项目；区域轮廓与规则判定共用同一套归一化椭圆几何，因此投影所见边界就是开始验证时采用的边界。引导仅在摆球和就绪阶段显示，训练正式开始后自动隐藏。

训练主提示和区域内的摆球标签统一使用 Unicode 字体渲染。文字不会固定沿投影画布水平线显示，而是根据当前工程标定，在各自锚点处将桌面 X 轴投影到最终画布，并使用得到的局部角度旋转；因此更换投影标定或调整桌面边界后无需手工填写文字角度，也不会依赖颗星公式中的手动角度。

训练运行期间不要求所有号码球持续在线：人体、球杆或其他球造成的遮挡，编号模型漏检，球被手动移动，乃至已记进球的号码再次被识别，都不会让本次训练失败，也不会撤销已经记录的进球。没有袋口证据的消失只视为“当前不可见”，不会推断进球或违规；共享判定器仍可等待稍后出现的袋口视觉穿越、合法轨迹确认或重现反证。每次开始、重新开始、重置或切换训练项目都会清理上一轮的共享进球候选和袋口观察历史，视觉证据不会跨轮次继承。

进阶训练继续使用原有严格策略：进错号码、黑八过早进入、白球落袋，以及多球同时确认进袋而无法判断顺序时，本次训练失败。新手训练使用宽松策略：未进球保持当前进度；误进球记录错误并提示重摆，目标不会错误前移；白球落袋提示重摆后继续，已完成进度不清空；自由清台项目显示剩余球、已进数量和“自由选择模式”。带安全距离约束的场景会统一检查球心可达区域、贴库、贴球和袋口距离；缺球、多球、错区或明显错序会给出具体调整提示。

旧配置中的 `training.disappearance_confirm_frames` 会被兼容忽略；`training.pocket_proximity_mm` 也仅为兼容已有 YAML 保留，训练进球判定不读取它。

### 训练核心重构与规则扩展

训练流程现在只有一套共享引擎 `TrainingSession`。检测、编号跟踪、摆球校验、袋口判定、进度、计时、事件和投影输出均由所有训练项目复用；项目本身只声明球号、阶段、布局约束和 `rule_set_id`。

- `bas/training/rules.py` 是训练规则 seam。当前提供 `guided`（提示重摆后继续）和 `strict`（立即失败）两个规则集，白球落袋、误进球和多球同时进袋都在这里作纯规则判断。
- `bas/training/session.py` 只执行规则结果，不再判断“新手还是进阶”。未来新增训练类型时，优先复用现有规则集；确实有新行为时再增加一个规则集。
- `bas/training/layout_validation.py` 按场景声明的布局和安全约束执行校验，不再按训练分组分叉。
- `bas/training/prompts.py` 集中维护运行消息和投影中文提示，避免会话、桌面界面和投影各自拼接不同文案。

### 训练中文交互提示

在桌面端“设置 → 投影输出 → 训练中文提示”中可以：

- 开启或关闭训练中文提示；
- 用最终投影画布的水平、垂直百分比调整位置；默认 `X=50%`、`Y=74%`，即中间偏下；
- 用像素调整字号，默认 `36 px`。

这些设置只保存在桌面后台的 `local_settings/user_settings.json`，Web 控制页不提供读取或修改入口。文字使用支持中文的字体直接绘制到最终投影像素画布，发生在路线和球位完成几何校准之后，因此校准矩阵只影响几何图形，不会拉伸字形。

投影提示覆盖以下状态：

- 摆球中：`请完成摆球`，第二行显示缺球、多球、错区、贴库、贴袋或校准缺失等具体原因；
- 已就绪：`摆球正确，可以开始训练`；
- 固定目标：`当前目标：N 号球`；
- 自由目标：`当前可进：N、N…`；
- 等待判定：`正在确认训练结果`；
- 白球落袋（引导规则）：`请先完成重摆`，并说明重摆白球、保留进度；
- 误进球（引导规则）：`请先完成重摆`，并说明误进号码和当前目标；
- 多球同时入袋（引导规则）：`请先完成重摆`，并列出需重摆的球；
- 严格规则失败：`本次训练结束`，第二行显示白球落袋、误进号码或无法确认顺序，并提示重新开始；
- 完成：`训练完成！`，第二行显示进度、用时和累计提示次数。

桌面端使用步骤：

1. 在“显示与模式 → 运行模式”选择“编号球训练”。
2. 在“编号球训练”区域选择项目，并按摆球说明摆放。
3. 开始采集；状态显示“摆球已通过”后点击“开始验证”。
4. 预览和投影会高亮当前目标球，并显示中文交互提示、进度、用时和成功/失败信息；更换球形后点击“重置本次”。

Web 端使用步骤：

1. 打开局域网控制页，点击“编号球训练”。
2. 选择训练项目，按页面说明摆球。
3. 点击“开始验证”，页面每秒同步训练进度；可随时点击“重置本次”。

相关 HTTP 接口：`POST /api/runtime_mode/set`、`POST /api/training/scenario/set`、`POST /api/training/start`、`POST /api/training/reset`。训练状态和项目列表随 `GET /api/state` 返回。

训练模式回归测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_beginner_training.py tests\test_cue_ball_control_training.py tests\test_training_mode.py tests\test_web_control.py -q --basetemp=pytest_tmp_training -o cache_dir=pytest_cache_local\training
```

## 仓库约定

- 本地大文件、日志、回放、模型和缓存目录不要提交，相关路径应维护在 `.gitignore`
- 调试优先做成独立模块、脚本或测试，避免把临时代码堆进主链路
- 复现和排查问题时，优先依赖 `tests`、`scripts` 和回放素材，保证结论可重复验证

### 清理本地临时文件

测试命令中的 `--basetemp=pytest_tmp_*` 会在项目根目录创建 pytest 隔离工作区；其中的空文件和小型 JSON 文件是测试夹具，不是业务数据。调试截图、临时视频和测试运行副本集中放在 `.tmp/`，Python 字节码缓存位于 `__pycache__/`。这些内容均可重新生成。

预览将要清理的目录：

```powershell
.\scripts\Clean-Workspace.ps1 -WhatIf
```

执行清理：

```powershell
.\scripts\Clean-Workspace.ps1
```

脚本只清理项目内的 `pytest_tmp*`、`python_tmp*`、pytest/Python 缓存、`.tmp/`、`tmp/` 和 `logs/`，并明确保护 `.git/` 与 `.venv/`。校准数据、预设、回放、模型和业务配置不会被该脚本删除。

## 局域网 Web 点选目标球

规则比赛模式下，打开局域网 Web 控制页后，直接点击实时视频中的非白球即可指定进洞规划目标。该能力适用于规则模式与勾球模式；勾球模式下，手动点选目标会覆盖当前花色的全局选优，并为该目标生成最多两库的最佳进洞路线。白球始终不可被选中。编号球训练模式由训练项目自动判定当前目标，不接受手动点选覆盖。

点选后目标由规划器强制保持，不会因球杆方向变化、短暂丢失或自动目标选择而切换。下一次状态机确认 `SHOT_STARTED`（出杆）事件时，手动目标自动释放；前端暂不显示锁定标记。

Web 端会根据实时画面的 `object-fit` 显示方式补偿裁剪和黑边，因此普通页面、原生全屏、PWA 横屏以及非 16:9 相机画面都使用同一套相机帧坐标。黑边区域不会发送选球请求。兼容全屏模式下，点击画面继续用于选球；需要退出时点击右下角全屏按钮。

坐标换算回归测试使用 Node.js 内置测试运行器，无需安装额外 npm 包：

```powershell
node --test tests\web_control_coordinates.test.js
```

## 训练模式进洞路线

训练模式会根据当前训练阶段自动复用已有的进洞路线规划器，不维护第三套路线算法：

- 当前阶段只有一个明确球号时（例如 `1→7` 顺序清台），只把该球号交给现有勾球模式规划器，比较直球、一库和两库路线并推荐最优路线。
- 当前阶段允许多个球号时（例如实色球清台或花色球清台），把当前仍可进的球号交给现有规则模式规划器，并继续根据白球、球杆方向和合法目标球画线。
- `实色清台 + 黑八` 与 `花色清台 + 黑八` 在组内球清完之前会把 8 号球排除在候选目标之外；进入最后的黑八阶段后，才会为 8 号球规划路线。
- 训练目标圈、球号、进度和用时会与进洞路线叠加显示；训练完成或当前没有合法目标时不会生成新的路线。

本地验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_training_mode.py tests\test_planner.py -q --basetemp=pytest_tmp_training_routes -o cache_dir=pytest_cache_local\training_routes
```

## 更换台面几何文件后的区域规则

运行时会读取设置中的 `outline.json`、`inline.json` 和 `pocket.json`。系统使用文件内容哈希检查变化；即使新旧文件大小和修改时间相同，只要内容变化也会自动重载。校验成功后会重建边界，并清空检测缓存、跟踪历史、袋口观察历史、状态机历史和规划锁定，避免旧轨迹继续绑定旧袋位。六个袋口会统一规范为 `左上、上中、右上、右下、下中、左下`，因此 `pocket.json` 中 shape 的书写顺序不会改变袋号。

球检测与球心可达域采用彼此独立的职责：

1. YOLO 球候选只以 `inline + pocket` 拼接出的相机画面多边形作为硬准入边界。位于该边界内的真实贴库球不会因为规划安全边而被删除。
2. `center_playable_polygon_mm` 继续由当前物理库边内缩、球半径和球心额外安全边实时计算，用于路线规划、摆球合法性、训练安全距离和袋口几何判断，不再作为检测层的硬过滤条件。

袋口保护区仅供袋口视觉观察器分析进袋画面，不允许位于 `inline + pocket` 外的新 YOLO 球候选进入跟踪器。合法球进入袋口、离开检测边界后，观察器会使用新袋位 ROI、最近轨迹和画面运动继续完成进洞接力。

如果首次启动时配置的几何文件不存在、JSON 未写完整或校验失败，检测器会进入 `region_disabled` 状态，不会退回宽标定区域识别球；运行中替换文件发生临时半写入时，则继续使用上一套有效几何并自动重试。

更换文件后建议打开“投影调试模式”，确认 `inline`、`pocket`、`physical` 和 `center` 四类参考线位置正确。定向回归测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_detection_regions.py tests\test_detection_service.py tests\test_geometry_runtime.py tests\test_geometry_pocket_integration.py tests\test_pocket_observer.py tests\test_pocket_visual_state.py
```
