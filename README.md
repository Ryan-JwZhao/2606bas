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

## 主控台界面

当前主控台围绕“三栏 + 中间 16:9 预览”组织：

- 左侧为运行控制、采集设置、显示与模式、现场抓取、交互调试。
- 中间为实时预览，以及最佳路线与候选路线列表。
- 右侧为模块状态、状态机人工介入、复核、事件与日志。

常用操作入口如下：

- `开始采集`：启动或停止实时采集链路。
- `开始投影`：打开或关闭投影窗口。
- `开启 Web 控制`：按“设置 → Web 控制”中的监听地址和端口启动局域网控制页。
- `抓拍原图`：保存当前无画线校正照片。
- `导出纯净回放`：导出前 60 秒纯净视频回放。
- `开始原始录制` / `开始路线录制`：分别录制无画线视频和进洞路线画线视频。

界面改动后，左侧和右侧都支持滚动，交互调试与复核区域支持折叠。若窗口缩小，预览区域会继续保持 16:9 比例，避免黑边占满整块布局。

当前实时预览会按可用空间计算最大的 16:9 画面，宽或高至少有一边顶满预览视口；小尺寸输入帧也会被放大到预览区域内显示，保持不裁切。预览标题与画面之间只保留紧凑间距，避免标题上下出现大块空白。

## 局域网 Web 控制

Web 控制沿用 `2604BilliardsAssistanceSystem` 的移动端页面与接口设计，默认监听 `0.0.0.0:17070`。使用步骤：

1. 打开桌面控制台的 `设置 → Web 控制`，确认监听地址为 `0.0.0.0`、接口端口为 `17070`；如端口冲突可在此修改并保存。
2. 回到主界面，点击 `开启 Web 控制`。按钮变为 `关闭 Web 控制` 即表示服务已启动。
3. 同一局域网内的手机或电脑访问 `http://<运行 BAS 的电脑 IP>:17070`。
4. 开始采集后，网页会显示实时 MJPEG 画面和运行状态；点击画面中的目标球可临时指定规划目标。

移动端页面按手机竖屏组织，组件宽度随屏幕变化：顶部显示连接状态，全宽视频保持 16:9，右下角按钮可进入横屏纯视频全屏。顶部可在“规则比赛”和“编号球训练”之间切换。规则比赛继续提供纯色球/花色球切换、颗星公式、规则模式/勾球模式、下一杆勾球/黑球和精彩瞬间录制控制；编号球训练则显示训练项目、摆球说明、实时判定、进度、用时、开始与重置按钮。

兼容接口包括 `/health`、`/state`、`/frame.jpg`、`/mjpeg` 及对应的 `/api/*` 别名；控制请求由 HTTP 线程排队后交给 Qt 主线程执行。若其它设备无法打开，请检查 Windows 防火墙是否允许 Python 监听当前端口，并确认两台设备处于同一局域网。该服务默认不含身份认证，请只在可信局域网内开启。

## PWA 模式（可选）

普通浏览器访问时保持原有行为，不会因为手机横屏自动切换布局。将页面安装为 PWA 并从桌面图标启动后，页面运行在 `standalone` 窗口中：横屏时自动使用现有画面全屏布局，竖屏时自动恢复普通页面布局。

PWA 安装需要 HTTPS 安全上下文（或在本机 `localhost` 调试）；当前默认的 `http://<局域网 IP>:17070` 仍可正常使用普通 Web 模式，但手机浏览器可能不会提供 PWA 安装入口。局域网 HTTPS 反向代理配置位于 `deploy/nginx`，默认使用 `10.1.5.175:443` 转发到 BAS 的 `17070` 端口。自签名证书可用于验证转发，但浏览器信任和 PWA 安装资格仍受证书信任链限制。PWA service worker 只缓存页面壳资源，不会缓存实时 MJPEG 或 `/api/*` 数据。

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

首批训练项目：

1. `1–7 顺序一字线`：校验 1–7 是否按号码排成直线，并要求按 `1→7` 进球。
2. `15 球中线蛇彩`：校验 1–15 中线排列，并要求按 `1→15` 清台。
3. `1–9 轮转清台`：球形可自由分散，始终要求先打当前最低号球。
4. `单双号双排阶梯`：奇数、偶数分别成排，按 `1/3/5/7/9/2/4/6/8` 训练跨区走位。
5. `实色清台 + 黑八`：1–7 组内顺序自由，8 号球必须最后进。
6. `花色清台 + 黑八`：9–15 组内顺序自由，8 号球必须最后进。
7. `6–7–8 三球收官`：短局反复训练关键球、过渡球和黑八收尾。

训练验证会检查所需球是否齐全、是否存在项目外的球、白球是否可见，以及一字线/双排项目的队形。点击“开始验证”后，系统根据编号球持续可见性和最后可见位置确认进球；默认连续 `8` 帧未识别，且最后位置在袋口约 `190 mm` 范围内才认定进球，以过滤瞬时漏检和球台中部遮挡。以下情况会直接判定本次失败：进错号码、黑八过早进入、白球落袋、多球同时消失无法判断顺序、球在非袋口区域持续丢失、已判定进球的球重新出现。

桌面端使用步骤：

1. 在“显示与模式 → 运行模式”选择“编号球训练”。
2. 在“编号球训练”区域选择项目，并按摆球说明摆放。
3. 开始采集；状态显示“摆球已通过”后点击“开始验证”。
4. 预览和投影会高亮当前目标球，并显示进度、用时和成功/失败信息；更换球形后点击“重置本次”。

Web 端使用步骤：

1. 打开局域网控制页，点击“编号球训练”。
2. 选择训练项目，按页面说明摆球。
3. 点击“开始验证”，页面每秒同步训练进度；可随时点击“重置本次”。

相关 HTTP 接口：`POST /api/runtime_mode/set`、`POST /api/training/scenario/set`、`POST /api/training/start`、`POST /api/training/reset`。训练状态和项目列表随 `GET /api/state` 返回。

训练模式回归测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_training_mode.py tests\test_web_control.py -q --basetemp=pytest_tmp_training -o cache_dir=pytest_cache_local\training
```

## 仓库约定

- 本地大文件、日志、回放、模型和缓存目录不要提交，相关路径应维护在 `.gitignore`
- 调试优先做成独立模块、脚本或测试，避免把临时代码堆进主链路
- 复现和排查问题时，优先依赖 `tests`、`scripts` 和回放素材，保证结论可重复验证

## 局域网 Web 点选目标球

规则比赛模式下，打开局域网 Web 控制页后，直接点击实时视频中的非白球即可指定进洞规划目标。该能力适用于规则模式与勾球模式；勾球模式下，手动点选目标会覆盖当前花色的全局选优，并为该目标生成最多两库的最佳进洞路线。白球始终不可被选中。编号球训练模式由训练项目自动判定当前目标，不接受手动点选覆盖。

点选后目标由规划器强制保持，不会因球杆方向变化、短暂丢失或自动目标选择而切换。下一次状态机确认 `SHOT_STARTED`（出杆）事件时，手动目标自动释放；前端暂不显示锁定标记。

Web 端会根据实时画面的 `object-fit` 显示方式补偿裁剪和黑边，因此普通页面、原生全屏、PWA 横屏以及非 16:9 相机画面都使用同一套相机帧坐标。黑边区域不会发送选球请求。兼容全屏模式下，点击画面继续用于选球；需要退出时点击右下角全屏按钮。

坐标换算回归测试使用 Node.js 内置测试运行器，无需安装额外 npm 包：

```powershell
node --test tests\web_control_coordinates.test.js
```
