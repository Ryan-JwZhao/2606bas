# Billiards Assistance System

这是一个从头重构的台球智能辅助系统骨架，围绕 `Capture -> Detect -> Track -> State -> Plan -> Projection -> Replay` 这条链路拆分模块。旧项目只作为接口和参数格式参考使用，尤其是 Nori 工业相机 SDK 与 OpenCV 标定文件读取。

当前版本保留几何候选和快速物理筛选作为基础算法；学习系统采用独立的采集与训练工具，训练结果可通过设置中的“学习排序模型”路径接入主程序。模型路径为空或加载失败时，主程序会自动退回基础几何规划。

## 快速运行

双击根目录的 `Start_BAS.cmd` 可以直接打开桌面控制台 UI。默认不再启用 synthetic 虚拟画面；真实使用时在 UI 里选择 `auto`、`nori`、`opencv` 或 `video`。

当前工作区的 `.venv` 已安装运行、YOLO 和学习训练依赖。`Start_BAS.cmd` 只负责设置本地运行目录并启动 UI，不会要求手工安装依赖。若以后移动目录或破坏 `.venv`，再运行一次：

```powershell
.\Setup_Environment.cmd
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt  # 使用 Ultralytics/YOLO 时需要
.\.venv\Scripts\python.exe -m bas smoke-run --frames 90
```

常用命令：

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
.\.venv\Scripts\python.exe -m bas doctor
.\.venv\Scripts\python.exe -m bas inspect-calib
.\.venv\Scripts\python.exe -m bas verify-calib C:\path\to\holdout.json
.\.venv\Scripts\python.exe -m bas ui
.\.venv\Scripts\python.exe -m bas run --headless --max-frames 300
.\.venv\Scripts\python.exe -m bas run
```

学习排序训练也提供一键脚本：

```powershell
.\Train_RL_One_Click.cmd
```

该脚本会读取 `rl/data/samples` 下的学习样本，训练后导出 `rl/models/ranker.json`。主程序设置页中把“学习排序模型”指向这个 JSON 即可使用训练结果。

## 模块说明

- `bas.capture`: OpenCV、视频文件、合成源、Nori SDK 采集。Nori SDK 只筛选 MJPG。
- `bas.calibration`: OpenCV YAML 内参、投影单应矩阵、局部残差场、台面毫米坐标映射。
- `bas.perception`: 调试颜色检测器和 Ultralytics YOLO 模型入口。
- `bas.tracking`: 双阈值时序跟踪、短时遮挡保留、速度估计。
- `bas.state`: `STABLE_IDLE/PRE_SHOT_ARMED/SHOT_ACTIVE/SETTLING/TURN_RESOLVE/ANOMALY_RECOVERY` 宏状态。
- `bas.planning`: 规则模式几何候选、自由模式球杆路线与两次碰撞预测；可选加载学习排序 JSON 模型。
- `bas.learning`: 将在线击球过程整理成 shot 级训练样本。
- `bas.projection`: 投影 overlay 构建和 Qt 全屏窗口，支持实线/虚线画线样式。
- `bas.replay`: JSONL 结构化回放日志。

## 画线预览与模式

主界面左侧“画线模式”可在两种路线之间切换，切换会立即写入 `local_settings/user_settings.json`，运行中也会让下一帧规划生效。

- 规则模式：根据白球、目标球、袋口生成进袋候选；预览和投影会显示白色实线瞄准线、目标球到袋口虚线，以及白球碰撞后的分离方向虚线。
- 自由模式：根据白球附近的球杆边缘或 `cue_stick` 检测框估计出杆方向；预览和投影会用白色显示球杆拟合线、白球运动路径，并最多预测两次库边/球碰撞。撞到目标球时，会用白色虚线显示目标球后续传播方向。

如果识别框和 track 正常但没有路线，优先查看右侧“规划”状态：规则模式通常是没有合法进袋候选，自由模式会显示 `no_cue_ball`、`no_cue_stick` 或 `invalid_path` 等状态，便于现场排查。

## 配置

默认配置在 `configs/default.yaml`。如果要接入已导出的 YOLO 模型，把检测器配置改成：

```yaml
detector:
  backend: ultralytics
  model_path: C:/path/to/model.pt
  class_names: [cue, solid, stripe, black, cue_stick]
```

`Start_BAS.cmd` 不做依赖安装。若当前 `local_settings/user_settings.json` 或默认配置中的检测后端是 `ultralytics`，请确保已经完成一次性环境配置。手动检查可执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe -m bas doctor
```

`requirements-yolo.txt` 固定使用 PyTorch CUDA 12.8 轮子，已验证可驱动 RTX 5080。`python -m bas doctor` 会显示 `module:torch`、`torch.version.cuda` 和当前 CUDA 设备名。

实时运行时默认限制 YOLO 检测频率为 `10Hz`，并在中间帧复用上一轮检测/跟踪结果，以保持 1080p MJPG 采集和 UI 预览的帧率。需要更高检测密度时，可在设置页调整“检测间隔(帧)”和“检测频率上限(Hz)”。

UI 中的“设置”会保存到 `local_settings/user_settings.json`，该文件已被 `.gitignore` 忽略。设置页支持：

- 台球模型路径
- 类别文件路径
- 视频文件路径
- Nori SDK 目录
- 工业相机畸变矫正开关与 OpenCV 内参文件路径
- 工业相机曝光控制：自动曝光，或手动 `-10` 到 `0` 档
- `outline.json`
- `inline.json`
- `pocket.json`
- 相机标定文件
- 投影校正文件
- 默认投影设备和投影分辨率
- 学习排序模型路径、学习样本目录、学习排序权重
- 颗星公式位置、缩放、旋转、标签偏移等参数

主界面直接支持：

- 输入类型选择：`auto`、`nori`、`opencv`、`video`、`synthetic`
- 工业相机 ID 选择
- OpenCV 设备号
- 分辨率和帧率
- 初始化图形图像模块
- 开始/结束采集
- 开始/停止投影
- 校正投影仪
- 导出诊断快照

## 使用流程

1. 打开 UI 后先进入“设置”，确认模型、类别、几何 JSON、Nori SDK、畸变矫正文件、相机标定文件、投影校正文件和投影分辨率。
2. 点击“初始化图形图像模块”。新版会加载/校验检测器、几何文件、相机标定和投影校正；这一步不打开相机。
3. 点击“探测相机”，确认工业相机出现在列表中；需要固定曝光时，在“设置”里关闭自动曝光并选择 `-10` 到 `0` 档，当前已按旧版默认填为 `-5`。
4. 点击“开始采集”。如果启用了畸变矫正，采集帧会先按 OpenCV 内参进行实时去畸变，然后再进入检测、跟踪和规划。
5. 点击“开始投影”。当前 `local_settings/user_settings.json` 已迁入旧版颗星公式预设，并默认启用；即使暂时没有路线 overlay，也会显示颗星公式网格。
6. 点击“校正投影仪”会打开投影仪校正向导。向导可以显示 ChArUco 校正码、编码网格、当前校正多边形、对应点、局部残差箭头，并可导入 holdout JSON 做自动验收；重新生成或替换投影校正 JSON 后，再点“显示当前校正结果”即可重新加载预览。
7. 运行中如需排查问题，点击“导出诊断快照”。当前配置、模块状态、最新 frame/detection/track/state/plan/overlay 会写入 `local_settings/diagnostics/diagnostic_*.json`。

## 投影仪校准现场步骤

当前程序负责现场显示校正码、加载校正 JSON、显示校正结果和残差；结构光采集、bundle adjustment 和 holdout 验证数据生成仍由外部标定脚本完成。建议按以下流程操作：

1. 在“设置”中确认相机内参文件、投影校正文件、投影屏幕编号和投影分辨率。
2. 点击“校正投影仪”，先点“打开投影窗口”，再根据外部标定脚本需要选择“显示 ChArUco 校正码”或“显示编码网格”。
3. 在台面放置主标定板，采集不少于 20 个不同平移/俯仰姿态，确保靠近投影端和远离投影端都有覆盖。
4. 在六个袋口、两条长边中段、靠近投影端短边、远离投影端短边布置 10-14 张 A6/A7 小 ChArUco 纸板，采集局部精修和 holdout 数据。
5. 外部脚本生成 `projection_calibration.json` 后，回到向导点“显示当前校正结果”，检查桌面多边形、对应点编号和投影位置是否对齐。
6. 点“显示残差箭头”，检查局部误差方向和覆盖密度。袋口、近端、远端都应有控制点；若某一区域箭头明显同向偏移，需要补采该区域。
7. 以独立 holdout 验收：图像重投影 `mean < 0.20px, P95 < 0.35px`；台面毫米误差 MVP 阶段 `median < 1.5mm, P95 < 3.0mm`，正式交付建议 `median < 1.0mm, P95 < 2.0mm`；任一袋口区域 P95 应小于 `2.5mm`。

Holdout JSON 可以是数组，也可以是带 `samples` 字段的对象。每个样本建议包含：

```json
{
  "samples": [
    {
      "camera_px": [123.4, 567.8],
      "projector_px": [456.7, 321.0],
      "world_mm": [1000.0, 500.0],
      "zone": "middle",
      "distance_cm": 120.0
    }
  ]
}
```

`camera_px + projector_px` 用于图像重投影误差；`world_mm + projector_px` 或 `world_mm + camera_px` 用于台面毫米误差；`zone` 用于分区 P95；`distance_cm` 用于误差随距离梯度。UI 向导里的“验证 Holdout”和命令行 `verify-calib` 使用同一套计算逻辑。

## 状态机人工介入

主界面右侧“状态机人工介入”用于现场纠偏：

- “强制状态”：将状态机切到 `STABLE_IDLE/PRE_SHOT_ARMED/SHOT_ACTIVE/SETTLING/TURN_RESOLVE/ANOMALY_RECOVERY` 中的指定状态，并写入 `OPERATOR_FORCE_PHASE` 事件。
- “冻结状态机”：停止自动推进，保持当前相位和最近稳定布局；检测、跟踪、回放仍继续记录。再次点击可恢复自动推进。
- “确认当前布局稳定”：把当前轨迹快照保存为稳定布局，并强制回到 `STABLE_IDLE`。
- “重置状态机”：清空状态机计数器、异常计数、进洞复核标记和事件列表。
- “清除复核标记”：清掉已进洞/异常复核缓存，用于人工确认误判后继续当前局。

右侧“模块状态”会显示采集、检测、跟踪、状态机、规划、投影、回放、标定的运行状态和关键细节，例如检测延迟、轨迹数、候选数、校正有效性和回放会话。

状态机现在会输出以下可排查事件：

- `BALL_COLLISION_CANDIDATE`: 球心距离接近接触阈值，且存在闭合速度或航向变化。
- `RAIL_COLLISION_CANDIDATE`: 球靠近库边并存在法向速度或反弹迹象。
- `SHOT_START_VOTED`: 球杆尖端邻域、母球速度跃迁、母球加速度峰值三项中至少两项成立。
- `POT_PROBABLE`: 球在口袋 funnel 范围内消失，且最后轨迹趋势合理。
- `BALL_DISAPPEARED`: 球在击球/收敛/结算阶段消失，但不满足口袋趋势，需复核。
- `OPERATOR_*`: 人工强制状态、冻结、确认稳定布局、清除复核等操作。

事件列表会显示关键字段；鼠标悬停事件项可查看完整 payload，回放 JSONL 中也会记录这些事件。

如果要使用工业相机 SDK：

```yaml
camera:
  backend: nori
  nori_sdk_root: C:/path/to/Nori_Xvision_Development_Kit_Ver10.00.06_Windows
  exposure_auto: false
  exposure_level: -5
  distortion_correction_enabled: true
  distortion_correction_file: C:/path/to/intrinsics_opencv.yaml
```

## Git 管理策略

仓库只管理核心代码、配置模板和文档。模型、SDK、旧版历史资料、日志、回放、截图、视频和本机标定文件都在 `.gitignore` 中排除。
