# Billiards Assistance System

这是一个从头重构的台球智能辅助系统骨架，围绕 `Capture -> Detect -> Track -> State -> Plan -> Projection -> Replay` 这条链路拆分模块。旧项目只作为接口和参数格式参考使用，尤其是 Nori 工业相机 SDK 与 OpenCV 标定文件读取。

当前版本刻意没有实现进洞路线学习系统，也没有采集学习数据；在线规划只包含几何候选和快速物理筛选，学习排序层保留为禁用接口。

## 快速运行

双击根目录的 `Start_BAS.cmd` 可以直接打开桌面控制台 UI。默认不再启用 synthetic 虚拟画面；真实使用时在 UI 里选择 `auto`、`nori`、`opencv` 或 `video`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m bas smoke-run --frames 90
```

常用命令：

```powershell
.\.venv\Scripts\python.exe -m bas probe-cameras
.\.venv\Scripts\python.exe -m bas inspect-calib
.\.venv\Scripts\python.exe -m bas ui
.\.venv\Scripts\python.exe -m bas run --headless --max-frames 300
.\.venv\Scripts\python.exe -m bas run
```

## 模块说明

- `bas.capture`: OpenCV、视频文件、合成源、Nori SDK 采集。Nori SDK 只筛选 MJPG。
- `bas.calibration`: OpenCV YAML 内参、投影单应矩阵、局部残差场、台面毫米坐标映射。
- `bas.perception`: 调试颜色检测器和 Ultralytics YOLO 模型入口。
- `bas.tracking`: 双阈值时序跟踪、短时遮挡保留、速度估计。
- `bas.state`: `STABLE_IDLE/PRE_SHOT_ARMED/SHOT_ACTIVE/SETTLING/TURN_RESOLVE/ANOMALY_RECOVERY` 宏状态。
- `bas.planning`: 几何候选 + 快速物理筛选；学习排序禁用占位。
- `bas.projection`: 投影 overlay 构建和 Qt 全屏窗口。
- `bas.replay`: JSONL 结构化回放日志。

## 配置

默认配置在 `configs/default.yaml`。如果要接入已导出的 YOLO 模型，把检测器配置改成：

```yaml
detector:
  backend: ultralytics
  model_path: C:/path/to/model.pt
  class_names: [cue, solid, stripe, black, cue_stick]
```

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

## 使用流程

1. 打开 UI 后先进入“设置”，确认模型、类别、几何 JSON、Nori SDK、畸变矫正文件、相机标定文件、投影校正文件和投影分辨率。
2. 点击“初始化图形图像模块”。新版会加载/校验检测器、几何文件、相机标定和投影校正；这一步不打开相机。
3. 点击“探测相机”，确认工业相机出现在列表中；需要固定曝光时，在“设置”里关闭自动曝光并选择 `-10` 到 `0` 档，当前已按旧版默认填为 `-5`。
4. 点击“开始采集”。如果启用了畸变矫正，采集帧会先按 OpenCV 内参进行实时去畸变，然后再进入检测、跟踪和规划。
5. 点击“开始投影”。当前 `local_settings/user_settings.json` 已迁入旧版颗星公式预设，并默认启用；即使暂时没有路线 overlay，也会显示颗星公式网格。
6. 点击“校正投影仪”会加载 `projection_calibration.json`，打开投影窗口并显示当前校正多边形和对应点，用于确认投影映射是否对齐。重新生成或替换投影校正 JSON 后，再点一次即可重新加载预览。

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
