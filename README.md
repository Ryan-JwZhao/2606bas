# BAS 本地说明

## 简介
这是一个台球辅助系统，包含实时采集、检测、跟踪、状态机、路线规划、前端预览和投影显示。

本次更新重点补了两类能力：

- 修正 `inline` 与 `pocket` 的组合几何显示与边界生成逻辑，避免每个袋口被单独闭合成“月牙”。
- 保留原有实体校正板流程，同时新增“一键联动校正”，用多图样混合校正投影仪，其中以 ChArUco 为主、编码网格为辅。

## 几何与袋口说明

- `outline.json` 表示台球桌外轮廓矩形范围。
- `inline.json` 表示台面内沿线段。
- `pocket.json` 表示六个袋口的开口曲线。
- 程序现在会把 `inline + pocket` 自动拼成一条真实的组合闭合边界，用于后续标定与重点区域联动。
- 前端参考线中，`pocket` 会按开口曲线显示，不再单独闭合成封闭月牙。

## 一键联动校正

### 作用

- 自动播放总览网格、全域 ChArUco、中心细化和六个袋口重点图样。
- 让投影范围完整覆盖 `outline` 对应的矩形区域。
- 对 `inline` 与 `pocket` 相接的关键区域做重点采样。
- 自动识别 ChArUco 角点，自动求解 `camera_px -> projector_px` 的投影校正。
- 自动保存结果，并可再次加载与可视化显示。

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

- 默认联动校正结果会保存到当前 `calibration.projection_file`。
- 如果没有配置投影校正输出路径，程序会回落到 `local_settings/projection_calibration_linked.json`。
- `local_settings/`、日志、回放、临时目录和大文件应保持在 `.gitignore` 中，避免提交运行产物。
