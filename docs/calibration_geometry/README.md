# 独立几何与球心精修模块

## 简介

本模块将原先串联的“相机像素 → 投影仪像素 → 台面坐标”重构为两条互不依赖的二维标定链路：

```text
相机观测 → 台面毫米坐标
台面毫米坐标 → 投影仪像素
```

系统不读取也不使用相机外参。相机内参仅在启用镜头畸变校正时用于像素去畸变和反向畸变。

球心位置使用独立的“相机球心像素 → 台面目标坐标”平滑映射。新模型会在仿射、二阶多项式和正则化高斯残差之间做空间交叉验证，只有复杂模型的 P95 明显更好时才会启用，避免旧 KNN 在采样点上精确、在新位置上漂移的问题。

## 模块结构

- `bas/calibration/geometry.py`
  - 相机台面映射
  - 球心台面映射
  - 台面投影映射
  - 平滑残差和数值反演
- `bas/perception/ball_geometry.py`
  - 分割轮廓亚像素椭圆拟合
  - 无分割掩膜时的台布背景差分
  - 球心、半径和几何质量评估
- `bas/calibration/service.py`
  - 保留旧调用方式的兼容外壳
  - 所有实际坐标变换委托给独立几何模块

## 使用方式

应用程序启动方式不变：

```powershell
.\Start_BAS.cmd
```

查看当前独立几何模型及交叉验证指标：

```powershell
.\.venv\Scripts\python.exe -m bas inspect-calib
```

输出中的 `geometry_model` 包含：

- `geometry_mode`：固定为 `independent_2d`
- `camera_extrinsics_used`：固定为 `0`
- `projector_residual_model`：投影残差模型
- `projector_residual_cv_p95_px`：投影残差空间交叉验证 P95
- `ball_map_model`：球心映射模型
- `ball_map_cv_p95_mm`：球心空间交叉验证 P95

## 重新采集球心补偿

继续使用“工程球体补偿采样”向导即可。新文件会额外保存：

- `target_table_mm`
- `sample_weights`
- 每个采样点的置信度和稳定性

建议：

1. 相机、投影仪和球桌固定后一次性完成平面标定与球心采样。
2. 每个位置等球体完全静止后再采集。
3. 覆盖四角、长边、短边和中心区域。
4. 正式采样不少于 30 点；验收点必须独立于拟合点。
5. 不要混用不同日期、不同分辨率或不同畸变开关下的标定文件。

Holdout 样本如用于验证球心，应增加：

```json
{
  "camera_px": [960.0, 540.0],
  "world_mm": [1270.0, 635.0],
  "kind": "ball_center",
  "zone": "center"
}
```

运行：

```powershell
.\.venv\Scripts\python.exe -m bas verify-calib C:\path\to\holdout.json
```

## 球心检测行为

球检测仍通过 `Detection.center` 和 `Detection.radius_px` 使用，不需要调用方处理轮廓：

1. YOLO 提供分割掩膜时，优先拟合亚像素椭圆。
2. 只有检测框时，在局部区域估计台布背景并提取球体轮廓。
3. 轮廓不可靠时自动退回检测框中心。
4. 同一帧内使用标准球直径一致性检查尺寸离群值。
5. 低置信度且尺寸严重离群的候选不会进入跟踪器。
6. 几何质量会参与轨迹质量计算。

## 本地验证

核心测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_independent_geometry.py tests\test_ball_geometry.py tests\test_calibration.py -q
```

全量测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

若桌面程序正在运行，Windows 单实例互斥锁测试可能失败；这与几何模块无关，关闭正在运行的 BAS 后再执行该测试。
