# BAS 本地说明

## 简介
这是一个台球辅助系统，包含实时采集、检测、跟踪、状态机、路线规划、前端预览和投影显示。

本次更新重点补了两类能力：

- 修正 `inline` 与 `pocket` 的组合几何显示与边界生成逻辑，避免每个袋口被单独闭合成“月牙”。
- 保留原有实体校正板流程，同时新增“一键联动校正”，用多图样混合校正投影仪，其中以 ChArUco 为主、编码网格为辅。
- 新增运行时“投影调试模式”，用于现场肉眼检查投影映射是否与台面几何和识别球位置一致。

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
